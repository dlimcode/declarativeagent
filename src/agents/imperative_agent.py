"""Imperative agent: 8-phase state machine controlling workflow.

Python code determines which phase the conversation is in, restricts
tools to those relevant for the phase, and constructs phase-specific
prompts. The LLM handles NLU, argument generation, and NLG within
each phase.
"""

from typing import List, Optional

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from agents.state import AgentState

# Phase definitions: goal, allowed tools, expected output type
PHASES = {
    "GREETING": {
        "goal": (
            "Understand what the customer needs. Ask clarifying questions "
            "if their request is unclear."
        ),
        "allowed_tools": [],
        "expect": "text",
    },
    "IDENTIFICATION": {
        "goal": (
            "Identify the customer. Ask for their user ID, name, or email, "
            "then look them up."
        ),
        "allowed_tools": [
            "get_user_information_by_id",
            "get_user_information_by_name",
            "get_user_information_by_email",
        ],
        "expect": "tool_call",
    },
    "VERIFICATION": {
        "goal": (
            "Verify the customer's identity. Use log_verification to "
            "record the verification."
        ),
        "allowed_tools": ["log_verification"],
        "expect": "tool_call",
    },
    "KB_SEARCH": {
        "goal": (
            "Search the knowledge base for the procedure relevant to the "
            "customer's request. Extract the exact tool name "
            "(format: tool_name_NNNN) from the search results."
        ),
        "allowed_tools": ["KB_search"],
        "expect": "tool_call",
    },
    "TOOL_DISCOVERY": {
        "goal": (
            "Unlock the discoverable tool found in the KB search. "
            "Use unlock_discoverable_agent_tool with the exact tool name."
        ),
        "allowed_tools": ["unlock_discoverable_agent_tool"],
        "expect": "tool_call",
    },
    "EXECUTION": {
        "goal": (
            "Execute the customer's request using the appropriate tools. "
            "Use call_discoverable_agent_tool with the correct arguments "
            "as a JSON string. You may also use standard tools if needed."
        ),
        # unlock included for golden_retrieval where KB_SEARCH is skipped
        # All standard + discoverable tools allowed in EXECUTION
        "allowed_tools": [
            "unlock_discoverable_agent_tool",
            "call_discoverable_agent_tool",
            "list_discoverable_agent_tools",
            "give_discoverable_user_tool",
            "transfer_to_human_agents",
            "get_user_information_by_id",
            "get_user_information_by_name",
            "get_user_information_by_email",
            "change_user_email",
            "get_current_time",
            "get_referrals_by_user",
            "get_credit_card_transactions_by_user",
            "get_credit_card_accounts_by_user",
            "log_verification",
            "KB_search",
        ],
        "expect": "tool_call",
    },
    "CONFIRMATION": {
        "goal": (
            "Communicate the result to the customer. Summarize what was "
            "done, confirm the outcome, and ask if they need anything else."
        ),
        "allowed_tools": [],
        "expect": "text",
    },
    "COMPLETE": {
        "goal": (
            "The task is complete. Say goodbye if the customer has no "
            "more questions."
        ),
        "allowed_tools": ["transfer_to_human_agents"],
        "expect": "text",
    },
}

# Identical to tau2-bench's LLMAgent AGENT_INSTRUCTION
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


class ImperativeAgent(LLMConfigMixin, HalfDuplexAgent[AgentState]):
    """Agent with Python-controlled workflow phases.

    The state machine determines which phase the conversation is in,
    restricts tools to those relevant for the phase, and constructs
    phase-specific prompts. The LLM handles NLU, argument generation,
    and NLG within each phase.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.all_tools = {tool.name: tool for tool in tools}

    @property
    def base_system_prompt(self) -> str:
        return f"""
<instructions>
{AGENT_INSTRUCTION}
</instructions>
<policy>
{self.domain_policy}
</policy>
""".strip()

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AgentState:
        if message_history is None:
            message_history = []
        return AgentState(
            system_messages=[
                SystemMessage(role="system", content=self.base_system_prompt)
            ],
            messages=message_history,
            phase="GREETING",
            phase_retries=0,
        )

    def _get_tools_for_phase(self, phase: str) -> list[Tool]:
        """Return only the tools allowed in the current phase."""
        allowed = PHASES[phase]["allowed_tools"]
        return [self.all_tools[name] for name in allowed if name in self.all_tools]

    def _build_phase_instruction(self, phase: str) -> str:
        """Build a phase-specific instruction block."""
        phase_info = PHASES[phase]
        allowed = [t for t in phase_info["allowed_tools"] if t in self.all_tools]
        return (
            f"\n<current_phase>\n"
            f"You are in the {phase} phase.\n"
            f"Your goal: {phase_info['goal']}\n"
            f"Available tools: {', '.join(allowed) if allowed else 'none (respond with text)'}\n"
            f"</current_phase>"
        )

    def _determine_phase(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> str:
        """Determine current phase based on conversation state.

        Forward-only flow with skip logic for golden_retrieval
        (no KB_search tool available). Handles follow-up requests
        by cycling back from COMPLETE to IDENTIFICATION.
        """
        current = state.phase

        if current == "GREETING":
            # Stay in GREETING on the first user message so the agent
            # can acknowledge the request before moving on. Transition
            # to IDENTIFICATION only after the agent has responded
            # (i.e., the last message in history is an AssistantMessage).
            if self._last_assistant_was_text(state):
                return "IDENTIFICATION"
            return "GREETING"

        if current == "IDENTIFICATION":
            if self._last_was_successful_tool_result(state):
                return "VERIFICATION"
            return "IDENTIFICATION"

        if current == "VERIFICATION":
            if self._last_was_successful_tool_result(state):
                if "KB_search" in self.all_tools:
                    return "KB_SEARCH"
                return "EXECUTION"
            return "VERIFICATION"

        if current == "KB_SEARCH":
            if self._last_was_successful_tool_result(state):
                return "TOOL_DISCOVERY"
            if state.phase_retries < 2:
                return "KB_SEARCH"
            return "EXECUTION"

        if current == "TOOL_DISCOVERY":
            if self._last_was_successful_tool_result(state):
                return "EXECUTION"
            return "TOOL_DISCOVERY"

        if current == "EXECUTION":
            if self._last_assistant_was_text(state):
                return "CONFIRMATION"
            if self._last_was_successful_tool_result(state):
                return "EXECUTION"
            return "EXECUTION"

        if current == "CONFIRMATION":
            # If the user responds with a new request, cycle back
            if isinstance(message, UserMessage):
                return "IDENTIFICATION"
            return "COMPLETE"

        if current == "COMPLETE":
            # Handle follow-up requests after task completion
            if isinstance(message, UserMessage):
                return "IDENTIFICATION"
            return "COMPLETE"

        return current

    def _last_was_successful_tool_result(self, state: AgentState) -> bool:
        """Check if the last message was a successful tool result."""
        if not state.messages:
            return False
        last = state.messages[-1]
        if isinstance(last, ToolMessage):
            return not getattr(last, "error", False)
        if isinstance(last, MultiToolMessage):
            return all(
                not getattr(tm, "error", False) for tm in last.tool_messages
            )
        return False

    def _last_assistant_was_text(self, state: AgentState) -> bool:
        """Check if the last assistant message was text (not tool calls)."""
        for msg in reversed(state.messages):
            if isinstance(msg, AssistantMessage):
                return msg.has_text_content() and not msg.is_tool_call()
        return False

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio messages not supported.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Determine phase (THE IMPERATIVE CONTROL LOGIC)
        new_phase = self._determine_phase(message, state)
        if new_phase != state.phase:
            state.phase = new_phase
            state.phase_retries = 0
        else:
            state.phase_retries += 1

        # Get phase-specific tools and instruction
        phase_tools = self._get_tools_for_phase(state.phase)
        phase_instruction = self._build_phase_instruction(state.phase)

        # Build messages with phase instruction injected
        phase_system = SystemMessage(
            role="system",
            content=self.base_system_prompt + phase_instruction,
        )
        messages = [phase_system] + state.messages

        # Phases expecting text get no tools so the LLM produces text.
        # Phases expecting tool calls get only their allowed tools (never
        # fall back to all tools — empty list means no tools available).
        expects_text = PHASES[state.phase]["expect"] == "text"
        tools_arg = None if expects_text else phase_tools

        assistant_message = generate(
            model=self.llm,
            tools=tools_arg,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

        state.messages.append(assistant_message)
        return assistant_message, state
