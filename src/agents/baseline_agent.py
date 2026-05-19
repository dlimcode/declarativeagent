"""Baseline agent: τ-Bench LLMAgent + prompt caching, no skill files, no orchestration.

Mirrors tau3-bench's headline baseline (Section 5 of the paper): the LLM is given
the default tau2 system prompt (instructions + domain policy) plus the standard
banking_knowledge tool set (14 permanent tools + KB_search + the discoverable-tool
mechanism). Workflow, tool selection, ordering, verification — all handled by the
LLM with no programmatic guidance.

Identical to tau2's built-in LLMAgent except generate() is swapped for
cached_generate, which adds Anthropic/DeepSeek prompt caching on the system prompt.
"""

from typing import List, Optional

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from agents.cached_generate import generate_cached as generate
from agents.state import AgentState

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


class BaselineAgent(LLMConfigMixin, HalfDuplexAgent[AgentState]):
    """Default τ-Bench LLM agent with prompt caching. No skill files, no state machine."""

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

    @property
    def system_prompt(self) -> str:
        return (
            f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n"
            f"<policy>\n{self.domain_policy}\n</policy>"
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AgentState:
        if message_history is None:
            message_history = []
        return AgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio messages not supported.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

        state.messages.append(assistant_message)
        return assistant_message, state
