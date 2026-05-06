"""Drop-in replacement for tau2's generate() with prompt caching + reasoning passthrough.

For Anthropic/Claude and DeepSeek models, wraps the system message content in a
cache_control block so every turn after the first reuses the cached KV state
(~90% discount on cached input tokens).  Falls back to the plain format for
all other providers.

For DeepSeek thinking/reasoning models (e.g. deepseek-v4-pro), the API requires
reasoning_content from each assistant turn to be echoed back in the next request.
This is extracted from raw_data stored on AssistantMessage and re-injected into
the litellm message list before each call.
"""

import json
import time
from typing import Optional

import litellm
from litellm import completion

from tau2.config import DEFAULT_MAX_RETRIES
from tau2.data_model.message import AssistantMessage, Message, SystemMessage, ToolCall
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import (
    get_response_cost,
    get_response_usage,
    to_litellm_messages,
    validate_message_history,
)

litellm.modify_params = True


def _supports_cache_control(model: str) -> bool:
    m = model.lower()
    return "claude" in m or m.startswith("anthropic/") or "deepseek" in m


def _is_deepseek(model: str) -> bool:
    return "deepseek" in model.lower()


def generate_cached(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    **kwargs,
) -> AssistantMessage:
    validate_message_history(messages)
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    use_cache = _supports_cache_control(model)

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    litellm_messages: list[dict] = []

    for sm in system_msgs:
        if use_cache:
            litellm_messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sm.content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            })
        else:
            litellm_messages.append({"role": "system", "content": sm.content})

    converted = to_litellm_messages(other_msgs)

    # DeepSeek thinking models require reasoning_content echoed back each turn.
    # Walk assistant messages in order and inject from raw_data when present.
    if _is_deepseek(model):
        assistant_tau2 = (m for m in other_msgs if isinstance(m, AssistantMessage))
        for lm in converted:
            if lm.get("role") == "assistant":
                tau2_msg = next(assistant_tau2, None)
                if tau2_msg and tau2_msg.raw_data:
                    choices = tau2_msg.raw_data.get("choices", [])
                    if choices:
                        rc = choices[0].get("message", {}).get("reasoning_content")
                        if rc:
                            lm["reasoning_content"] = rc

    litellm_messages.extend(converted)

    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    start = time.perf_counter()
    response = completion(
        model=model,
        messages=litellm_messages,
        tools=tools_schema,
        tool_choice=tool_choice,
        **kwargs,
    )
    generation_time_seconds = time.perf_counter() - start

    cost = get_response_cost(response)
    usage = get_response_usage(response)

    choice = response.choices[0]
    content = choice.message.content
    raw_tool_calls = choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments),
        )
        for tc in raw_tool_calls
    ] or None

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
        generation_time_seconds=generation_time_seconds,
    )
