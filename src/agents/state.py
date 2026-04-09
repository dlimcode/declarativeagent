"""Shared state model for custom agents."""

from typing import Optional

from pydantic import BaseModel

from tau2.data_model.message import APICompatibleMessage, SystemMessage


class AgentState(BaseModel):
    """State shared by declarative and imperative agents.

    Extends LLMAgentState with optional phase tracking for the imperative agent.
    """

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    phase: Optional[str] = None
    phase_retries: int = 0
