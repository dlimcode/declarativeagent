"""Agent registration: factory functions + register_all().

Call register_all() before using tau2 CLI to make custom agents
available via --agent declarative_agent / --agent imperative_agent.
"""

from tau2.registry import registry

from agents.declarative_agent import DeclarativeAgent
from agents.imperative_agent import ImperativeAgent


def create_declarative_agent(tools, domain_policy, **kwargs):
    """Factory for DeclarativeAgent."""
    return DeclarativeAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def create_imperative_agent(tools, domain_policy, **kwargs):
    """Factory for ImperativeAgent."""
    return ImperativeAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def register_all():
    """Register all custom agents with the tau2 registry."""
    registry.register_agent_factory(
        factory=create_declarative_agent,
        name="declarative_agent",
    )
    registry.register_agent_factory(
        factory=create_imperative_agent,
        name="imperative_agent",
    )
