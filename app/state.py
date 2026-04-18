from typing import TypedDict


class MemoryTurn(TypedDict, total=False):
    user: str
    assistant: str
    # clarify = last assistant was a clarifying question; next user message skips intent LLM
    mode: str


class GraphState(TypedDict, total=False):
    prompt: str
    chat_history: list[MemoryTurn]
    intent: str
    nl_response: str
    route: str
    needs_clarification: bool
    clarification_question: str
    plan: str
    raw_output: str
    bindings: dict[str, str]
    validation_error: str
    attempts: int
