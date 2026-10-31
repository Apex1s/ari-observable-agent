"""The ladder: seven stages from a bare API wrapper to an observable agent.

Each stage adds exactly one capability to the previous one, so running the same question (or the
eval suite) across stages shows what each layer buys.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageConfig:
    key: str
    title: str
    adds: str
    persona: bool = False
    rag: bool = False
    tools: bool = False
    memory: bool = False
    guardrails: bool = False
    tracing: bool = False


_LADDER = [
    ("v0", "API wrapper", "A raw call to the model. No identity, no knowledge of EMINES: it answers like a generic chatbot.",
     {}),
    ("v1", "Persona", "A system prompt: Ari's identity, audience, spoken-answer style and language rules.",
     {"persona": True}),
    ("v2", "Grounded (RAG)", "Retrieval over EMINES documents; answers must cite numbered sources or admit not knowing.",
     {"rag": True}),
    ("v3", "Tools (agent)", "A router picks a tool: campus directory, timetable, robot navigation, clock, calculator.",
     {"tools": True}),
    ("v4", "Memory", "A sliding window of recent turns per session, so follow-up questions work.",
     {"memory": True}),
    ("v5", "Guardrails", "Input checks (prompt injection, length), refusal when nothing is grounded, citation validation.",
     {"guardrails": True}),
    ("v6", "Observable", "Every request is traced span by span (latency, tokens, retrieval scores, tool calls) "
     "into SQLite, with a live dashboard and user feedback.",
     {"tracing": True}),
]


def _build() -> dict[str, StageConfig]:
    stages, flags = {}, {}
    for key, title, adds, new_flags in _LADDER:
        flags = {**flags, **new_flags}
        stages[key] = StageConfig(key, title, adds, **flags)
    return stages


STAGES: dict[str, StageConfig] = _build()
