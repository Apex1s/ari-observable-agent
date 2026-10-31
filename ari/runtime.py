"""Wires settings, model, index, tools, memory and tracer together; hands out one agent per stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .agent import Agent
from .config import Settings, load_settings
from .llm import LLM, make_embed_client, make_llm
from .memory import ConversationMemory
from .rag import VectorStore, load_chunks, make_embedder
from .stages import STAGES
from .tools import ToolRegistry
from .tracing import Tracer

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    llm: LLM
    store: VectorStore
    tools: ToolRegistry
    memory: ConversationMemory
    tracer: Tracer
    min_score: float
    _agents: dict[str, Agent] = field(default_factory=dict)

    def agent(self, stage: str) -> Agent:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; choose one of {', '.join(STAGES)}")
        if stage not in self._agents:
            self._agents[stage] = Agent(STAGES[stage], self.llm, self.store, self.tools, self.memory, self.tracer,
                                        top_k=self.settings.top_k, min_score=self.min_score)
        return self._agents[stage]


def build_runtime(settings: Settings | None = None, **overrides) -> Runtime:
    settings = settings or load_settings(**overrides)
    llm = make_llm(settings)
    embedder = make_embedder(llm if settings.embed_provider == settings.provider else make_embed_client(settings))
    index_dir = settings.index_dir / embedder.id.replace(":", "_").replace("/", "_")
    store = VectorStore(embedder, index_dir).build_or_load(load_chunks(settings.knowledge_dir))
    min_score = settings.min_retrieval_score if settings.min_retrieval_score is not None else embedder.default_min_score
    log.info("Ari ready: chat=%s:%s embedder=%s chunks=%d min_score=%.2f",
             settings.provider, llm.model, embedder.id, len(store.chunks), min_score)
    return Runtime(settings, llm, store, ToolRegistry(settings.campus_file, settings.schedule_file),
                   ConversationMemory(settings.memory_turns), Tracer(settings.trace_db), min_score)
