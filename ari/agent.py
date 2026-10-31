"""The agent: one request pipeline whose steps switch on stage by stage.

    guard.input -> route/tool calls -> retrieve -> (scope check) -> llm.answer -> guard.output -> memory
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from . import guardrails, prompts
from .llm import LLM, LLMResponse, Message
from .memory import ConversationMemory
from .rag import Hit, VectorStore
from .stages import StageConfig
from .tools import ToolRegistry
from .tracing import NullTrace, Trace, Tracer

MAX_TOOL_STEPS = 2


@dataclass
class AgentReply:
    text: str
    stage: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    issues: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    trace_id: str | None = None
    trace: dict | None = None


class Agent:
    def __init__(self, stage: StageConfig, llm: LLM, store: VectorStore, tools: ToolRegistry,
                 memory: ConversationMemory, tracer: Tracer, top_k: int = 3, min_score: float = 0.5):
        self.stage = stage
        self.llm = llm
        self.store = store
        self.tools = tools
        self.memory = memory
        self.tracer = tracer
        self.top_k = top_k
        self.min_score = min_score

    def chat(self, message: str, session_id: str = "default") -> AgentReply:
        trace = self.tracer.start_trace(session_id, self.stage.key, message) if self.stage.tracing else NullTrace()
        t0 = time.perf_counter()
        try:
            reply = self._run(message, session_id, trace)
        except Exception as exc:
            trace.finish(output="", error=f"{type(exc).__name__}: {exc}")
            raise
        reply.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        reply.trace_id = trace.trace_id
        reply.trace = trace.finish(output=reply.text, refused=reply.refused, refusal_reason=reply.refusal_reason,
                                   tool_calls=len(reply.tool_calls))
        return reply

    # -- pipeline ---------------------------------------------------------------------------------

    def _run(self, message: str, session_id: str, trace: Trace | NullTrace) -> AgentReply:
        cfg = self.stage
        memory_key = f"{session_id}:{cfg.key}"

        if cfg.guardrails:
            with trace.span("guard.input") as span:
                verdict = guardrails.check_input(message)
                span.set(allowed=verdict.allowed, reason=verdict.reason)
            if not verdict.allowed:
                return AgentReply(verdict.message, cfg.key, refused=True, refusal_reason=verdict.reason)

        history = self.memory.history(memory_key) if cfg.memory else []
        tool_calls = self._use_tools(message, history, trace) if cfg.tools else []

        hits: list[Hit] = []
        if cfg.rag:
            with trace.span("retrieve", top_k=self.top_k, min_score=self.min_score) as span:
                query = self._retrieval_query(message, history)
                candidates = self.store.search(query, self.top_k)
                hits = [h for h in candidates if h.score >= self.min_score]
                span.set(query=query, kept=len(hits),
                         candidates=[{"chunk": h.chunk.id, "score": round(h.score, 3)} for h in candidates])

        answerable_from_history = bool(history) and guardrails.refers_to_conversation(message)
        if (cfg.guardrails and cfg.rag and not hits and not tool_calls
                and not guardrails.is_smalltalk(message) and not answerable_from_history):
            with trace.span("guard.scope", reason="no_grounding"):
                pass
            reply = AgentReply(guardrails.REFUSALS["no_grounding"], cfg.key, refused=True, refusal_reason="no_grounding")
            if cfg.memory:
                self.memory.append(memory_key, message, reply.text)
            return reply

        response = self._llm(self._build_messages(message, history, hits, tool_calls), trace, "answer")
        text, issues = response.text.strip(), []
        if cfg.guardrails:
            with trace.span("guard.output") as span:
                text, issues = guardrails.check_output(text, len(hits))
                if tool_calls:  # answers built on live tool data need no document citation
                    issues = [i for i in issues if i != "uncited"]
                span.set(issues=issues)

        cited = set(guardrails.cited_numbers(text))
        citations = [{"n": i, "source": h.chunk.source, "section": h.chunk.heading, "score": round(h.score, 3),
                      "cited": i in cited} for i, h in enumerate(hits, 1)]
        if cfg.memory:
            self.memory.append(memory_key, message, text)
        return AgentReply(text, cfg.key, citations=citations, tool_calls=tool_calls, issues=issues)

    def _use_tools(self, message: str, history: list[Message], trace: Trace | NullTrace) -> list[dict]:
        calls: list[dict] = []
        for step in range(MAX_TOOL_STEPS):
            system = prompts.ROUTER.format(tools=self.tools.describe())
            if calls:
                system += "\n\nTool results so far:\n" + "\n".join(f"- {c['tool']} -> {c['result']}" for c in calls)
            with trace.span("route", step=step) as span:
                response = self._llm([{"role": "system", "content": system}, *history, {"role": "user", "content": message}],
                                     trace, "route", json_mode=True, temperature=0.0)
                decision = self._parse_decision(response.text)
                span.set(decision=decision, raw=response.text[:300])
            name = decision.get("tool")
            if not name or name not in self.tools.tools or any(c["tool"] == name for c in calls):
                break
            args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
            with trace.span(f"tool.{name}", args=args) as span:
                result = self.tools.run(name, args)
                span.set(result=result[:500])
                if result.startswith("ERROR"):
                    span.status = "error"
            calls.append({"tool": name, "args": args, "result": result})
        return calls

    @staticmethod
    def _parse_decision(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {"tool": None, "parse_error": True}
        try:
            decision = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"tool": None, "parse_error": True}
        return decision if isinstance(decision, dict) else {"tool": None, "parse_error": True}

    @staticmethod
    def _retrieval_query(message: str, history: list[Message]) -> str:
        """Short follow-ups ("and on Saturday?") are retrieved together with the previous question."""
        previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{previous} {message}" if previous and len(message.split()) < 6 else message

    def _build_messages(self, message: str, history: list[Message], hits: list[Hit], tool_calls: list[dict]) -> list[Message]:
        cfg = self.stage
        system: list[str] = []
        if cfg.persona:
            system.append(prompts.PERSONA)
        if cfg.rag:
            if hits:
                passages = "\n\n".join(f"[{i}] ({h.chunk.source}, {h.chunk.heading})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
                system.append(f"{prompts.GROUNDING}\n\nCONTEXT:\n{passages}")
            else:
                system.append(prompts.NO_CONTEXT)
        if tool_calls:
            system.append(prompts.TOOL_RESULTS + "\nTOOL RESULTS:\n" + "\n".join(
                f"- {c['tool']}({json.dumps(c['args'], ensure_ascii=False)}) -> {c['result']}" for c in tool_calls))
        messages: list[Message] = [{"role": "system", "content": "\n\n".join(system)}] if system else []
        return messages + history + [{"role": "user", "content": message}]

    def _llm(self, messages: list[Message], trace: Trace | NullTrace, purpose: str,
             json_mode: bool = False, temperature: float = 0.2) -> LLMResponse:
        with trace.span(f"llm.{purpose}", model=self.llm.model, messages=len(messages)) as span:
            response = self.llm.chat(messages, json_mode=json_mode, temperature=temperature)
            span.set(prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens)
            trace.add_usage(response.prompt_tokens, response.completion_tokens, response.model)
        return response
