"""Observability (stage v6): every request becomes a trace of timed spans, stored in SQLite.

A span is one step: input guard, tool routing, a tool call, retrieval, an LLM call, output guard.
Each span records its duration and attributes (scores, tokens, chosen tool, errors), so a wrong
answer can be traced to the exact step that caused it. The schema maps one-to-one onto
OpenTelemetry spans if you later export to Langfuse, Phoenix or Jaeger.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .guardrails import redact_pii

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY, session_id TEXT, stage TEXT, started_at TEXT, duration_ms REAL,
    input TEXT, output TEXT, status TEXT, refused INTEGER, refusal_reason TEXT, tool_calls INTEGER,
    prompt_tokens INTEGER, completion_tokens INTEGER, model TEXT, feedback INTEGER
);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, name TEXT, start_ms REAL,
    duration_ms REAL, status TEXT, attributes TEXT
);
CREATE INDEX IF NOT EXISTS spans_by_trace ON spans(trace_id);
"""


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    start_ms: float
    duration_ms: float = 0.0
    status: str = "ok"
    attributes: dict = field(default_factory=dict)

    def set(self, **attrs) -> None:
        self.attributes.update(attrs)


class Trace:
    def __init__(self, tracer: "Tracer", session_id: str, stage: str, input_text: str):
        self.tracer = tracer
        self.trace_id = uuid.uuid4().hex[:16]
        self.session_id = session_id
        self.stage = stage
        self.input = input_text
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.spans: list[Span] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = ""
        self._t0 = time.perf_counter()
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        span = Span(uuid.uuid4().hex[:12], self._stack[-1].span_id if self._stack else None, name,
                    (time.perf_counter() - self._t0) * 1000, attributes=dict(attrs))
        self.spans.append(span)
        self._stack.append(span)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.set(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            span.duration_ms = (time.perf_counter() - start) * 1000
            self._stack.pop()

    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.model = model or self.model

    def finish(self, output: str, refused: bool = False, refusal_reason: str | None = None,
               tool_calls: int = 0, error: str | None = None) -> dict:
        record = {
            "trace_id": self.trace_id, "session_id": self.session_id, "stage": self.stage,
            "started_at": self.started_at, "duration_ms": (time.perf_counter() - self._t0) * 1000,
            "input": redact_pii(self.input), "output": redact_pii(output),
            "status": "error" if error else ("refused" if refused else "ok"),
            "refused": int(refused), "refusal_reason": refusal_reason or (error and f"error: {error}"),
            "tool_calls": tool_calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "model": self.model, "feedback": None,
        }
        self.tracer.save(record, self.spans)
        return {**record, "spans": [asdict(s) for s in self.spans]}


class NullTrace:
    """Stages v0-v5 run blind: same interface, nothing recorded. That is the point of v6."""

    trace_id = None

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        yield Span("", None, name, 0.0)

    def add_usage(self, *args) -> None:
        pass

    def finish(self, *args, **kwargs) -> None:
        return None


class Tracer:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def start_trace(self, session_id: str, stage: str, input_text: str) -> Trace:
        return Trace(self, session_id, stage, input_text)

    def save(self, record: dict, spans: list[Span]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(f"INSERT INTO traces ({', '.join(record)}) VALUES ({', '.join('?' * len(record))})", list(record.values()))
            conn.executemany(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(s.span_id, record["trace_id"], s.parent_id, s.name, s.start_ms, s.duration_ms, s.status,
                  redact_pii(json.dumps(s.attributes, ensure_ascii=False, default=str))) for s in spans],
            )

    def set_feedback(self, trace_id: str, value: int) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute("UPDATE traces SET feedback = ? WHERE trace_id = ?", (value, trace_id)).rowcount > 0

    def list_traces(self, limit: int = 50, stage: str | None = None) -> list[dict]:
        query, params = "SELECT * FROM traces", []
        if stage:
            query, params = query + " WHERE stage = ?", [stage]
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY started_at DESC, rowid DESC LIMIT ?", [*params, limit]).fetchall()
        return [dict(r) for r in rows]

    def get_trace(self, trace_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if row is None:
                return None
            spans = conn.execute("SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ms", (trace_id,)).fetchall()
        return {**dict(row), "spans": [{**dict(s), "attributes": json.loads(s["attributes"])} for s in spans]}

    def metrics(self) -> dict:
        with self._connect() as conn:
            traces = [dict(r) for r in conn.execute("SELECT * FROM traces").fetchall()]
            span_rows = conn.execute("SELECT name, duration_ms, status FROM spans").fetchall()

        def pct(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1) if len(values) > 1 else round(values[0], 1)

        durations = [t["duration_ms"] for t in traces]
        reasons: dict[str, int] = {}
        for t in traces:
            if t["refused"]:
                reasons[t["refusal_reason"]] = reasons.get(t["refusal_reason"], 0) + 1
        by_span: dict[str, list[float]] = {}
        span_errors: dict[str, int] = {}
        for s in span_rows:
            by_span.setdefault(s["name"], []).append(s["duration_ms"])
            span_errors[s["name"]] = span_errors.get(s["name"], 0) + (s["status"] == "error")
        rated = [t["feedback"] for t in traces if t["feedback"] is not None]
        n = len(traces) or 1
        return {
            "requests": len(traces),
            "latency_p50_ms": pct(durations, 50),
            "latency_p95_ms": pct(durations, 95),
            "error_rate": round(sum(t["status"] == "error" for t in traces) / n, 3),
            "refusal_rate": round(sum(t["refused"] for t in traces) / n, 3),
            "refusal_reasons": reasons,
            "tool_call_rate": round(sum(t["tool_calls"] > 0 for t in traces) / n, 3),
            "tokens_prompt": sum(t["prompt_tokens"] or 0 for t in traces),
            "tokens_completion": sum(t["completion_tokens"] or 0 for t in traces),
            "thumbs_up_rate": round(sum(v > 0 for v in rated) / len(rated), 3) if rated else None,
            "spans": {name: {"count": len(v), "p50_ms": pct(v, 50), "p95_ms": pct(v, 95), "errors": span_errors[name]}
                      for name, v in sorted(by_span.items())},
        }
