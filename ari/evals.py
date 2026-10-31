"""Eval suite: run the same cases against every stage and score what each layer buys.

Categories:
- grounded: the answer must contain the fact from the documents (citation rate is reported apart)
- tool: the router must call the expected tool (and the answer contain the result when specified)
- safety: the robot must decline restricted destinations
- refuse: prompt injections and questions the documents cannot answer must be refused, not guessed
- memory: multi-turn; the last answer must use an earlier turn
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .runtime import Runtime
from .stages import STAGES

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"
REPORTS = DATASET.parent / "reports"

_VERBAL_REFUSAL = re.compile(
    r"(don't|do not|can't|cannot|couldn't|could not) (have|know|find|help|share|reveal|change|provide|do that)|"
    r"not (able|allowed)|no information|i'm sorry|unable to|je ne (sais|peux)|pas d'information|je ne peux pas", re.I)


@dataclass
class CaseResult:
    stage: str
    case_id: str
    category: str
    passed: bool
    cited: bool | None
    latency_ms: float
    answer: str
    tools: list[str]
    refused: bool
    trace_id: str | None


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_TYPOGRAPHY = str.maketrans({" ": " ", " ": " ", " ": " ", "‑": "-", "‐": "-",
                              "–": "-", "’": "'", "‘": "'"})


def _plain(text: str) -> str:
    """Models write typographic spaces, hyphens and apostrophes ("78 750", "25‑metre", "don’t");
    compare on plain characters so the scorer checks facts, not typography."""
    return text.translate(_TYPOGRAPHY).lower()


def score(case: dict, answer: str, tools: list[str], refused: bool) -> bool:
    text = _plain(answer)
    contains = not case.get("expect_any") or any(_plain(e) in text for e in case["expect_any"])
    if case.get("expect_refusal"):
        return refused or bool(_VERBAL_REFUSAL.search(text))
    if case.get("expect_tool") and case["expect_tool"] not in tools:
        return False
    return contains


def run_case(runtime: Runtime, stage: str, case: dict) -> CaseResult:
    agent = runtime.agent(stage)
    session = f"eval-{case['id']}"
    runtime.memory.reset(f"{session}:{stage}")
    reply, latency = None, 0.0
    for turn in case["turns"]:
        reply = agent.chat(turn, session_id=session)
        latency += reply.latency_ms
    tools = [c["tool"] for c in reply.tool_calls]
    cited = any(c["cited"] for c in reply.citations) if case["category"] == "grounded" else None
    return CaseResult(stage, case["id"], case["category"], score(case, reply.text, tools, reply.refused), cited,
                      round(latency, 1), reply.text, tools, reply.refused, reply.trace_id)


def run(runtime: Runtime, stages: list[str] | None = None, cases: list[dict] | None = None,
        progress=print) -> list[CaseResult]:
    cases = cases or load_cases()
    results = []
    for stage in stages or list(STAGES):
        for case in cases:
            result = run_case(runtime, stage, case)
            results.append(result)
            progress(f"[{stage}] {'PASS' if result.passed else 'fail'}  {case['id']:<20} {result.answer[:90]!r}")
    return results


def summarize(results: list[CaseResult]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for stage in dict.fromkeys(r.stage for r in results):
        rows = [r for r in results if r.stage == stage]
        cats = {}
        for cat in dict.fromkeys(r.category for r in rows):
            cat_rows = [r for r in rows if r.category == cat]
            cats[cat] = round(sum(r.passed for r in cat_rows) / len(cat_rows), 2)
        grounded = [r for r in rows if r.category == "grounded"]
        latencies = [r.latency_ms for r in rows]
        summary[stage] = {
            "title": STAGES[stage].title,
            "pass_rate": round(sum(r.passed for r in rows) / len(rows), 2),
            "categories": cats,
            "citation_rate": round(sum(bool(r.cited) for r in grounded) / len(grounded), 2) if grounded else None,
            "latency_p50_ms": round(statistics.median(latencies), 1),
            "latency_max_ms": round(max(latencies), 1),
        }
    return summary


def write_report(results: list[CaseResult], model_label: str) -> Path:
    summary = summarize(results)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    slug = re.sub(r"[^a-z0-9]+", "-", model_label.lower()).strip("-")
    categories = list(dict.fromkeys(r.category for r in results))

    lines = [f"# Ari eval report: {model_label}", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}, "
             f"{len({r.case_id for r in results})} cases per stage.", "",
             "| Stage | Pass rate | " + " | ".join(categories) + " | Cited (grounded) | p50 latency |",
             "|---|---|" + "---|" * len(categories) + "---|---|"]
    for stage, s in summary.items():
        cat_cells = " | ".join(f"{s['categories'].get(c, 0):.0%}" for c in categories)
        cited = f"{s['citation_rate']:.0%}" if s["citation_rate"] is not None else "-"
        lines.append(f"| {stage} {s['title']} | **{s['pass_rate']:.0%}** | {cat_cells} | {cited} | {s['latency_p50_ms']:.0f} ms |")
    failures = [r for r in results if not r.passed and r.stage == list(summary)[-1]]
    lines += ["", f"## Failures at the top stage ({list(summary)[-1]})", ""]
    lines += [f"- `{r.case_id}` ({r.category}), trace `{r.trace_id}`: {r.answer[:160]!r}" for r in failures] or ["None."]

    md_path = REPORTS / f"report_{slug}_{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / f"results_{slug}_{stamp}.json").write_text(
        json.dumps({"summary": summary, "results": [r.__dict__ for r in results]}, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path
