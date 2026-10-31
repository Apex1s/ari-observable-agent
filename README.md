# Ari: from API wrapper to observable agent

**Ari** is the AI assistant robot of EMINES – School of Industrial Management (UM6P). It stands in the
entrance hall, answers questions about the school, reads the timetable, and physically guides visitors.

This repository rebuilds Ari's brain the way companies take an LLM to production: **seven stages, each
adding exactly one layer**, from a bare API call to an agent whose every decision is traced, measured
and evaluated. Pick any stage in the UI, ask the same question, and see what that layer buys.

> **Origin.** The original Ari was built at EMINES by a cross-functional team of 11 (LangChain, a locally
> hosted Llama, speech-to-text / text-to-speech, AstraDB retrieval). It was built, not deployed. This repo
> is my end-to-end rebuild of its conversational stack, structured as a production ladder. The campus
> data in `knowledge/` is **sample data**, to be replaced with official documents.

## The ladder

| Stage | Adds | What goes wrong without it |
|---|---|---|
| **v0** API wrapper | A raw call to the model | Knows nothing about EMINES; answers like a generic chatbot |
| **v1** Persona | System prompt: identity, audience, spoken style, language rules | Sounds right, **invents facts** ([real example](docs/sample_ladder_gemma3-4b.md)) |
| **v2** Grounded (RAG) | Retrieval over school documents, numbered citations, "I don't know" rule | Can't act or read live data |
| **v3** Tools (agent) | JSON router → campus directory, timetable, **robot navigation**, clock, calculator | Forgets the previous turn |
| **v4** Memory | Sliding window of recent turns per session | Obeys prompt injections, guesses when nothing is retrieved |
| **v5** Guardrails | Injection and length checks, refusal when nothing is grounded, citation validation, restricted-area safety | Runs blind: no way to know *why* an answer was wrong or slow |
| **v6** Observable | Span-level traces in SQLite, latency/token/refusal metrics, 👍/👎 feedback, live dashboard, PII redaction in logs | n/a |

Every stage is the same pipeline (`ari/agent.py`) with feature flags switched on (`ari/stages.py`), so
the difference between two stages is exactly one layer.

```mermaid
flowchart LR
    mic([🎤 STT]) --> G1[guard.input]
    G1 --> R[route<br/>LLM → JSON]
    R -->|tool| T[tool.*<br/>directory · timetable · robot_navigate]
    R --> RET[retrieve<br/>nomic embeddings · top-k · threshold]
    T --> RET
    RET --> S{grounded?}
    S -->|no| REF[refuse: no_grounding]
    S -->|yes| L[llm.answer<br/>persona + context + tool results + memory]
    L --> G2[guard.output<br/>citation check]
    G2 --> spk([🔊 TTS])
    G1 & R & T & RET & L & G2 -.spans.-> DB[(SQLite traces)] --> D[Dashboard · Evals]
```

## Quick start

```bash
pip install -r requirements-dev.txt
```

**Option A: Ollama (default, like the original Ari)**
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python -m ari serve          # http://127.0.0.1:8000
```

**Option B: LM Studio** (or any OpenAI-compatible server): start its server, load a chat model and
`nomic-embed-text`, then
```bash
python -m ari --provider openai --model google/gemma-3-4b --embed-model text-embedding-nomic-embed-text-v1.5 serve
```

**Option C: no model at all.** The deterministic mock provider runs the whole pipeline offline:
```bash
python -m ari --provider mock serve
```

**Option D: Docker** (Ollama + models + Ari):
```bash
docker compose up --build
```

### Other commands
```bash
python -m ari chat --stage v6                              # terminal chat with trace ids
python -m ari ladder "What are the library opening hours?" # same question, all 7 stages
python -m ari eval --stages v0,v2,v6                       # eval suite -> evals/reports/
python -m ari voice                                        # mic -> Whisper -> Ari -> TTS
python -m pytest -q                                        # 15 offline tests
```

## Observability (v6)

Each request is a trace; each step is a span with its duration and what it saw and decided:

| Span | Attributes recorded |
|---|---|
| `guard.input` | allowed, refusal reason |
| `route` / `llm.route` | router JSON (raw and parsed), tokens |
| `tool.<name>` | arguments, result, error status |
| `retrieve` | query, every candidate chunk with its score, how many passed the threshold |
| `llm.answer` | model, prompt and completion tokens |
| `guard.output` | invalid citations removed, uncited answer flag |

The dashboard aggregates p50/p95 latency per step, refusal reasons, tool-call rate, token usage and
the 👍 rate. Inputs, outputs and attributes are **PII-redacted before storage** (e-mails, phone numbers).
The schema maps one-to-one onto OpenTelemetry spans, so it can be exported to Langfuse, Phoenix or Jaeger.

**Why it matters, concretely:** when Ari says something wrong, the trace tells you which layer failed.
Was the right chunk retrieved, and with what score? Did the router pick the wrong tool? Did the model
ignore the context? Each of those has a different fix (documents, router prompt, or model).

## Evals

`evals/dataset.jsonl` has 24 cases in five categories: **grounded** facts (13, including one in French),
**tool** routing (4), robot **safety** (1), **refuse** (4: two prompt injections, one off-topic and one
on-topic question the documents can't answer) and multi-turn **memory** (2). `python -m ari eval` runs
every case against every stage and writes a Markdown report with pass rates per category, citation rate
and latency. Failures at the top stage are listed with their **trace id**, so each one can be opened in
the dashboard.

Offline (mock provider), the ladder goes from **21% (v0) to 92% (v6)**; CI runs this on every push.
Mock numbers only prove the plumbing. Run the suite with your real model to get the numbers that matter.

## Design decisions

- **Local-first.** Ollama or LM Studio, nomic embeddings, SQLite: no conversation leaves the campus network.
- **Prompt-based JSON routing** instead of native function calling. Small local models (3–4B) handle a
  constrained JSON reply more reliably than tool-calling APIs, and it works on every server.
- **Two-layer grounding.** A similarity threshold alone can't separate "on-topic but unanswerable" from
  "answerable" (see [calibration](docs/sample_ladder_gemma3-4b.md#retrieval-calibration-nomic-embed-text-v15)),
  so v5 combines it with a grounding instruction plus output citation checks.
- **No framework in the core.** The original Ari used LangChain. Here each layer is about 50 lines of
  plain Python, so the ladder stays readable. `VectorStore` is the single class to swap for AstraDB.
- **Robot safety lives in the tool, not the prompt.** `robot_navigate` refuses restricted or
  non-reachable places in code, whatever the model asks for. `SimulatedRobot` logs commands; on the real
  robot, implement `RobotInterface.navigate` with a ROS 2 Nav2 goal.

## Project structure

```
ari/
  stages.py      the ladder: 7 cumulative feature-flag configs
  agent.py       the pipeline (guard → route/tools → retrieve → answer → guard → memory)
  llm.py         Ollama, OpenAI-compatible, and mock providers (stdlib HTTP only)
  rag.py         chunking, nomic/hashing embedders, NumPy vector store with fingerprinted cache
  tools.py       campus directory, timetable, robot navigation, clock, safe calculator
  guardrails.py  injection/length checks, scope refusal, citation validation, PII redaction
  memory.py      per-session sliding window
  tracing.py     spans → SQLite, metrics aggregation
  evals.py       eval runner and Markdown report
  server.py      FastAPI: /api/chat, /api/traces, /api/metrics, /api/feedback
  voice.py       optional Whisper STT + pyttsx3 TTS loop
  static/        web console (chat + trace waterfall + dashboard), no build step
knowledge/       sample EMINES documents and campus/timetable JSON
evals/           dataset.jsonl (+ generated reports/)
docs/            real-model sample run and retrieval calibration
tests/           offline pytest suite (mock provider)
```

## Roadmap

- Export spans over OTLP to Langfuse/Phoenix; add prompt versions to the trace
- LLM-as-judge faithfulness scoring alongside keyword checks
- ROS 2 `RobotInterface` implementation and a wake word for the voice loop
- Query rewriting for follow-ups instead of concatenating the previous question

Built with Claude Code.
