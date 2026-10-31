# Ari eval report: keyless ch.at/gpt-4o

Generated 2026-10-31 14:01, 35 cases per stage.

| Stage | Pass rate | grounded | tool | safety | refuse | memory | Cited (grounded) | p50 latency |
|---|---|---|---|---|---|---|---|---|
| v0 API wrapper | **20%** | 24% | 0% | 0% | 40% | 0% | 0% | 2495 ms |
| v1 Persona | **29%** | 33% | 0% | 0% | 60% | 0% | 0% | 2187 ms |
| v2 Grounded (RAG) | **80%** | 100% | 0% | 100% | 80% | 50% | 100% | 2143 ms |
| v3 Tools (agent) | **94%** | 100% | 100% | 100% | 80% | 50% | 100% | 3910 ms |
| v4 Memory | **97%** | 100% | 100% | 100% | 80% | 100% | 100% | 3838 ms |
| v5 Guardrails | **100%** | 100% | 100% | 100% | 100% | 100% | 100% | 4010 ms |
| v6 Observable | **100%** | 100% | 100% | 100% | 100% | 100% | 100% | 4044 ms |

## Failures at the top stage (v6)

None.
