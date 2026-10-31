# Sample run: one question up the ladder (real model)

Recorded on 2026-10-31 with `python -m ari --provider openai ladder "What are the library opening hours?"`,
Gemma-3-4B and nomic-embed-text-v1.5 served by LM Studio **on CPU** (no GPU), hence the latencies.
Answers copied verbatim; v4–v6 were not reached because the model server was stopped mid-run.

Ground truth in `knowledge/campus_services.md`: *Monday to Friday 8:30–22:00, Saturday 9:00–18:00.*

| Stage | Answer | Verdict |
|---|---|---|
| v0 API wrapper (25 s) | "Please tell me which library you're asking about! … For example … the **New York Public Library**?" | Generic chatbot: doesn't know where it is |
| v1 Persona (37 s) | "Bonjour ! The library's opening hours are from Monday to Friday, 8 AM to 6 PM. Weekends are closed …" | **Hallucinated**: confident, on-brand, and wrong |
| v2 Grounded (79 s) | "The library is open Monday to Friday from 8:30 to 22:00 [1]. It's also open Saturday from 9:00 to 18:00. … reception desk [3]." | Correct, cited |
| v3 Tools (262 s) | "The library is open Monday to Friday from 8:30 to 22:00 [1]. On Saturday, it's open from 9:00 to 18:00 [1]. …" | Correct, cited; the router call adds a second LLM round-trip |

**The lesson of the v1 row:** a persona makes the bot *sound* right, not *be* right. Grounding is the
step that fixes correctness, and only tracing (v6) shows you that v3's extra latency comes from the router span.

## Retrieval calibration (nomic-embed-text-v1.5)

| Query | Top score | Should answer? |
|---|---|---|
| How long is the second-year internship? | 0.865 | yes |
| What are the library opening hours? | 0.787 | yes |
| Quels sont les horaires de l'infirmerie ? | 0.564 | yes (French query, English docs) |
| Who won the 2022 football World Cup? | 0.530 | no → refused by the 0.55 threshold |
| What is the capital of France? | 0.601 | no → passes the threshold; the grounding prompt must decline |
| What is the name of the director of EMINES? | 0.714 | no → on-topic but absent from the docs; only the prompt can decline |

A similarity threshold alone cannot separate "on-topic but unanswerable" from "answerable". That is why
v5 combines a threshold with a grounding instruction, and why the eval suite has `refuse` cases for both.
