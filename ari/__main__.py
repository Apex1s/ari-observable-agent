"""Command line: python -m ari {chat,ladder,eval,calibrate,serve,voice} [--provider groq|gemini|openai|ollama|mock] [--model NAME]"""

from __future__ import annotations

import argparse
import logging
import sys

from .stages import STAGES


def _runtime(args):
    from .runtime import build_runtime

    return build_runtime(provider=args.provider, model=args.model, embed_model=args.embed_model)


def cmd_chat(args) -> None:
    runtime = _runtime(args)
    agent = runtime.agent(args.stage)
    print(f"Ari [{args.stage}: {STAGES[args.stage].title}]  (Ctrl+C to quit)")
    while True:
        try:
            message = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        reply = agent.chat(message, session_id="cli")
        print(f"ari > {reply.text}")
        details = [f"{reply.latency_ms:.0f} ms"]
        details += [f"tool {c['tool']}({c['args']})" for c in reply.tool_calls]
        details += [f"[{c['n']}] {c['source']} > {c['section']} ({c['score']})" for c in reply.citations if c["cited"]]
        if reply.refused:
            details.append(f"refused: {reply.refusal_reason}")
        if reply.trace_id:
            details.append(f"trace {reply.trace_id}")
        print("      " + " | ".join(details))


def cmd_ladder(args) -> None:
    runtime = _runtime(args)
    for key, stage in STAGES.items():
        reply = runtime.agent(key).chat(args.question, session_id="ladder")
        print(f"\n{key} {stage.title}  ({reply.latency_ms:.0f} ms)\n  {reply.text}")


def cmd_eval(args) -> None:
    from . import evals

    runtime = _runtime(args)
    stages = args.stages.split(",") if args.stages else None
    results = evals.run(runtime, stages)
    label = f"{runtime.settings.provider} {runtime.llm.model}"
    path = evals.write_report(results, label)
    print("\n" + path.read_text(encoding="utf-8"))
    print(f"Report written to {path}")


def cmd_serve(args) -> None:
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(_runtime(args)), host=args.host, port=args.port)


OFF_TOPIC = ["Who won the 2022 football World Cup?", "What is the capital of France?", "Give me a pasta recipe.",
             "How do I fix a flat bicycle tyre?", "What is the price of Bitcoin today?", "Write a poem about the sea."]


def cmd_calibrate(args) -> None:
    """Pick the retrieval cut-off for the current embedding model from where relevant and off-topic
    questions actually score. Embedding calls only: cheap, and remote with the default setup."""
    from .evals import load_cases

    runtime = _runtime(args)
    top = lambda q: runtime.store.search(q, 1)[0].score  # noqa: E731
    relevant = [(c["turns"][0], top(c["turns"][0])) for c in load_cases() if c["category"] == "grounded"]
    off_topic = [(q, top(q)) for q in OFF_TOPIC]
    for label, rows in (("relevant", relevant), ("off-topic", off_topic)):
        print(f"\n{label}:")
        for q, s in sorted(rows, key=lambda r: r[1]):
            print(f"  {s:.3f}  {q}")
    low_relevant, high_off = min(s for _, s in relevant), max(s for _, s in off_topic)
    suggestion = round((low_relevant + high_off) / 2, 3)
    print(f"\nembedder {runtime.store.embedder.id}: lowest relevant {low_relevant:.3f}, highest off-topic {high_off:.3f}")
    if low_relevant > high_off:
        print(f"Clean separation. Set ARI_MIN_SCORE={suggestion} in .env")
    else:
        print(f"Overlap: no cut-off separates them. ARI_MIN_SCORE={suggestion} is a compromise; "
              "the grounding prompt has to catch the rest (see the eval 'refuse' category).")


def cmd_voice(args) -> None:
    from .voice import run_voice_loop

    run_voice_loop(_runtime(args), args.stage, args.seconds, args.whisper)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ari", description="Ari: from API wrapper to observable agent")
    parser.add_argument("--provider", choices=["groq", "gemini", "openai", "ollama", "mock"], help="overrides ARI_PROVIDER")
    parser.add_argument("--model", help="overrides ARI_MODEL")
    parser.add_argument("--embed-model", help="overrides ARI_EMBED_MODEL")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chat", help="talk to one stage in the terminal")
    p.add_argument("--stage", default="v6", choices=list(STAGES))
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ladder", help="ask one question to every stage, side by side")
    p.add_argument("question")
    p.set_defaults(func=cmd_ladder)

    p = sub.add_parser("eval", help="run the eval suite and write a report")
    p.add_argument("--stages", help="comma-separated, e.g. v0,v3,v6 (default: all)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="web UI + API + observability dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("calibrate", help="suggest the retrieval threshold for the current embedding model")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("voice", help="talk to Ari through the microphone (needs requirements-voice.txt)")
    p.add_argument("--stage", default="v6", choices=list(STAGES))
    p.add_argument("--seconds", type=float, default=5.0, help="recording length per utterance")
    p.add_argument("--whisper", default="small", help="faster-whisper model size")
    p.set_defaults(func=cmd_voice)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose or args.command == "serve" else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
