"""Optional voice loop, as on the robot: microphone -> Whisper STT -> agent -> TTS speaker.

Install the extras first:  pip install -r requirements-voice.txt
Run:                       python -m ari voice --stage v6
Press Enter to record a fixed-length utterance; Ctrl+C to quit. Speech-to-text and text-to-speech
are timed as their own spans next to the agent's trace, so voice latency shows on the dashboard too.
"""

from __future__ import annotations

import time

SAMPLE_RATE = 16_000


def run_voice_loop(runtime, stage: str = "v6", seconds: float = 5.0, whisper_size: str = "small") -> None:
    try:
        import numpy as np
        import pyttsx3
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(f"Voice extras missing ({exc.name}). Run: pip install -r requirements-voice.txt") from exc

    stt = WhisperModel(whisper_size, device="auto", compute_type="int8")
    tts = pyttsx3.init()
    agent = runtime.agent(stage)
    print(f"Ari voice [{stage}]: press Enter and speak for {seconds:.0f} s (Ctrl+C to quit)")

    while True:
        try:
            input("\n[Enter] to talk ")
        except (EOFError, KeyboardInterrupt):
            return
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        t0 = time.perf_counter()
        segments, info = stt.transcribe(np.squeeze(audio), vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        stt_ms = (time.perf_counter() - t0) * 1000
        if not text:
            print("(heard nothing)")
            continue
        print(f"you ({info.language}, {stt_ms:.0f} ms STT) > {text}")
        reply = agent.chat(text, session_id="voice")
        print(f"ari ({reply.latency_ms:.0f} ms) > {reply.text}")
        t1 = time.perf_counter()
        tts.say(reply.text)
        tts.runAndWait()
        print(f"      tts {(time.perf_counter() - t1) * 1000:.0f} ms | trace {reply.trace_id}")
