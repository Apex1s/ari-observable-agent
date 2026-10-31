"""Talk to Ari: an English voice assistant grounded in the EMINES / UM6P knowledge base.

    python -m ari.talk            # voice: speak, pause, Ari answers out loud
    python -m ari.talk --text     # type instead of speaking (Ari still answers out loud)
    python -m ari.talk --mute     # text replies only

Fast path, built for 1-3 s replies, nothing heavy on this machine:
  1. Microphone -> end of speech detected by loudness (no model).
  2. Speech-to-text, remote: Groq Whisper if GROQ_API_KEY is set, otherwise Google's keyless web API.
  3. Retrieval (RAG): the knowledge base is searched locally (BM25, or embeddings when a key is set),
     the best passages plus the campus map go into ONE chat call, with conversation memory.
  4. Text-to-speech, remote: Microsoft Edge neural voices (edge-tts). The first sentence starts
     playing while the next ones are still being synthesized. Falls back to Windows' built-in voice.
Each turn prints its latency split so speed is measurable.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import io
import json
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

from .llm import LLMError
from .runtime import build_runtime

SAMPLE_RATE = 16_000
BLOCK_MS = 30
VOICE = os.environ.get("ARI_VOICE", "en-US-AvaNeural")
EXIT_WORDS = re.compile(r"^\s*(stop|exit|quit|bye|goodbye)\W*$", re.I)

PERSONA = """You are Ari, the friendly assistant robot in the entrance hall of EMINES - School of Industrial \
Management, Mohammed VI Polytechnic University (UM6P), Ben Guerir, Morocco. You talk with students, staff and visitors.
Your words are spoken aloud: answer in ENGLISH ONLY, in one to three short, natural sentences. No lists, no markdown, \
no URLs, no source numbers. Sound warm and confident, like a helpful person at a front desk.
Use ONLY the facts below. If they don't contain the answer, say briefly that you don't know and suggest asking at \
the EMINES reception or checking the official website. Never invent hours, names, rooms or numbers.
If someone asks you to take or guide them somewhere listed under CAMPUS PLACES as reachable, say you'll take them \
there and give the walking time. Never guide anyone to restricted or private places."""


# -- microphone ----------------------------------------------------------------------------------

def measure_noise(sd, np, seconds: float = 0.6) -> float:
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return float(np.sqrt(np.mean(audio ** 2)))


def record_utterance(sd, np, noise_floor: float, max_seconds: float = 15.0, silence_ms: int = 600) -> bytes | None:
    """Record from the first sound until `silence_ms` of quiet; return 16-bit mono PCM."""
    block = int(SAMPLE_RATE * BLOCK_MS / 1000)
    start_level, keep_level = noise_floor * 3.0 + 0.004, noise_floor * 2.0 + 0.003
    frames, started, quiet_ms, voiced_ms = [], False, 0, 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block) as stream:
        while True:
            data, _ = stream.read(block)
            level = float(np.sqrt(np.mean(data ** 2)))
            if not started:
                frames = (frames + [data])[-10:]  # 300 ms pre-roll so the first syllable isn't cut
                started = level > start_level
                continue
            frames.append(data)
            voiced_ms += BLOCK_MS if level >= keep_level else 0
            quiet_ms = quiet_ms + BLOCK_MS if level < keep_level else 0
            if quiet_ms >= silence_ms or len(frames) * BLOCK_MS >= max_seconds * 1000:
                break
    if voiced_ms < 200:  # a click or a cough, not speech
        return None
    audio = np.concatenate(frames)[:, 0]
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


# -- speech to text (remote) ---------------------------------------------------------------------

def transcribe(pcm: bytes) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    return _groq_whisper(pcm, key) if key else _google(pcm)


def _google(pcm: bytes) -> str:
    import speech_recognition as sr

    try:
        return sr.Recognizer().recognize_google(sr.AudioData(pcm, SAMPLE_RATE, 2), language="en-US").strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        raise LLMError(f"speech recognition unreachable: {exc}") from exc


def _groq_whisper(pcm: bytes, key: str) -> str:
    boundary = uuid.uuid4().hex
    fields = {"model": "whisper-large-v3-turbo", "language": "en", "temperature": "0",
              "prompt": "Ari, EMINES, UM6P, Ben Guerir, Learning Center, FabLab."}
    body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields.items())
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="s.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n").encode() + _wav(pcm) + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8")).get("text", "").strip()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError(f"speech-to-text failed: {exc}") from exc


# -- text to speech: neural voice, sentence-pipelined -----------------------------------------------

def _mci(command: str) -> str:
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.winmm.mciSendStringW(command, buf, 255, 0)
    return buf.value


class Speaker:
    """Synthesizes sentences in one thread and plays them in another, so playback starts after the
    first sentence and the gaps between sentences are hidden."""

    def __init__(self, enabled: bool = True, voice: str = VOICE):
        self.enabled = enabled
        self.voice = voice
        self._text: queue.Queue[str] = queue.Queue()
        self._audio: queue.Queue[Path | str] = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self.first_audio_at: float | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="ari-tts-"))
        self._fallback = None
        if enabled:
            threading.Thread(target=self._synth_worker, daemon=True).start()
            threading.Thread(target=self._play_worker, daemon=True).start()

    def say(self, text: str) -> None:
        if not self.enabled:
            return
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        with self._lock:
            self._pending += len(sentences)
            self._idle.clear()
        self.first_audio_at = None
        for s in sentences:
            self._text.put(s)

    def wait(self) -> None:
        self._idle.wait()

    def _done_one(self) -> None:
        with self._lock:
            self._pending -= 1
            if self._pending <= 0:
                self._pending = 0
                self._idle.set()

    def _synth_worker(self) -> None:
        import edge_tts

        loop = asyncio.new_event_loop()
        n = 0
        while True:
            sentence = self._text.get()
            path = self._dir / f"s{n}.mp3"
            n += 1
            try:
                loop.run_until_complete(edge_tts.Communicate(sentence, self.voice, rate="+5%").save(str(path)))
                self._audio.put(path)
            except Exception:  # offline or service hiccup: speak this sentence with the Windows voice
                self._audio.put(sentence)

    def _play_worker(self) -> None:
        n = 0
        while True:
            item = self._audio.get()
            if self.first_audio_at is None:
                self.first_audio_at = time.perf_counter()
            try:
                if isinstance(item, Path):
                    alias = f"ari{n}"
                    n += 1
                    _mci(f'open "{item}" type mpegvideo alias {alias}')
                    _mci(f"play {alias} wait")
                    _mci(f"close {alias}")
                    item.unlink(missing_ok=True)
                else:
                    self._say_offline(item)
            finally:
                self._done_one()

    def _say_offline(self, text: str) -> None:
        try:
            import pyttsx3
        except ImportError:
            return
        if self._fallback is None:
            self._fallback = pyttsx3.init()
        self._fallback.say(text)
        self._fallback.runAndWait()


# -- the brain: retrieval + one chat call --------------------------------------------------------

class Brain:
    def __init__(self, runtime, memory_turns: int = 4):
        self.rt = runtime
        self.history: list[dict[str, str]] = []
        self.memory_turns = memory_turns
        places = json.loads(runtime.settings.campus_file.read_text(encoding="utf-8"))["places"]
        lines = []
        for p in places.values():
            status = ("restricted, never guide" if p.get("restricted") else "private, never guide" if p.get("private")
                      else f"reachable, {p.get('walk_minutes')} min walk" if p.get("navigable") else "not reachable on foot")
            lines.append(f"- {p['name']}: {p.get('where', '')} ({status})")
        self.places = "\n".join(lines)

    def answer(self, question: str) -> tuple[str, list[str]]:
        query = question
        if len(question.split()) < 6 and self.history:  # "and on Saturday?" -> search with the previous question
            query = f"{self.history[-2]['content']} {question}"
        hits = [h for h in self.rt.store.search(query, 4) if h.score >= self.rt.min_score]
        facts = "\n\n".join(f"({h.chunk.heading}) {re.sub(r'Source: .*', '', h.chunk.text).strip()}" for h in hits)
        system = f"{PERSONA}\n\nFACTS:\n{facts or '(nothing relevant found)'}\n\nCAMPUS PLACES:\n{self.places}"
        messages = [{"role": "system", "content": system}, *self.history[-2 * self.memory_turns:],
                    {"role": "user", "content": question}]
        for attempt in range(3):
            try:
                text = self.rt.llm.chat(messages, temperature=0.3).text
                break
            except LLMError as exc:
                wait = re.search(r"[Rr]etry after (\d+)", str(exc))  # keyless tier: 1 reply / 45 s per IP
                if attempt == 2 or (wait and int(wait.group(1)) > 60):
                    raise
                if wait:
                    print(f"      (free keyless tier: waiting {wait.group(1)} s; add a Groq key for instant replies)")
                time.sleep(int(wait.group(1)) + 1 if wait else 1 + attempt)
        text = re.sub(r"\[\d+\]|\*+|#+|https?://\S+", "", text).strip()
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
        return text, sorted({h.chunk.source for h in hits})


# -- provider choice -----------------------------------------------------------------------------

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _choose_provider() -> str | None:
    """Groq when a key exists (fast: ~0.3 s answers, Whisper STT); otherwise offer to save one once,
    and fall back to the keyless tier configured in .env."""
    from .config import _load_dotenv

    _load_dotenv(ENV_FILE)
    if not os.environ.get("GROQ_API_KEY") and sys.stdin.isatty():
        print("Ari answers instantly with a free Groq key (1 minute: https://console.groq.com/keys).")
        key = input("Paste it here and press Enter, or just press Enter for the slow keyless mode: ").strip()
        if key.startswith("gsk_"):
            with ENV_FILE.open("a", encoding="utf-8") as f:
                f.write(f"\nGROQ_API_KEY={key}\n")
            os.environ["GROQ_API_KEY"] = key
            print("Saved to .env. You won't be asked again.")
        elif key:
            print("That doesn't look like a Groq key (they start with gsk_); using keyless mode.")
    if os.environ.get("GROQ_API_KEY"):
        os.environ["ARI_MODEL"] = os.environ.get("ARI_GROQ_MODEL", "llama-3.3-70b-versatile")
        os.environ.pop("ARI_BASE_URL", None)
        os.environ.pop("ARI_API_KEY", None)
        return "groq"
    return None


# -- main loop -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ari.talk", description="Talk to Ari by voice (English)")
    parser.add_argument("--text", action="store_true", help="type instead of speaking")
    parser.add_argument("--mute", action="store_true", help="don't speak replies")
    parser.add_argument("--provider", help="override ARI_PROVIDER (mock = offline demo)")
    parser.add_argument("--voice", default=VOICE, help="edge-tts voice, e.g. en-US-AndrewNeural, en-GB-SoniaNeural")
    parser.add_argument("--stage", help=argparse.SUPPRESS)  # accepted for CLI compatibility
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.ERROR)

    provider = args.provider or _choose_provider()
    print("Starting Ari…")
    try:
        brain = Brain(build_runtime(provider=provider))
    except LLMError as exc:
        sys.exit(f"\n{exc}")
    speaker = Speaker(enabled=not args.mute, voice=args.voice)

    voice = not args.text
    if voice:
        try:
            import numpy as np
            import sounddevice as sd
            noise = measure_noise(sd, np)
        except Exception as exc:
            print(f"Microphone unavailable ({exc}): switching to typing mode.")
            voice = False

    print(f"\nAri is ready ({brain.rt.settings.provider}:{brain.rt.llm.model}, {len(brain.rt.store.chunks)} knowledge passages).")
    print("Just talk, then pause. Say 'stop' to quit." if voice else "Type your question; 'stop' to quit.")
    greeting = "Hi! I'm Ari, the EMINES assistant robot. How can I help you?"
    print(f"ari > {greeting}")
    speaker.say(greeting)

    while True:
        speaker.wait()  # never listen while Ari is talking: it would hear itself
        stt_ms = 0.0
        try:
            if voice:
                print("\n🎤 listening…", flush=True)
                pcm = record_utterance(sd, np, noise)
                if pcm is None:
                    continue
                t0 = time.perf_counter()
                text = transcribe(pcm)
                stt_ms = (time.perf_counter() - t0) * 1000
                if not text:
                    print("(didn't catch that)")
                    continue
                print(f"you > {text}")
            else:
                text = input("\nyou > ").strip()
                if not text:
                    continue
        except (KeyboardInterrupt, EOFError):
            break
        except LLMError as exc:
            print(f"(speech error: {exc})")
            continue
        if EXIT_WORDS.match(text):
            break

        t1 = time.perf_counter()
        try:
            reply, sources = brain.answer(text)
        except LLMError as exc:
            print(f"(model error: {exc})")
            speaker.say("Sorry, I lost my connection for a second. Could you say that again?")
            continue
        brain_ms = (time.perf_counter() - t1) * 1000
        speaker.say(reply)
        print(f"ari > {reply}")
        if speaker.enabled:
            while speaker.first_audio_at is None and not speaker._idle.is_set():
                time.sleep(0.02)
        heard_ms = ((speaker.first_audio_at or time.perf_counter()) - t1) * 1000 + stt_ms
        timing = ([f"stt {stt_ms:.0f} ms"] if voice else []) + [f"answer {brain_ms:.0f} ms",
                                                                  f"voice starts after {heard_ms:.0f} ms"]
        print("      " + " | ".join(timing) + (f" | from {', '.join(sources)}" if sources else ""))

    farewell = "Goodbye, and have a great day!"
    print(f"ari > {farewell}")
    speaker.say(farewell)
    speaker.wait()


if __name__ == "__main__":
    main()
