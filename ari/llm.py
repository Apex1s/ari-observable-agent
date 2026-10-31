"""LLM providers behind one small interface.

- OpenAICompatLLM: any OpenAI-compatible API. The default setup is fully remote: Groq
  (Llama 3.3 70B) for chat and Gemini for embeddings, so no model runs on the host machine.
- OllamaLLM: Ollama's REST API, for a self-hosted Llama on a dedicated server (as on the robot).
- MockLLM: deterministic, offline; used by the tests and CI so the pipeline runs without a model.

Only the standard library is used for HTTP so the provider layer has no dependencies.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .config import Settings

log = logging.getLogger(__name__)

Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str
    model: str
    embed_model: str

    def chat(self, messages: list[Message], *, json_mode: bool = False, temperature: float = 0.2) -> LLMResponse: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


_RETRYABLE = {429, 500, 502, 503, 504}


def _post_json(url: str, payload: dict, timeout: float, headers: dict[str, str] | None = None,
               max_retries: int = 5) -> dict:
    """POST with retries: free API tiers answer 429 when rate-limited, so wait (honouring Retry-After) and retry."""
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json", "User-Agent": "ari-agent/1.0", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code not in _RETRYABLE or attempt == max_retries:
                raise LLMError(f"{url} returned HTTP {exc.code}: {body}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2 ** attempt
            log.warning("HTTP %s from %s, retrying in %.1fs (%d/%d)", exc.code, url, delay, attempt + 1, max_retries)
            time.sleep(min(delay, 60))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == max_retries:
                raise LLMError(f"cannot reach {url}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


class OllamaLLM:
    name = "ollama"

    def __init__(self, base_url: str, model: str, embed_model: str, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout

    def chat(self, messages, *, json_mode=False, temperature=0.2):
        payload = {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
        if json_mode:
            payload["format"] = "json"
        r = _post_json(f"{self.base_url}/api/chat", payload, self.timeout)
        return LLMResponse(
            text=r["message"]["content"],
            prompt_tokens=r.get("prompt_eval_count", 0),
            completion_tokens=r.get("eval_count", 0),
            model=r.get("model", self.model),
        )

    def embed(self, texts):
        r = _post_json(f"{self.base_url}/api/embed", {"model": self.embed_model, "input": texts}, self.timeout)
        return r["embeddings"]


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, embed_model: str, api_key: str = "not-needed", timeout: float = 120,
                 name: str = "openai", max_tokens: int | None = None, max_retries: int = 5):
        self.name = name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def chat(self, messages, *, json_mode=False, temperature=0.2):
        # json_mode is enforced by the prompt: response_format support varies across local servers.
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        r = _post_json(f"{self.base_url}/chat/completions", payload, self.timeout, self._headers, self.max_retries)
        usage = r.get("usage") or {}
        return LLMResponse(
            text=r["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=r.get("model", self.model),
        )

    def embed(self, texts):
        r = _post_json(f"{self.base_url}/embeddings", {"model": self.embed_model, "input": texts}, self.timeout,
                       self._headers, self.max_retries)
        # Gemini's OpenAI-compatible endpoint omits "index"; its items are already in input order.
        items = sorted(r["data"], key=lambda d: d["index"]) if all("index" in d for d in r["data"]) else r["data"]
        return [item["embedding"] for item in items]


# Keyless OpenAI-compatible endpoints, in order of preference (scouted and live-tested 2026-10-31):
# ch.at serves GPT-4o with no published limit; Pollinations allows ~1 anonymous call per 45 s (HTTP 402
# when exceeded); LLM7's keyless tier rate-limits hard (429 with Retry-After up to an hour). None needs
# an account, none has an SLA, and all forward prompts to third-party model hosts: never send personal data.
KEYLESS_PROVIDERS = [
    ("ch.at", "https://ch.at/v1", "gpt-4o", None),
    ("pollinations", "https://text.pollinations.ai/openai", "openai-fast", None),
    ("llm7", "https://api.llm7.io/v1", "gpt-oss:20b", 800),  # reasoning model: needs room for its thoughts
]


class FallbackLLM:
    """Tries providers in order and fails over on any error. A provider that fails sits out a cooldown
    (longer when it was rate-limited), so a throttled service is not hammered on every request."""

    name = "keyless"
    embed_model = "none"

    def __init__(self, clients: list[tuple[str, "OpenAICompatLLM"]], cooldown_s: float = 60):
        self.clients = clients
        self.cooldown_s = cooldown_s
        self._resting_until: dict[str, float] = {}
        self.model = f"{clients[0][0]}/{clients[0][1].model}"

    def chat(self, messages, *, json_mode=False, temperature=0.2):
        errors = []
        now = time.monotonic()
        ready = [c for c in self.clients if self._resting_until.get(c[0], 0) <= now] or self.clients
        for label, client in ready:
            try:
                response = client.chat(messages, json_mode=json_mode, temperature=temperature)
            except (LLMError, KeyError, IndexError, TypeError) as exc:
                rate_limited = any(code in str(exc) for code in ("HTTP 429", "HTTP 402"))
                self._resting_until[label] = time.monotonic() + self.cooldown_s * (5 if rate_limited else 1)
                errors.append(f"{label}: {str(exc)[:120]}")
                log.warning("Keyless provider %s failed, failing over: %s", label, str(exc)[:160])
                continue
            if not response.text.strip():
                errors.append(f"{label}: empty answer")
                continue
            self.model = f"{label}/{client.model}"
            response.model = self.model
            return response
        raise LLMError("all keyless providers failed: " + " | ".join(errors))

    def embed(self, texts):
        raise LLMError("keyless providers serve no embeddings; retrieval uses BM25")


class MockLLM:
    """Rule-based stand-in for a model. It reads the same prompts a real model would get, so every
    stage of the pipeline (routing, grounding, memory, guardrails, tracing) is exercised offline."""

    name = "mock"
    model = "mock"
    embed_model = "hashing"

    _ROUTES = [
        (r"\b(take|guide|bring|walk|accompany|lead) me\b|\bemm[eè]ne[- ]moi\b|\bconduis[- ]moi\b", "robot_navigate"),
        (r"\bwhere is\b|\bwhere's\b|\bo[uù] (est|se trouve)\b", "campus_directory"),
        (r"\bweather\b|\bm[ée]t[ée]o\b|\btemperature\b|\brain(ing)?\b|\bpleuvoir\b", "campus_weather"),
        (r"\bprayer\b|\bpri[eè]res?\b|\bsalat\b|\badhan\b", "prayer_times"),
        (r"\bwhat time\b|\bwhat day\b|\bquelle heure\b|\btoday's date\b", "get_time"),
        (r"\d+(\.\d+)?\s*[-+*/x]\s*\d+", "calculator"),
    ]

    def chat(self, messages, *, json_mode=False, temperature=0.2):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        if system.startswith("QUERY REWRITE"):
            text = user  # the mock has no translation ability: it searches with the original words
        else:
            text = json.dumps(self._route(user)) if json_mode else self._answer(messages, system, user)
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        return LLMResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=len(text.split()), model="mock")

    def embed(self, texts):
        raise LLMError("MockLLM has no embedding model; the hashing embedder is used instead")

    def _route(self, question: str) -> dict:
        q = question.lower()
        for pattern, tool in self._ROUTES:
            if not re.search(pattern, q):
                continue
            if tool in ("robot_navigate", "campus_directory"):
                place = re.split(r"\bto\b|\bis\b|\bvers\b|\best\b|\btrouve\b|\bmoi\b", q)[-1]
                place = re.sub(r"^(the|la|le|l')\s*", "", place.strip(" ?.!"))
                return {"tool": tool, "args": {"destination" if tool == "robot_navigate" else "place": place}}
            if tool == "calculator":
                expr = re.search(r"[\d.\s+\-*/x()]+[\d)]", q).group(0).replace("x", "*")
                return {"tool": tool, "args": {"expression": expr.strip()}}
            return {"tool": tool, "args": {}}
        return {"tool": None}

    def _answer(self, messages: list[Message], system: str, user: str) -> str:
        if re.search(r"\bmy name\b|\bmon nom\b", user, re.I) and "?" in user:
            for m in messages[:-1]:
                found = re.search(r"\b(?i:my name is|je m'appelle|i am|i'm)\s+([A-Z][a-z]+)", m["content"])
                if m["role"] == "user" and found:
                    return f"Your name is {found.group(1)}."
            return "I don't know your name yet."
        introduced = re.search(r"\b(?i:my name is|je m'appelle)\s+([A-Z][a-z]+)", user)
        if introduced:
            return f"Nice to meet you, {introduced.group(1)}! How can I help you today?"
        if "TOOL RESULTS:" in system:
            result = system.split("TOOL RESULTS:", 1)[1].strip().splitlines()[0].split("->", 1)[-1].strip()
            if result.startswith(("REFUSED:", "ERROR:")):
                return "Sorry, " + result.split(":", 1)[1].strip()
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                return f"Here is what I found: {result[:240]}"
            if data.get("status") == "navigation_started":
                return f"Follow me to the {data['target']}, it's about {data.get('eta_minutes') or 'a few'} minutes away."
            if "now" in data:
                return f"Right now on campus it's {data['now']['temperature_c']}°C with {data['now']['sky']}."
            if "times" in data:
                return "Today's prayer times in Ben Guerir: " + ", ".join(f"{k} {v}" for k, v in data["times"].items()) + "."
            if "where" in data:
                return f"{data['name']}: {data['where']}. Opening hours: {data['hours']}."
            return f"Here is what I found: {result[:240]}"
        if "CONTEXT:\n[1]" in system:
            passage = system.split("CONTEXT:\n[1]", 1)[1].split("\n", 1)[1]
            passage = re.split(r"\n\n(?:\[2\]|TOOL RESULTS)", passage)[0].strip()
            return f"{self._best_sentences(passage, user)} [1]"
        if "Ari" in system:
            return "I'm Ari, the EMINES assistant robot. I don't have that information, please ask the reception desk."
        return "As an AI language model, I don't have specific information about your school, but most universities have a library and a help desk."

    @staticmethod
    def _best_sentences(passage: str, question: str, n: int = 2) -> str:
        """Extractive 'answer': the n sentences sharing the most words with the question, in passage order."""
        def stems(text: str) -> set[str]:
            return {re.sub(r"(ing|s)$", "", w) for w in re.findall(r"[a-zà-ÿ]{4,}", text.lower())}

        words = stems(question)
        asks_when = bool(re.search(r"\b(when|hours?|open|time|how (long|many))\b", question, re.I))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]
        score = {i: len(words & stems(s)) + (asks_when and bool(re.search(r"\d", s))) for i, s in enumerate(sentences)}
        best = sorted(sorted(score, key=score.get, reverse=True)[:n])
        return " ".join(sentences[i] for i in best)


_REMOTE = {"groq", "gemini"}


def _client(provider: str, base_url: str, model: str, embed_model: str, api_key: str, timeout: float) -> LLM:
    if provider == "keyless":
        return FallbackLLM([(label, OpenAICompatLLM(url, m, "none", timeout=min(timeout, 45), name=label,
                                                    max_tokens=max_tokens, max_retries=0))
                            for label, url, m, max_tokens in KEYLESS_PROVIDERS])
    if provider in _REMOTE and api_key == "not-needed":
        raise LLMError(f"{provider} needs an API key: set {provider.upper()}_API_KEY in .env (see .env.example)")
    if provider == "ollama":
        return OllamaLLM(base_url, model, embed_model, timeout)
    if provider in _REMOTE or provider == "openai":
        return OpenAICompatLLM(base_url, model, embed_model, api_key, timeout, name=provider)
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"unknown provider {provider!r} (expected auto, groq, gemini, keyless, openai, ollama or mock)")


def make_llm(settings: Settings) -> LLM:
    """The chat model."""
    return _client(settings.provider, settings.base_url, settings.model, settings.embed_model,
                   settings.api_key, settings.request_timeout)


def make_embed_client(settings: Settings) -> LLM:
    """The embedding model, which may live on another provider (Groq has none, so Gemini by default)."""
    return _client(settings.embed_provider, settings.embed_base_url, settings.model, settings.embed_model,
                   settings.embed_api_key, settings.request_timeout)
