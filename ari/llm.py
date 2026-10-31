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
                 name: str = "openai"):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def chat(self, messages, *, json_mode=False, temperature=0.2):
        # json_mode is enforced by the prompt: response_format support varies across local servers.
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        r = _post_json(f"{self.base_url}/chat/completions", payload, self.timeout, self._headers)
        usage = r.get("usage") or {}
        return LLMResponse(
            text=r["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=r.get("model", self.model),
        )

    def embed(self, texts):
        r = _post_json(f"{self.base_url}/embeddings", {"model": self.embed_model, "input": texts}, self.timeout, self._headers)
        return [item["embedding"] for item in sorted(r["data"], key=lambda d: d["index"])]


class MockLLM:
    """Rule-based stand-in for a model. It reads the same prompts a real model would get, so every
    stage of the pipeline (routing, grounding, memory, guardrails, tracing) is exercised offline."""

    name = "mock"
    model = "mock"
    embed_model = "hashing"

    _ROUTES = [
        (r"\b(take|guide|bring|walk|accompany|lead) me\b|\bemm[eè]ne[- ]moi\b|\bconduis[- ]moi\b", "robot_navigate"),
        (r"\bwhere is\b|\bwhere's\b|\bo[uù] (est|se trouve)\b", "campus_directory"),
        (r"\bschedule\b|\btimetable\b|\bemploi du temps\b|\bclasses\b", "class_schedule"),
        (r"\bwhat time\b|\bwhat day\b|\bquelle heure\b|\btoday's date\b", "get_time"),
        (r"\d+(\.\d+)?\s*[-+*/x]\s*\d+", "calculator"),
    ]
    _DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

    def chat(self, messages, *, json_mode=False, temperature=0.2):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
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
            if tool == "class_schedule":
                day = next((d for d in self._DAYS if d in q), "tomorrow" if "tomorrow" in q else "today")
                program = "3A-DS" if re.search(r"3a|data science|third", q) else "1A"
                return {"tool": tool, "args": {"program": program, "day": day}}
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
            result = system.split("TOOL RESULTS:", 1)[1].strip().splitlines()[0]
            return f"Here is what I found: {result.split('->', 1)[-1].strip()[:240]}"
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
    if provider in _REMOTE and api_key == "not-needed":
        raise LLMError(f"{provider} needs an API key: set {provider.upper()}_API_KEY in .env (see .env.example)")
    if provider == "ollama":
        return OllamaLLM(base_url, model, embed_model, timeout)
    if provider in _REMOTE or provider == "openai":
        return OpenAICompatLLM(base_url, model, embed_model, api_key, timeout, name=provider)
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"unknown provider {provider!r} (expected groq, gemini, openai, ollama or mock)")


def make_llm(settings: Settings) -> LLM:
    """The chat model."""
    return _client(settings.provider, settings.base_url, settings.model, settings.embed_model,
                   settings.api_key, settings.request_timeout)


def make_embed_client(settings: Settings) -> LLM:
    """The embedding model, which may live on another provider (Groq has none, so Gemini by default)."""
    return _client(settings.embed_provider, settings.embed_base_url, settings.model, settings.embed_model,
                   settings.embed_api_key, settings.request_timeout)
