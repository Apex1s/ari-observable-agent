"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:1234/v1",  # any OpenAI-compatible server (LM Studio, vLLM, ...)
    "mock": "",
}
_KEY_VARS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
# Remote by default: chat on Groq (Llama 3.3 70B), embeddings on Gemini. Nothing runs on the laptop.
_DEFAULT_MODELS = {"groq": "llama-3.3-70b-versatile", "gemini": "gemini-2.5-flash", "ollama": "llama3.2:3b"}
_DEFAULT_EMBED_MODELS = {"gemini": "gemini-embedding-001", "ollama": "nomic-embed-text"}


def _api_key(provider: str, explicit: str | None) -> str:
    return explicit or os.environ.get(_KEY_VARS.get(provider, ""), "") or "not-needed"


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, never overrides real environment variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str
    model: str
    embed_model: str
    api_key: str
    embed_provider: str
    embed_base_url: str
    embed_api_key: str
    knowledge_dir: Path
    data_dir: Path
    min_retrieval_score: float | None  # None = use the embedder's calibrated default
    top_k: int
    memory_turns: int
    request_timeout: float

    @property
    def trace_db(self) -> Path:
        return self.data_dir / "traces.db"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def campus_file(self) -> Path:
        return self.knowledge_dir / "data" / "campus.json"

    @property
    def schedule_file(self) -> Path:
        return self.knowledge_dir / "data" / "schedule.json"


def load_settings(**overrides) -> Settings:
    _load_dotenv(ROOT / ".env")
    provider = overrides.pop("provider", None) or os.environ.get("ARI_PROVIDER", "groq")
    # Embeddings default to Gemini when chat is on Groq (Groq serves no embedding model).
    embed_provider = overrides.pop("embed_provider", None) or (
        "mock" if provider == "mock"  # tests and CI stay fully offline
        else os.environ.get("ARI_EMBED_PROVIDER") or ("gemini" if provider == "groq" else provider))
    min_score = os.environ.get("ARI_MIN_SCORE")
    values = dict(
        provider=provider,
        base_url=os.environ.get("ARI_BASE_URL") or _DEFAULT_URLS.get(provider, ""),
        model=os.environ.get("ARI_MODEL") or _DEFAULT_MODELS.get(provider, ""),
        embed_model=os.environ.get("ARI_EMBED_MODEL") or _DEFAULT_EMBED_MODELS.get(embed_provider, ""),
        api_key=_api_key(provider, os.environ.get("ARI_API_KEY")),
        embed_provider=embed_provider,
        embed_base_url=os.environ.get("ARI_EMBED_BASE_URL") or _DEFAULT_URLS.get(embed_provider, ""),
        embed_api_key=_api_key(embed_provider, os.environ.get("ARI_EMBED_API_KEY")),
        knowledge_dir=Path(os.environ.get("ARI_KNOWLEDGE_DIR", ROOT / "knowledge")),
        data_dir=Path(os.environ.get("ARI_DATA_DIR", ROOT / "data")),
        min_retrieval_score=float(min_score) if min_score else None,
        top_k=int(os.environ.get("ARI_TOP_K", "3")),
        memory_turns=int(os.environ.get("ARI_MEMORY_TURNS", "6")),
        request_timeout=float(os.environ.get("ARI_TIMEOUT", "120")),
    )
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
