"""Grounding: chunk the knowledge base, embed it, and retrieve passages by cosine similarity.

The vector store is a NumPy matrix persisted to disk: a few hundred chunks need no database.
The original Ari used AstraDB; `VectorStore` is the single class to swap for a hosted store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .llm import LLM, LLMError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def load_chunks(knowledge_dir: Path, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    """One chunk per Markdown section; long sections are split into overlapping windows."""
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        sections: list[tuple[str, list[str]]] = [(path.stem, [])]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                sections.append((line.lstrip("#").strip(), []))
            else:
                sections[-1][1].append(line)
        n = 0
        for heading, lines in sections:
            body = "\n".join(lines).strip()
            start = 0
            while body and start < len(body):
                piece = body[start:start + max_chars]
                chunks.append(Chunk(f"{path.stem}#{n}", path.name, heading, piece))
                n += 1
                start += max_chars - overlap
    return chunks


class HashingEmbedder:
    """Dependency-free fallback: signed feature hashing of word unigrams and bigrams.
    Lexical, not semantic, but deterministic and good enough for tests and offline demos."""

    id = "hashing-v1"
    default_min_score = 0.07
    _STOP = set("the a an of to in on for and or is are be with at by it this that what where when how do does can i you my me your "
                "le la les de des du un une et ou est sont pour dans sur au aux il elle je tu vous".split())

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w.rstrip("s") if len(w) > 4 else w for w in re.findall(r"[a-zà-ÿ0-9]+", text.lower()) if w not in self._STOP]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class LLMEmbedder:
    """Embeddings from the model server (e.g. nomic-embed-text on Ollama or LM Studio)."""

    default_min_score = 0.55

    def __init__(self, llm: LLM, batch_size: int = 16):
        self.llm = llm
        self.batch_size = batch_size
        self.id = f"{llm.name}:{llm.embed_model}"
        # nomic-embed models are trained with task prefixes; other models ignore this.
        nomic = "nomic" in llm.embed_model.lower()
        self._doc_prefix = "search_document: " if nomic else ""
        self._query_prefix = "search_query: " if nomic else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows += self.llm.embed([self._doc_prefix + t for t in texts[i:i + self.batch_size]])
        return _normalize(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.asarray(self.llm.embed([self._query_prefix + text]), dtype=np.float32))[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def make_embedder(llm: LLM) -> HashingEmbedder | LLMEmbedder:
    if llm.name == "mock":
        return HashingEmbedder()
    embedder = LLMEmbedder(llm)
    try:
        embedder.embed_query("ping")
        return embedder
    except (LLMError, KeyError) as exc:
        log.warning("Embedding model %r unavailable (%s); falling back to the hashing embedder", llm.embed_model, exc)
        return HashingEmbedder()


class VectorStore:
    def __init__(self, embedder: HashingEmbedder | LLMEmbedder, index_dir: Path):
        self.embedder = embedder
        self.index_dir = index_dir
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)

    def build_or_load(self, chunks: list[Chunk]) -> "VectorStore":
        """Reuse the on-disk index when neither the documents nor the embedder changed."""
        fingerprint = hashlib.sha256(
            (self.embedder.id + "".join(c.id + c.heading + c.text for c in chunks)).encode()
        ).hexdigest()
        meta_path, matrix_path = self.index_dir / "meta.json", self.index_dir / "vectors.npy"
        if meta_path.exists() and matrix_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint:
                self.chunks = [Chunk(**c) for c in meta["chunks"]]
                self.matrix = np.load(matrix_path)
                return self
        log.info("Indexing %d chunks with %s", len(chunks), self.embedder.id)
        self.chunks = chunks
        self.matrix = self.embedder.embed_documents([f"{c.heading}\n{c.text}" for c in chunks])
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, self.matrix)
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "embedder": self.embedder.id,
                                         "chunks": [asdict(c) for c in chunks]}, ensure_ascii=False), encoding="utf-8")
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.matrix @ self.embedder.embed_query(query)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]
