"""Grounding: chunk the knowledge base, embed it, and retrieve passages by cosine similarity.

The vector store is a NumPy matrix persisted to disk: a few hundred chunks need no database.
The original Ari used AstraDB; `VectorStore` is the single class to swap for a hosted store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import unicodedata
from collections import Counter
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
    """Embeddings from an API (gemini-embedding-001 by default, or nomic-embed-text on Ollama)."""

    # Cosine scores are not comparable across embedding models, so the "is this passage relevant at
    # all" cut-off is per model. nomic: measured on this knowledge base (relevant 0.56-0.87, off-topic
    # 0.51-0.60). Gemini: provisional until `python -m ari calibrate` has been run with a key.
    _MIN_SCORES = {"nomic": 0.55, "gemini-embedding": 0.65}
    default_min_score = 0.6

    def __init__(self, llm: LLM, batch_size: int = 16):
        self.llm = llm
        self.batch_size = batch_size
        self.id = f"{llm.name}:{llm.embed_model}"
        self.default_min_score = next((v for k, v in self._MIN_SCORES.items() if k in llm.embed_model.lower()),
                                      LLMEmbedder.default_min_score)
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


_URL = re.compile(r"(?:Sources?:\s*)?https?://\S+")


def _fold(text: str) -> str:
    """Lowercase and strip accents so 'bibliothèque' matches 'bibliotheque'."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch))


class BM25Store:
    """Keyless retrieval: Okapi BM25 over the chunks, no model and no API, so it costs nothing to run.

    Scores are normalised to [0, 1]: the share of the query's information (IDF mass) that the passage
    matches, so a fixed threshold works across questions. Query words absent from every document keep
    their IDF, which is what pushes off-topic questions ("World Cup") below the threshold.
    """

    id = "bm25-v1"
    default_min_score = 0.12  # measured on the real EMINES/UM6P corpus: in-scope top hits >= 0.12, "World Cup" 0.10
    _STOP = set(_fold(
        "the a an of to in on for and or is are be was were with at by it this that what where when how who which do does "
        "did can could i you my me your we our there their about from as any some please tell know "
        "le la les de des du un une et ou est sont pour dans sur au aux il elle je tu vous nous quel quelle quels quelles "
        "est-ce que qui quoi comment ou mon ma mes ton ta votre vos ce cette ces y a avez peut").split())

    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.embedder = self  # health/report code reads store.embedder.id

    def _tokens(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", _fold(_URL.sub(" ", text)))
        return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if w not in self._STOP and len(w) > 1]

    def build_or_load(self, chunks: list[Chunk]) -> "BM25Store":
        self.chunks = chunks
        self._docs = [Counter(self._tokens(f"{c.heading} {c.heading} {c.text}")) for c in chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / max(len(self._docs), 1)
        df = Counter(t for d in self._docs for t in d)
        n = len(self._docs)
        self._idf = lambda t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = set(self._tokens(query))
        if not self.chunks or not terms:
            return []
        best_possible = sum(self._idf(t) * (self.k1 + 1) for t in terms)
        scored = []
        for chunk, doc, length in zip(self.chunks, self._docs, self._lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self._idf(t) * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self._avg))
            scored.append(Hit(chunk, s / best_possible))
        return sorted(scored, key=lambda h: -h.score)[:k]


def make_store(embed_client: LLM | None, index_dir: Path, chunks: list[Chunk]) -> "VectorStore | BM25Store":
    """Embeddings when an embedding API answers; otherwise keyless BM25. The mock keeps its hashing embedder."""
    embedder = make_embedder(embed_client) if embed_client is not None else None
    if embedder is None or (isinstance(embedder, HashingEmbedder) and embed_client.name != "mock"):
        log.warning("No embedding API available: using keyless BM25 retrieval")
        return BM25Store().build_or_load(chunks)
    return VectorStore(embedder, index_dir / embedder.id.replace(":", "_").replace("/", "_")).build_or_load(chunks)


class VectorStore:
    def __init__(self, embedder: HashingEmbedder | LLMEmbedder, index_dir: Path):
        self.embedder = embedder
        self.index_dir = index_dir
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)

    @property
    def default_min_score(self) -> float:
        return self.embedder.default_min_score

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
