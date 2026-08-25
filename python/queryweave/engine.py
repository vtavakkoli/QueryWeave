from __future__ import annotations

import json
from typing import Any, Sequence

from .plugins import EmbedderPlugin, RerankerPlugin, SparseEncoderPlugin

try:
    from ._native import Engine as NativeEngine
except ImportError:  # source checkout without a built wheel
    NativeEngine = None


class QueryWeave:
    """Python-friendly native QueryWeave engine with pluggable ML models."""

    def __init__(
        self,
        *,
        embedder: EmbedderPlugin | None = None,
        sparse_encoder: SparseEncoderPlugin | None = None,
        reranker: RerankerPlugin | None = None,
    ) -> None:
        if NativeEngine is None:
            raise RuntimeError("QueryWeave native module is not built. Run `maturin develop` or use QueryWeaveClient.")
        self._native = NativeEngine()
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.reranker = reranker

    def upsert(self, documents: Sequence[dict[str, Any]]) -> int:
        docs = [dict(document) for document in documents]
        texts = [str(d["text"]) for d in docs]
        if self.embedder:
            for d, vector in zip(docs, self.embedder.embed(texts), strict=True):
                d["dense"] = vector
        if self.sparse_encoder:
            for d, vector in zip(docs, self.sparse_encoder.encode(texts), strict=True):
                d["sparse"] = vector
        return self._native.upsert_json(json.dumps(docs))

    def search(self, query: str, limit: int = 10, *, mode: str = "auto", filters: dict[str, str] | None = None) -> dict[str, Any]:
        dense = self.embedder.embed([query])[0] if self.embedder else None
        sparse = self.sparse_encoder.encode([query])[0] if self.sparse_encoder else None
        payload = {"query": query, "limit": limit, "mode": mode, "filter": filters or {}, "explain": True, "dense": dense, "sparse": sparse}
        result = json.loads(self._native.search_json(json.dumps(payload)))
        if self.reranker and result["hits"]:
            scores = self.reranker.score(query, [h["text"] for h in result["hits"]])
            for hit, rerank in zip(result["hits"], scores, strict=True):
                hit["components"]["rerank"] = float(rerank)
                hit["score"] = 0.8 * hit["score"] + 0.2 * float(rerank)
                if hit.get("explanation"):
                    hit["explanation"]["reranked"] = True
                    hit["explanation"]["reranker"] = type(self.reranker).__name__
            result["hits"].sort(key=lambda h: h["score"], reverse=True)
        return result

    def stats(self) -> dict[str, Any]:
        return json.loads(self._native.stats_json())

    def reset(self) -> None:
        self._native.reset()
