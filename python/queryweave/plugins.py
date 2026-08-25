from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbedderPlugin(Protocol):
    """Dense embedding provider. Implement with FastEmbed, ONNX, Torch, an API, etc."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class SparseEncoderPlugin(Protocol):
    """Sparse encoder provider (SPLADE, BM25 expansion, learned sparse model, etc.)."""

    def encode(self, texts: Sequence[str]) -> list[dict[str, list]]: ...


@runtime_checkable
class RerankerPlugin(Protocol):
    """Late-interaction or cross-encoder plugin.

    Return one score per candidate. Implementations can wrap ColBERT/MaxSim, a cross encoder,
    an LLM reranker, or an enterprise ranking service.
    """

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedDense:
    """Optional BGE-family plugin backed by FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(list(texts))]


class FastEmbedSparse:
    """Optional SPLADE plugin backed by FastEmbed."""

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: Sequence[str]) -> list[dict[str, list]]:
        output = []
        for vector in self._model.embed(list(texts)):
            output.append({"indices": vector.indices.tolist(), "values": vector.values.tolist()})
        return output
