"""QueryWeave Python SDK.

The package supports both the native PyO3 engine and the standalone HTTP service. Vector,
sparse, and reranking models remain Python plugins so research users can swap FastEmbed,
SentenceTransformers, SPLADE, ColBERT, or proprietary models without recompiling Rust.
"""

from .client import QueryWeaveClient
from .engine import QueryWeave
from .plugins import EmbedderPlugin, SparseEncoderPlugin, RerankerPlugin

__all__ = ["QueryWeave", "QueryWeaveClient", "EmbedderPlugin", "SparseEncoderPlugin", "RerankerPlugin"]
__version__ = "0.1.0"
