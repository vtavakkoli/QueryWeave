# QueryWeave

**Query-adaptive hybrid retrieval in Rust: lexical + sparse + dense + confidence-triggered reranking.**

QueryWeave is a Rust-first retrieval engine and research library designed to answer a practical question:

> Can a search system match or improve always-on hybrid/reranking quality while spending less compute on easy queries?

Instead of applying one fixed fusion recipe to every query, QueryWeave extracts query and retrieval signals, chooses a route, learns/derives per-query fusion weights, measures retriever disagreement, and only takes the expensive path when the evidence is ambiguous.

## Highlights

- **Rust core** with no unsafe code.
- **BM25-style lexical retrieval** for exact terms, identifiers, entities and rare tokens.
- **Dense retrieval** with a replaceable `VectorIndex` contract; the reference backend is exact cosine search for deterministic baselines.
- **Sparse retrieval** with explicit sparse-vector input; Python plugins include a FastEmbed/SPLADE adapter.
- **AQF — Adaptive Query Fusion** that changes lexical/sparse/dense weights from query characteristics and retrieval confidence.
- **Retriever disagreement** measured from top-k overlap.
- **Adaptive retrieval depth**: lexical early exit, hybrid fusion, or deep reranking.
- **Late-interaction hook** with a deterministic built-in proxy and replaceable ColBERT/cross-encoder/LLM reranker contract.
- **Metadata filtering** before scoring.
- **Explainable results** with component scores, weights, route, features, candidate-pool size, and reranker decision.
- **PyO3 native Python bindings** plus a pure-Python HTTP SDK.
- **Python ML plugins** for dense embeddings, learned sparse encoders and rerankers.
- **Axum HTTP service** for language-neutral integration.
- **Benchmark tooling** for nDCG, MRR, Recall, latency and Pareto-frontier experiments.

## Architecture

```text
                               query
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Query feature analysis │
                    └────────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        lexical/BM25        sparse/SPLADE        dense/vector
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    retriever disagreement
                                 │
                                 ▼
                       AQF adaptive fusion
                                 │
                    ┌────────────┴────────────┐
                    │                         │
               high confidence          ambiguous query
                    │                         │
                    ▼                         ▼
                 return              late interaction /
                                     cross-encoder plugin
                                             │
                                             ▼
                                           return
```

## AQF: Adaptive Query Fusion

Fixed `0.5 * BM25 + 0.5 * dense` weights treat a CVE identifier and a natural-language semantic query as if they were the same retrieval problem. QueryWeave instead derives features including:

- query length;
- numeric-token ratio;
- identifier ratio;
- rare-term ratio;
- lexical top-result margin;
- lexical-vs-dense top-k disagreement.

The result is a normalized per-query weight vector:

```text
exact identifier       -> lexical 0.62 / sparse 0.28 / dense 0.10
long semantic question -> lexical 0.18 / sparse 0.27 / dense 0.55
mixed query            -> lexical 0.30 / sparse 0.30 / dense 0.40
```

These are initial deterministic priors, not universal constants. The public score/fusion contracts are designed for learned weight predictors in later experiments.

## Adaptive retrieval depth

`mode="auto"` can take three paths:

1. **Lexical early exit** for high-confidence identifier/exact-term queries.
2. **Hybrid** lexical+sparse+dense fusion for normal queries.
3. **Deep** fusion + reranking when the retrievers disagree or confidence is weak.

You can force a path with `lexical`, `hybrid`, or `deep` for ablations.

## Rust API

```rust
use queryweave_core::{Document, Metadata, QueryWeaveEngine, SearchMode, SearchRequest};

let engine = QueryWeaveEngine::new();
engine.upsert(vec![Document {
    id: "doc-1".into(),
    text: "Hydraulic pump temperature failure".into(),
    source: "manual.pdf".into(),
    metadata: Metadata::new(),
    dense: None,   // built-in deterministic baseline if omitted
    sparse: None,  // built-in deterministic baseline if omitted
}]);

let response = engine.search(SearchRequest {
    query: "prevent pump failure caused by excessive heat".into(),
    limit: 10,
    mode: SearchMode::Auto,
    dense: None,
    sparse: None,
    filter: Metadata::new(),
    explain: true,
});
```

For real semantic search, pass vectors generated by your production embedding and sparse models. QueryWeave does not force a model vendor.

## HTTP service

```bash
docker compose up --build
```

Server: `http://localhost:7777`

### Upsert

```bash
curl -X POST http://localhost:7777/v1/documents:upsert \
  -H 'content-type: application/json' \
  -d '{"documents":[{"id":"1","text":"pump failure","source":"manual"}]}'
```

### Search

```bash
curl -X POST http://localhost:7777/v1/search \
  -H 'content-type: application/json' \
  -d '{"query":"pump failure","limit":5,"mode":"auto","explain":true}'
```

## Python native engine

```bash
python -m pip install maturin
maturin develop
```

```python
from queryweave import QueryWeave
from queryweave.plugins import FastEmbedDense, FastEmbedSparse

engine = QueryWeave(
    embedder=FastEmbedDense("BAAI/bge-small-en-v1.5"),
    sparse_encoder=FastEmbedSparse("prithivida/Splade_PP_en_v1"),
)

engine.upsert([
    {"id": "1", "text": "hydraulic pump failure", "source": "manual.pdf"},
    {"id": "2", "text": "coffee shop opening hours", "source": "city.txt"},
])

result = engine.search("prevent industrial pump failure")
```

A reranker only needs to implement:

```python
class MyReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        ...
```

so ColBERT/MaxSim, cross-encoders, LLM rerankers and enterprise APIs can be swapped without recompiling the Rust engine.

## External-vector contract

QueryWeave accepts dense and sparse vectors in both documents and queries. This is important for fair evaluation: Qdrant, Elasticsearch and QueryWeave can receive **the same BGE/SPLADE vectors** rather than accidentally benchmarking different models.

```json
{
  "query": "industrial pump failure",
  "dense": [0.12, -0.08, 0.31],
  "sparse": {"indices": [44, 901], "values": [1.7, 0.8]},
  "limit": 10,
  "mode": "auto",
  "filter": {"language": "en"},
  "explain": true
}
```

## Benchmark matrix

The recommended benchmark compares retrieval algorithms separately before comparing products:

| ID | Retrieval method |
|---|---|
| A | BM25 only |
| B | Dense only |
| C | Sparse only |
| D | BM25 + Dense RRF |
| E | BM25 + Sparse + Dense fixed fusion |
| F | **QueryWeave AQF** |
| G | AQF + always-on reranker |
| H | **AQF + adaptive reranking** |

Then compare product implementations under the same corpus, chunks and vectors:

- QueryWeave
- Qdrant
- Elasticsearch
- Vespa
- LanceDB
- Meilisearch

See `docs/BENCHMARKING.md`.

## Metrics

Quality:

- nDCG@10
- MRR@10
- Recall@100
- MAP

Performance:

- p50 / p95 / p99 latency
- throughput (QPS)
- ingestion docs/s
- RAM
- index size
- CPU/query
- reranker invocation rate

The flagship result should be a **quality-versus-cost Pareto frontier**, not a claim that one engine is universally faster.

## Workspace

```text
crates/
  queryweave-core/      adaptive retrieval/fusion engine
  queryweave-server/    Axum REST API
  queryweave-python/    PyO3 extension
python/queryweave/      Python SDK + model plugin protocols
benchmarks/             quality/latency evaluation utilities
docs/                   architecture and benchmark methodology
```

## Current backend status

QueryWeave v0.1 intentionally uses an **exact dense vector backend** as the built-in reference because it gives deterministic recall and a clean correctness baseline. The `VectorIndex` trait is the stable seam for HNSW, IVF/PQ, USearch or hardware-specific ANN backends. Likewise, production SPLADE and ColBERT live behind plugin contracts rather than being hard-coded into the Rust core.

This separation is deliberate: retrieval-policy experiments should not be confounded by one mandatory ML runtime.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Python:

```bash
python -m pip install -e '.[dev]'
maturin develop
pytest
```

## License

MIT. See `LICENSE`.
