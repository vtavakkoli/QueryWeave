# QueryWeave

**A Rust-first, query-adaptive hybrid retrieval engine for lexical, sparse, dense and neural search.**

QueryWeave is designed around one research and engineering question:

> Can a retrieval engine preserve or improve search quality while avoiding expensive dense, sparse and reranking stages when a query is already easy to answer?

Instead of applying one fixed hybrid formula to every query, QueryWeave analyzes the query and retrieval evidence, chooses a retrieval depth, dynamically fuses lexical/sparse/dense signals, measures retriever disagreement, and invokes deeper reranking only when confidence is weak.

## What makes QueryWeave different

- **Rust adaptive retrieval core** with `#![forbid(unsafe_code)]` in QueryWeave crates.
- **Tantivy BM25** production lexical backend.
- **USearch HNSW** production ANN backend with F32 and I8 quantized modes.
- **Exact cosine** and **built-in BM25-style** reference backends for deterministic ablations.
- **Sparse-vector retrieval** compatible with SPLADE-style learned sparse representations.
- **AQF — Adaptive Query Fusion** with per-query lexical/sparse/dense weights.
- **Retriever-disagreement routing** based on top-k overlap.
- **Adaptive retrieval depth**: lexical early exit → hybrid → deep reranking.
- **Replaceable reranker contract** for ColBERT/MaxSim, cross-encoders, LLM rankers or enterprise services.
- **Metadata pre-filtering**.
- **Explainable results** including route, signal scores, AQF weights, backend names, confidence features and reranker decision.
- **PyO3 native Python extension**.
- **Pure-Python HTTP SDK**.
- **Python model plugins** for dense encoders, learned sparse encoders and rerankers.
- **FastEmbed BGE + SPLADE plugins** included as practical examples.
- **Axum HTTP service** for Java/.NET/Go/Python/JS or remote consumers.
- **Benchmark tooling** for nDCG, MRR, Recall, p50/p95/p99 latency and Pareto-frontier analysis.

## Architecture

```text
                               QUERY
                                 │
                                 ▼
                       Query intelligence
                                 │
               identifiers / length / rarity
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
 Tantivy / BM25            Sparse / SPLADE          Dense / HNSW
       │                         │                         │
       └─────────────────────────┼─────────────────────────┘
                                 ▼
                      Retriever disagreement
                                 │
                                 ▼
                        AQF adaptive fusion
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              high confidence            ambiguous
                    │                         │
              early return              deep reranker
                                              │
                                              ▼
                                      ColBERT / CE / LLM
```

## AQF — Adaptive Query Fusion

Fixed fusion treats very different queries as the same problem. QueryWeave derives features such as:

- token count;
- numeric-token ratio;
- identifier ratio;
- rare-term ratio;
- lexical top-result margin;
- lexical-vs-dense top-k disagreement.

Initial deterministic priors are then adjusted using retrieval evidence. For example:

```text
CVE-2026-12345                -> lexical-heavy
industrial pump failure      -> balanced hybrid
explain causes of pump wear  -> dense-heavy
retrievers strongly disagree -> deeper fusion/reranking
```

The deterministic policy is intentionally replaceable by a learned fusion predictor in future experiments while preserving the same explanation contract.

## Adaptive retrieval depth

`mode="auto"` chooses among:

1. **Lexical early exit** — exact/identifier query with a strong lexical margin.
2. **Hybrid** — lexical + sparse + dense AQF.
3. **Deep** — AQF plus reranking when retrievers disagree or confidence is low.

For controlled experiments you can force:

```text
lexical
hybrid
deep
auto
```

## Backend matrix

The HTTP server defaults to the production-oriented configuration:

```text
QUERYWEAVE_LEXICAL_BACKEND=tantivy
QUERYWEAVE_VECTOR_BACKEND=hnsw-f32
```

Available backends:

| Layer | Backend | Purpose |
|---|---|---|
| lexical | `tantivy` | production BM25/inverted index |
| lexical | `builtin` / `bm25` | deterministic lightweight baseline |
| vector | `hnsw-f32` | production HNSW cosine ANN |
| vector | `hnsw-i8` | memory-efficient quantized HNSW |
| vector | `exact` | exact cosine recall/reference baseline |

This makes it possible to measure ANN recall and quantization trade-offs against exact retrieval without changing AQF.

## Run the server

```bash
docker compose up --build
```

Server:

```text
http://localhost:7777
```

Health includes the active backends:

```bash
curl http://localhost:7777/health
```

Example response:

```json
{
  "status": "ok",
  "engine": "QueryWeave",
  "version": "0.1.0",
  "lexical_backend": "tantivy-bm25",
  "vector_backend": "usearch-hnsw-f32"
}
```

### Index documents

```bash
curl -X POST http://localhost:7777/v1/documents:upsert \
  -H 'content-type: application/json' \
  -d '{
    "documents": [
      {
        "id": "doc-1",
        "text": "Hydraulic pump temperature failure",
        "source": "manual.pdf"
      }
    ]
  }'
```

### Search

```bash
curl -X POST http://localhost:7777/v1/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "prevent pump failure caused by excessive heat",
    "limit": 10,
    "mode": "auto",
    "explain": true
  }'
```

## External vectors for fair benchmarking

For serious hybrid search, pass the same production vectors used by competing engines:

```json
{
  "query": "industrial pump failure",
  "dense": [0.12, -0.08, 0.31],
  "sparse": {
    "indices": [44, 901],
    "values": [1.7, 0.8]
  },
  "limit": 10,
  "mode": "auto",
  "filter": {"language": "en"},
  "explain": true
}
```

This contract is central to QueryWeave benchmarking: QueryWeave, Qdrant and Elasticsearch can receive the **same chunks, BGE vectors and SPLADE vectors** rather than accidentally benchmarking different representation models.

## Python: native engine + ML plugins

Build the PyO3 extension:

```bash
python -m pip install 'maturin>=1.9,<2.0'
maturin develop
```

Use external dense/sparse models:

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

The native Python extension intentionally uses the dependency-light core backends. For the production Tantivy/HNSW configuration from Python, use `QueryWeaveClient` against the server while still generating BGE/SPLADE vectors in Python.

### Reranker plugins

A Python reranker only needs:

```python
class MyReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        ...
```

This lets research code integrate:

- ColBERT / MaxSim;
- cross-encoders;
- ONNX rankers;
- LLM rerankers;
- remote enterprise ranking APIs.

## Rust API

The core defaults to lightweight correctness backends:

```rust
use queryweave_core::{
    Document, Metadata, QueryWeaveEngine, SearchMode, SearchRequest,
};

let engine = QueryWeaveEngine::new();

engine.upsert(vec![Document {
    id: "doc-1".into(),
    text: "Hydraulic pump temperature failure".into(),
    source: "manual.pdf".into(),
    metadata: Metadata::new(),
    dense: None,
    sparse: None,
}]);

let result = engine.search(SearchRequest {
    query: "pump failure".into(),
    limit: 10,
    mode: SearchMode::Auto,
    dense: None,
    sparse: None,
    filter: Metadata::new(),
    explain: true,
});
```

Production applications can construct `QueryWeaveEngine::with_backends(...)` with `TantivyLexicalIndex` and `USearchHnswIndex`.

## Explainability

A QueryWeave hit can expose:

```json
{
  "score": 0.894,
  "components": {
    "lexical": 0.73,
    "sparse": 0.84,
    "dense": 0.91,
    "rerank": 0.0
  },
  "explanation": {
    "route": "hybrid",
    "weights": {
      "lexical": 0.24,
      "sparse": 0.31,
      "dense": 0.45
    },
    "early_exit": false,
    "reranked": false,
    "lexical_backend": "tantivy-bm25",
    "vector_backend": "usearch-hnsw-f32"
  }
}
```

## Benchmark design

Do not benchmark only product names. First isolate the algorithms:

| ID | Method |
|---|---|
| A | BM25 only |
| B | Dense only |
| C | Sparse only |
| D | BM25 + Dense RRF |
| E | lexical + sparse + dense fixed fusion |
| F | **QueryWeave AQF** |
| G | AQF + always-on reranking |
| H | **AQF + adaptive reranking** |
| I | AQF + exact cosine |
| J | AQF + HNSW F32 |
| K | AQF + HNSW I8 |

Then compare engines using identical corpus/chunks/vectors:

- QueryWeave;
- Qdrant;
- Elasticsearch;
- Vespa;
- LanceDB;
- Meilisearch.

The companion `vtavakkoli/Hybrid-Search` repository provides the first live comparison dashboard for **QueryWeave vs Qdrant vs Elasticsearch**.

Recommended datasets include BEIR NFCorpus, SciFact, FiQA, TREC-COVID, ArguAna and DBPedia, with MS MARCO for larger-scale throughput experiments.

## Metrics

Quality:

- nDCG@10
- MRR@10
- Recall@100
- MAP

Efficiency:

- p50 / p95 / p99 latency
- QPS
- ingestion docs/s
- resident RAM
- index size
- CPU/query
- reranker invocation rate
- HNSW recall vs exact
- F32 vs I8 memory/quality trade-off

The main scientific target is the **quality-versus-cost Pareto frontier**, not an unsupported claim that one engine is universally faster.

## Workspace

```text
crates/
  queryweave-core/       AQF, routing, filtering, explanations, baseline backends
  queryweave-tantivy/    Tantivy BM25 backend
  queryweave-usearch/    USearch HNSW F32/I8 backend
  queryweave-server/     Axum HTTP service
  queryweave-python/     PyO3 extension
python/queryweave/       HTTP SDK + ML plugin protocols
benchmarks/              normalized quality/latency evaluator
docs/                    architecture and benchmark methodology
```

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Python wheel:

```bash
maturin build --release
```

## License

MIT. See `LICENSE`.
