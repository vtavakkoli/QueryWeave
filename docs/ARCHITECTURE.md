# QueryWeave architecture

## Design goals

QueryWeave separates **representation**, **retrieval**, **fusion**, and **reranking**. This prevents a benchmark from silently changing both the search engine and the model at the same time.

### Representation layer

Documents and queries may carry externally generated dense and sparse vectors. If omitted, deterministic hash encoders provide a zero-model baseline suitable for unit tests and algorithmic smoke tests.

Python plugins are the recommended place for FastEmbed, SentenceTransformers, SPLADE, ColBERT, ONNX Runtime, remote embedding APIs, or organization-specific models.

### Retrieval layer

- lexical: native BM25-style scorer;
- dense: `VectorIndex` trait, exact cosine reference backend;
- sparse: sorted sparse-vector dot product;
- metadata: pre-scoring equality filters.

The exact vector backend is a correctness baseline. HNSW/IVF/PQ/USearch backends should implement `VectorIndex` and can then be benchmarked against exact recall.

### Adaptive layer

AQF derives weights from query features and retrieval feedback. Retriever disagreement is currently top-k Jaccard distance. Future predictors can replace the deterministic policy while keeping the explanation schema stable.

### Reranking layer

`Reranker` is a Rust trait and `RerankerPlugin` is its Python-side equivalent. Deep mode is triggered by uncertainty instead of always paying the reranker cost.

### Explanation contract

Every hit can expose:

- lexical, sparse, dense and rerank scores;
- AQF weights;
- query features;
- route;
- candidate-pool size;
- early-exit decision;
- reranker decision/name.

This is intended for research ablations, debugging and cost accounting.
