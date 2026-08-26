# Changelog

## Unreleased

### Added

- Bounded server work concurrency with explicit `429 server_busy` load shedding.
- Configurable HTTP body, ingestion batch, query-size, and heavy-work limits.
- Structured JSON API errors with stable error codes.
- Production deployment and horizontal-scaling guidance.
- Python SDK exceptions for HTTP and connection failures.
- Typed Python package marker, richer package metadata, security/contribution policies, citation metadata, and dependency automation.

### Changed

- CPU/blocking search, indexing, reset, and stats work no longer runs directly on Tokio async worker threads.
- Invalid backend configuration now fails fast at startup.
- CI now performs Rust quality checks, installs/imports the built Python wheel, and executes a real Docker index/search smoke flow in one consolidated workflow.
- Docker Compose exposes capacity limits and applies a read-only, no-new-privileges runtime profile.

## 0.1.0 - 2026-08-25

- Rust adaptive hybrid retrieval core.
- BM25-style lexical, exact dense and sparse retrieval.
- AQF query-aware fusion and retriever-disagreement routing.
- Adaptive lexical early exit and deep reranking path.
- Explainable component scores and metadata filtering.
- Axum HTTP service.
- PyO3 native Python extension and HTTP SDK.
- FastEmbed dense/SPLADE plugin examples and generic reranker protocol.
- Quality/latency benchmark utility and methodology.
