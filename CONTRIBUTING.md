# Contributing to QueryWeave

Thank you for contributing to QueryWeave. The project aims to keep research ideas reproducible while maintaining production-quality engineering standards.

## Development setup

Requirements:

- Rust toolchain compatible with the workspace `rust-version`;
- Python 3.10+ for the Python SDK and PyO3 extension;
- Docker for container smoke tests.

Run the core checks before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For the Python extension:

```bash
python -m pip install 'maturin>=1.9,<2.0'
maturin develop
```

## Design principles

Changes should preserve these boundaries:

1. **Core policy is backend-independent.** AQF, routing, filtering, explanations, and public search contracts belong in `queryweave-core`.
2. **Backends stay replaceable.** Tantivy, USearch, future ANN engines, model runtimes, and rerankers should implement explicit contracts rather than leak backend details into routing logic.
3. **Benchmarks compare like with like.** Use the same corpus, chunks, embeddings, sparse vectors, and relevance judgments when comparing engines.
4. **No unsupported performance claims.** Report hardware, dataset, concurrency, quality, and tail latency with benchmark results.
5. **Prefer safe Rust.** QueryWeave crates use `#![forbid(unsafe_code)]`; backend dependencies may contain unsafe internals, but QueryWeave-owned Rust should remain safe unless the project explicitly revisits this policy.
6. **Protect async runtimes.** CPU-heavy or blocking retrieval/indexing work in the HTTP service must not execute directly on Tokio async worker threads.

## Pull requests

Keep PRs focused and include:

- the problem being solved;
- the design/behavior change;
- tests or benchmark evidence where appropriate;
- compatibility notes for public API changes;
- documentation for new configuration or user-facing behavior.

Public API changes should be additive whenever practical. Breaking changes should explain migration steps and normally be reserved for a version boundary.

## Benchmark contributions

A benchmark result should include enough information to reproduce it: dataset/version, corpus size, model names, vector dimensions, backend configuration, hardware, warmup, run count, concurrency, quality metrics, latency percentiles, and memory usage.

Do not compare QueryWeave against another engine while silently changing representation models or chunking strategies between systems.

## Commit style

Conventional-style subjects are encouraged, for example:

```text
feat(core): add learned fusion policy hook
fix(server): preserve responsiveness under index rebuild
perf(usearch): reduce filtered-search allocation
docs: document benchmark methodology
```

## Security issues

Please do not open public issues for vulnerabilities. Follow `SECURITY.md` instead.
