## What changed

Describe the problem and the implementation.

## Validation

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] Python/package behavior tested when affected
- [ ] Docker/API smoke flow tested when affected

## Retrieval / benchmark impact

If retrieval behavior or performance changes, include the dataset, corpus size, model/vector configuration, backend settings, hardware, quality metrics, and latency/throughput measurements needed to reproduce the result.

## Compatibility

Describe public API, configuration, serialization, persistence, or deployment compatibility considerations. Use `N/A` if none.

## Documentation

- [ ] User-facing behavior/configuration is documented
- [ ] `CHANGELOG.md` updated when appropriate
