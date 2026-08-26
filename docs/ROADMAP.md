# QueryWeave roadmap

QueryWeave 0.1 establishes the adaptive retrieval core, production lexical/ANN backends, Python integration, HTTP service, benchmarking methodology, and bounded service backpressure. The roadmap below focuses on turning that foundation into a larger-scale retrieval platform without hiding current limitations.

## 0.2 — index lifecycle and ingestion scale

- incremental lexical/vector index updates instead of full backend rebuilds for every upsert batch;
- O(1) document-ID lookup for updates and explicit document deletion;
- bulk-ingestion API with chunked processing and progress metrics;
- durable/versioned index snapshots and deterministic restore;
- indexed metadata filters for high-cardinality corpora;
- dimension/schema validation before committing an ingestion batch;
- benchmark ingestion throughput and update amplification.

## 0.3 — observability and service maturity

- structured tracing with request IDs;
- Prometheus/OpenTelemetry metrics for route distribution, latency, errors, reranker rate, queue/load-shed rate, index size, and ingestion timing;
- readiness vs liveness semantics;
- optional authentication hooks / trusted-gateway integration;
- async Python client with connection pooling;
- standardized retry hints for overload responses;
- configurable HNSW tuning through server configuration.

## 0.4 — persistence and distributed reads

- immutable index-generation manifests;
- object-store/filesystem snapshot loading;
- coordinated rolling index generation across replicas;
- replica consistency metadata in health/stats responses;
- optional shard router abstraction for partitioned corpora;
- benchmark scale-out efficiency rather than assuming linear scaling.

## Research track

- learned AQF policy while preserving the explanation contract;
- calibrated confidence and uncertainty for early exit/deep routing;
- cost-aware reranker selection;
- multi-stage sparse/dense candidate budgeting;
- query-class-aware HNSW effort;
- offline policy learning from relevance, latency, and cost signals;
- reproducible Pareto-frontier comparisons against fixed hybrid/RRF and external engines.

## Non-goals for the current 0.1 line

QueryWeave 0.1 should not be described as a distributed database. It does not currently provide automatic sharding, replicated consensus, durable shared writes, or multi-tenant isolation. The goal of the 0.1 service is a fast, inspectable, backend-pluggable retrieval runtime that can be benchmarked honestly and embedded behind a production application/gateway.
