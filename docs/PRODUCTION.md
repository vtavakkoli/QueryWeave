# Production deployment and scaling

QueryWeave is designed as a Rust-first adaptive retrieval engine with replaceable lexical, sparse, dense, and reranking components. The HTTP service is intentionally thin, but production deployments should still treat retrieval and indexing as CPU- and memory-intensive work.

## Runtime model

The Axum service separates asynchronous network handling from synchronous retrieval/indexing work. Search, upsert, and reset operations execute on Tokio's blocking pool rather than occupying async worker threads.

Heavy work is additionally bounded by a semaphore. When all work slots are in use, the service rejects new heavy requests with HTTP `429` and a stable error code:

```json
{
  "error": {
    "code": "server_busy",
    "message": "QueryWeave is at its configured work limit; retry with backoff"
  }
}
```

Clients should use bounded exponential backoff with jitter for this condition.

## Capacity configuration

| Variable | Default | Purpose |
|---|---:|---|
| `QUERYWEAVE_ADDR` | `0.0.0.0:7777` | listen address |
| `QUERYWEAVE_LEXICAL_BACKEND` | `tantivy` | lexical backend: `tantivy`, `builtin`, `bm25` |
| `QUERYWEAVE_VECTOR_BACKEND` | `hnsw-f32` | vector backend: `hnsw-f32`, `f32`, `hnsw-i8`, `i8`, `exact` |
| `QUERYWEAVE_MAX_CONCURRENT_WORK` | `max(4, 2 × CPU parallelism)` | maximum concurrent heavy engine operations |
| `QUERYWEAVE_MAX_BODY_BYTES` | `33554432` | maximum HTTP request body size |
| `QUERYWEAVE_MAX_BATCH_DOCUMENTS` | `10000` | maximum documents accepted by one upsert request |
| `QUERYWEAVE_MAX_QUERY_CHARS` | `16384` | maximum query length |

Invalid backend names and invalid positive-integer limits fail fast at startup instead of silently selecting a different runtime configuration.

## Sizing guidance

Start with the default work limit and benchmark with your own corpus and embeddings. Increase it only when CPU utilization, tail latency, and memory usage remain healthy. A higher concurrency value is not automatically higher throughput: HNSW search, Tantivy search, embedding, sparse encoding, and reranking can all compete for the same CPU and memory bandwidth.

For latency-sensitive services:

- reserve CPU rather than heavily overcommitting containers;
- monitor p50, p95, and p99 latency separately;
- measure reranker invocation rate because deep routing changes cost per query;
- use `hnsw-f32` as the normal ANN baseline and compare `hnsw-i8` when memory pressure matters;
- keep ingestion batches reasonably large so index rebuild/setup work is amortized;
- keep `QUERYWEAVE_MAX_CONCURRENT_WORK` below the point where p99 latency collapses.

## Horizontal scaling

The current server keeps its corpus and indexes in process memory. Multiple replicas therefore do **not** automatically share writes. For read-heavy production use, the recommended pattern is:

1. build or ingest the same corpus into each replica;
2. place replicas behind a load balancer;
3. send health probes to `/health`;
4. use immutable/versioned corpus snapshots or coordinated ingestion so all replicas serve the same index generation.

QueryWeave does not currently provide distributed consensus, automatic sharding, durable index replication, or shared write coordination. Those concerns should be handled by the deployment platform or by a future persistence/sharding layer. This boundary is intentional and should be considered when comparing QueryWeave with distributed search databases.

## Backpressure and retry behavior

`429 server_busy` is deliberate load shedding. It is preferable to allowing an unbounded queue that increases memory usage and tail latency. A production caller should:

- retry only idempotent requests automatically;
- use exponential backoff with jitter;
- cap total retry time;
- surface persistent overload as a capacity signal;
- avoid retry storms across many replicas.

Search requests are naturally idempotent. Upserts should use stable document IDs so retrying a batch replaces the same documents rather than creating duplicates.

## Health and readiness

`GET /health` returns the active backends plus configured and currently available heavy-work capacity:

```json
{
  "status": "ok",
  "engine": "QueryWeave",
  "version": "0.1.0",
  "lexical_backend": "tantivy-bm25",
  "vector_backend": "usearch-hnsw-f32",
  "max_concurrent_work": 16,
  "available_work_slots": 16
}
```

The health path itself does not consume a heavy-work semaphore slot. Its stats lookup runs off the async worker threads so a concurrent index operation cannot block the network runtime.

## Container deployment

The provided image is multi-stage and runs the service as an unprivileged user. A typical deployment should also set explicit CPU/memory limits and use a read-only root filesystem when the selected backend does not need filesystem persistence.

Example:

```yaml
services:
  queryweave:
    build: .
    environment:
      QUERYWEAVE_LEXICAL_BACKEND: tantivy
      QUERYWEAVE_VECTOR_BACKEND: hnsw-f32
      QUERYWEAVE_MAX_CONCURRENT_WORK: 16
      QUERYWEAVE_MAX_BODY_BYTES: 33554432
      QUERYWEAVE_MAX_BATCH_DOCUMENTS: 10000
    ports:
      - "7777:7777"
```

## Benchmark before claiming scale

Scalability claims should be based on reproducible measurements. At minimum record:

- corpus size and vector dimensions;
- backend and HNSW tuning parameters;
- hardware and container limits;
- concurrent clients;
- p50/p95/p99 latency and QPS;
- CPU and resident memory;
- index build/upsert time;
- quality metrics such as nDCG@10, MRR@10, and Recall@100;
- adaptive route distribution and reranker invocation rate.

The goal is a quality-versus-cost Pareto frontier, not a universal "fastest engine" claim.
