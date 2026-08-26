FROM rust:1.98-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --release -p queryweave-server

FROM debian:bookworm-slim
LABEL org.opencontainers.image.title="QueryWeave" \
      org.opencontainers.image.description="Query-adaptive hybrid retrieval engine" \
      org.opencontainers.image.source="https://github.com/vtavakkoli/QueryWeave" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 10001 queryweave
COPY --from=builder /src/target/release/queryweave-server /usr/local/bin/queryweave-server
USER queryweave
EXPOSE 7777
ENV QUERYWEAVE_ADDR=0.0.0.0:7777 \
    QUERYWEAVE_LEXICAL_BACKEND=tantivy \
    QUERYWEAVE_VECTOR_BACKEND=hnsw-f32 \
    QUERYWEAVE_MAX_BODY_BYTES=33554432 \
    QUERYWEAVE_MAX_BATCH_DOCUMENTS=10000 \
    QUERYWEAVE_MAX_QUERY_CHARS=16384
STOPSIGNAL SIGTERM
ENTRYPOINT ["queryweave-server"]
