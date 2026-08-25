FROM rust:1.83-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --release -p queryweave-server

FROM debian:bookworm-slim
RUN useradd --create-home --uid 10001 queryweave
COPY --from=builder /src/target/release/queryweave-server /usr/local/bin/queryweave-server
USER queryweave
EXPOSE 7777
ENV QUERYWEAVE_ADDR=0.0.0.0:7777
ENTRYPOINT ["queryweave-server"]
