#!/usr/bin/env python3
"""Small dependency-free retrieval evaluator for normalized HTTP search endpoints.

Input JSONL rows:
  {"id":"q1","query":"...","relevant":["doc-1","doc-2"]}

The endpoint must accept QueryWeave's `/v1/search` payload and return `{"hits":[{"id":...}]}`.
Adapters for other engines can expose the same normalized contract, making the metric code identical.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def search(url: str, query: str, limit: int) -> tuple[list[str], float]:
    payload = json.dumps({"query": query, "limit": limit, "mode": "auto", "explain": False}).encode()
    req = Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(req, timeout=60) as response:  # noqa: S310
        result = json.load(response)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return [hit["id"] for hit in result["hits"]], elapsed_ms


def dcg(ranked: list[str], relevant: set[str], k: int) -> float:
    total = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            total += 1.0 / math.log2(i + 2)
    return total


def evaluate(rows: list[dict], endpoint: str, k: int) -> dict:
    ndcgs, mrrs, recalls, latencies = [], [], [], []
    for row in rows:
        ranked, latency = search(endpoint, row["query"], max(k, 100))
        relevant = set(row["relevant"])
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        ndcgs.append(dcg(ranked, relevant, k) / ideal if ideal else 0.0)
        first = next((i for i, item in enumerate(ranked[:k], start=1) if item in relevant), None)
        mrrs.append(0.0 if first is None else 1.0 / first)
        recalls.append(len(set(ranked[:100]) & relevant) / len(relevant) if relevant else 0.0)
        latencies.append(latency)
    latencies.sort()
    percentile = lambda p: latencies[min(len(latencies) - 1, int((len(latencies) - 1) * p))] if latencies else 0.0
    return {
        "queries": len(rows),
        "ndcg@k": statistics.fmean(ndcgs) if ndcgs else 0.0,
        "mrr@k": statistics.fmean(mrrs) if mrrs else 0.0,
        "recall@100": statistics.fmean(recalls) if recalls else 0.0,
        "latency_ms": {"p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("--endpoint", default="http://localhost:7777/v1/search")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    result = evaluate(rows, args.endpoint, args.k)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
