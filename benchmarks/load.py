#!/usr/bin/env python3
"""Dependency-free concurrent load probe for a running QueryWeave server.

This is intentionally a small engineering probe, not a substitute for a full benchmark harness.
It reports throughput, latency percentiles, HTTP status counts, and overload (429) behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class Sample:
    status: int
    latency_ms: float


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, bytes]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - benchmark URL is explicit
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def seed(base_url: str, count: int, timeout: float) -> None:
    if count <= 0:
        return
    documents = [
        {
            "id": f"load-{index}",
            "text": (
                f"industrial hydraulic pump maintenance record {index} "
                f"temperature vibration bearing pressure failure prevention"
            ),
            "source": "load-benchmark",
            "metadata": {"partition": str(index % 8)},
        }
        for index in range(count)
    ]
    status, body = post_json(
        f"{base_url}/v1/documents:upsert",
        {"documents": documents},
        timeout,
    )
    if status >= 300:
        raise RuntimeError(f"seed failed with HTTP {status}: {body.decode(errors='replace')}")


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def run_one(base_url: str, payload: dict[str, Any], timeout: float) -> Sample:
    started = time.perf_counter()
    status, _ = post_json(f"{base_url}/v1/search", payload, timeout)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return Sample(status=status, latency_ms=latency_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7777")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed-docs", type=int, default=0)
    parser.add_argument("--query", default="prevent hydraulic pump temperature failure")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=("auto", "lexical", "hybrid", "deep"), default="auto")
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    base_url = args.url.rstrip("/")
    seed(base_url, args.seed_docs, args.timeout)
    payload = {
        "query": args.query,
        "limit": args.limit,
        "mode": args.mode,
        "explain": False,
    }

    for _ in range(args.warmup):
        run_one(base_url, payload, args.timeout)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        samples = list(
            executor.map(
                lambda _: run_one(base_url, payload, args.timeout),
                range(args.requests),
            )
        )
    elapsed = time.perf_counter() - started

    latencies = sorted(sample.latency_ms for sample in samples)
    statuses = Counter(sample.status for sample in samples)
    successful = sum(count for status, count in statuses.items() if 200 <= status < 300)
    report = {
        "url": base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_s": round(elapsed, 4),
        "qps": round(args.requests / elapsed, 2) if elapsed else 0.0,
        "successful": successful,
        "success_rate": round(successful / args.requests, 4),
        "status_counts": {str(status): count for status, count in sorted(statuses.items())},
        "server_busy_429": statuses.get(429, 0),
        "latency_ms": {
            "min": round(latencies[0], 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "max": round(latencies[-1], 3),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
