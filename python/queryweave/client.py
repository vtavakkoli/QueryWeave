from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


@dataclass(slots=True)
class QueryWeaveClient:
    base_url: str = "http://localhost:7777"
    timeout: float = 30.0

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self.timeout) as response:  # noqa: S310 - caller controls URL
            body = response.read()
            return None if not body else json.loads(body)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/v1/stats")

    def upsert(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/v1/documents:upsert", {"documents": documents})

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        mode: str = "auto",
        dense: list[float] | None = None,
        sparse: dict[str, list] | None = None,
        filters: dict[str, str] | None = None,
        explain: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/search",
            {
                "query": query,
                "limit": limit,
                "mode": mode,
                "dense": dense,
                "sparse": sparse,
                "filter": filters or {},
                "explain": explain,
            },
        )

    def reset(self) -> None:
        self._request("DELETE", "/v1/index")
