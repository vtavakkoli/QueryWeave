from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class QueryWeaveError(RuntimeError):
    """Base exception raised by the QueryWeave HTTP SDK."""


class QueryWeaveConnectionError(QueryWeaveError):
    """Raised when the QueryWeave service cannot be reached."""


class QueryWeaveHTTPError(QueryWeaveError):
    """Raised for a non-success HTTP response from QueryWeave."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"QueryWeave HTTP {status_code} [{code}]: {message}")


@dataclass(slots=True)
class QueryWeaveClient:
    """Small dependency-free client for the QueryWeave HTTP service."""

    base_url: str = "http://localhost:7777"
    timeout: float = 30.0

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "queryweave-python/0.1.0",
            },
        )
        try:
            with urlopen(req, timeout=self.timeout) as response:  # noqa: S310 - caller controls URL
                body = response.read()
        except HTTPError as error:
            body = error.read()
            code = "http_error"
            message = error.reason or "QueryWeave request failed"
            if body:
                try:
                    parsed = json.loads(body)
                    detail = parsed.get("error", {})
                    code = str(detail.get("code", code))
                    message = str(detail.get("message", message))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    message = body.decode("utf-8", errors="replace")
            raise QueryWeaveHTTPError(error.code, code, message) from error
        except URLError as error:
            raise QueryWeaveConnectionError(
                f"Could not connect to QueryWeave at {self.base_url!r}: {error.reason}"
            ) from error

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
