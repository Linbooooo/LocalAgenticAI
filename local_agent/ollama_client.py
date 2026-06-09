from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class OllamaConnectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelMetrics:
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    output_tokens: int
    prompt_tps: float
    generation_tps: float


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def version(self) -> str:
        return str(self._get("api/version").get("version", "unknown"))

    def tags(self) -> dict[str, Any]:
        return self._get("api/tags")

    def ps(self) -> dict[str, Any]:
        return self._get("api/ps")

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "30m",
    ) -> dict[str, Any]:
        return self._post(
            "api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": options or {},
                "keep_alive": keep_alive,
            },
        )

    def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "30m",
    ) -> tuple[dict[str, Any], ModelMetrics]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options or {},
            "keep_alive": keep_alive,
        }
        request = Request(
            urljoin(self.base_url, "api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        final: dict[str, Any] = {}
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line)
                    content = str(event.get("message", {}).get("content", ""))
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(content)
                    if event.get("done"):
                        final = event
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaConnectionError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaConnectionError(f"Ollama request timed out after {self.timeout}s.") from exc
        except URLError as exc:
            raise OllamaConnectionError(str(exc)) from exc

        finished = time.perf_counter()
        ttft_ms = ((first_token_at or finished) - started) * 1000
        metrics = ModelMetrics(
            ttft_ms=ttft_ms,
            total_ms=_duration_ms(final.get("total_duration")) or (finished - started) * 1000,
            prompt_tokens=int(final.get("prompt_eval_count", 0) or 0),
            output_tokens=int(final.get("eval_count", 0) or 0),
            prompt_tps=_tokens_per_second(final.get("prompt_eval_count"), final.get("prompt_eval_duration")),
            generation_tps=_tokens_per_second(final.get("eval_count"), final.get("eval_duration")),
        )
        return {"message": {"role": "assistant", "content": "".join(chunks)}}, metrics

    def _get(self, path: str) -> dict[str, Any]:
        return self._send(Request(urljoin(self.base_url, path), method="GET"))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            urljoin(self.base_url, path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request)

    def _send(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaConnectionError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaConnectionError(f"Ollama request timed out after {self.timeout}s.") from exc
        except URLError as exc:
            raise OllamaConnectionError(str(exc)) from exc
        return json.loads(body) if body.strip() else {}


def _tokens_per_second(count: Any, duration_ns: Any) -> float:
    count_value = int(count or 0)
    duration_value = int(duration_ns or 0)
    return count_value / (duration_value / 1_000_000_000) if count_value and duration_value else 0.0


def _duration_ms(duration_ns: Any) -> float:
    return int(duration_ns or 0) / 1_000_000
