from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class OllamaConnectionError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def version(self) -> str:
        data = self._get("api/version")
        return str(data.get("version", "unknown"))

    def tags(self) -> dict[str, Any]:
        return self._get("api/tags")

    def ps(self) -> dict[str, Any]:
        return self._get("api/ps")

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
        keep_alive: str | int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": keep_alive,
        }
        if tools:
            payload["tools"] = tools
        return self._post("api/chat", payload)

    def _get(self, path: str) -> dict[str, Any]:
        request = Request(urljoin(self.base_url, path), method="GET")
        return self._send(request, self.timeout)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            urljoin(self.base_url, path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request, self.timeout)

    @staticmethod
    def _send(request: Request, timeout: int) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaConnectionError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaConnectionError(
                f"Ollama request timed out after {timeout}s. The model may be CPU-bound or overloaded."
            ) from exc
        except URLError as exc:
            raise OllamaConnectionError(str(exc)) from exc
        if not body.strip():
            return {}
        return json.loads(body)
