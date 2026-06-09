from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


LOCAL_OLLAMA_HOSTS = {"127.0.0.1", "localhost", "::1", "ollama", "host.docker.internal"}


@dataclass
class AgentConfig:
    model: str = "qwen3.5:9b-q4_K_M"
    ollama_url: str = "http://127.0.0.1:11434"
    workspace: Path = Path(".")
    num_ctx: int = 8192
    num_predict: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    num_thread: int = 12
    keep_alive: str | int = "30m"
    ollama_timeout: int = 300
    trust: str = "ask"
    allow_network_tools: bool = False
    max_steps: int = 20
    shell_timeout: int = 120

    @classmethod
    def load(cls, path: Path | None) -> "AgentConfig":
        config = cls()
        if path is not None:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Config file must contain a JSON object.")
            for key, value in data.items():
                if not hasattr(config, key):
                    raise ValueError(f"Unknown config key: {key}")
                if key == "workspace":
                    value = Path(value)
                setattr(config, key, value)
        config.apply_env()
        return config

    def apply_env(self) -> None:
        env_map = {
            "LOCAL_AGENT_MODEL": ("model", str),
            "OLLAMA_URL": ("ollama_url", str),
            "LOCAL_AGENT_WORKSPACE": ("workspace", Path),
            "LOCAL_AGENT_NUM_CTX": ("num_ctx", int),
            "LOCAL_AGENT_NUM_PREDICT": ("num_predict", int),
            "LOCAL_AGENT_TEMPERATURE": ("temperature", float),
            "LOCAL_AGENT_TOP_P": ("top_p", float),
            "LOCAL_AGENT_REPEAT_PENALTY": ("repeat_penalty", float),
            "LOCAL_AGENT_NUM_THREAD": ("num_thread", int),
            "LOCAL_AGENT_KEEP_ALIVE": ("keep_alive", str),
            "LOCAL_AGENT_OLLAMA_TIMEOUT": ("ollama_timeout", int),
            "LOCAL_AGENT_TRUST": ("trust", str),
            "LOCAL_AGENT_MAX_STEPS": ("max_steps", int),
            "LOCAL_AGENT_SHELL_TIMEOUT": ("shell_timeout", int),
        }
        for env_name, (field_name, converter) in env_map.items():
            value = os.environ.get(env_name)
            if value is not None and value != "":
                setattr(self, field_name, converter(value))

        allow_network = os.environ.get("LOCAL_AGENT_ALLOW_NETWORK_TOOLS")
        if allow_network is not None:
            self.allow_network_tools = allow_network.strip().lower() in {"1", "true", "yes", "on"}

    def finalize(self) -> None:
        parsed = urlparse(self.ollama_url)
        if parsed.scheme != "http":
            raise ValueError("Only plain local HTTP Ollama endpoints are allowed.")
        if parsed.hostname not in LOCAL_OLLAMA_HOSTS:
            raise ValueError(
                "The Ollama endpoint must be local: loopback, host.docker.internal, or the Compose service 'ollama'."
            )
        if self.trust not in {"ask", "auto"}:
            raise ValueError("trust must be either 'ask' or 'auto'.")
        if self.num_ctx < 1024:
            raise ValueError("num_ctx must be at least 1024.")
        if self.num_predict < 1 or self.num_predict >= self.num_ctx:
            raise ValueError("num_predict must be positive and smaller than num_ctx.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.shell_timeout < 1:
            raise ValueError("shell_timeout must be at least 1.")
        self.workspace = self.workspace.expanduser().resolve()
        if not self.workspace.exists():
            raise ValueError(f"Workspace does not exist: {self.workspace}")

    def ollama_options(self) -> dict[str, Any]:
        return {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "num_thread": self.num_thread,
        }

    def context_budget_tokens(self) -> int:
        return max(512, self.num_ctx - self.num_predict)
