from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import AgentConfig
from .context import prepare_messages
from .ollama_client import OllamaClient, OllamaConnectionError
from .prompts import SYSTEM_PROMPT
from .tools import ToolRegistry


@dataclass
class AgentResult:
    content: str
    turns: int


class LocalAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OllamaClient(config.ollama_url, timeout=config.ollama_timeout)
        self.tools = ToolRegistry(config)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(workspace=str(config.workspace))}
        ]

    def run(self, task: str) -> AgentResult:
        self.messages.append({"role": "user", "content": task})
        final_content = ""

        for turn in range(1, self.config.max_turns + 1):
            response = self._chat_with_adaptive_context()
            message = response.get("message", {})
            if not message:
                raise RuntimeError(f"Ollama response did not contain a message: {response}")

            self.messages.append(message)
            tool_calls = message.get("tool_calls") or []
            content_tool_calls = False
            if not tool_calls:
                tool_calls = self._tool_calls_from_content(str(message.get("content", "")))
                content_tool_calls = bool(tool_calls)
            if not tool_calls:
                final_content = str(message.get("content", "")).strip()
                break

            for call in tool_calls:
                tool_name, arguments = self._parse_tool_call(call)
                result = self.tools.run(tool_name, arguments)
                result_json = json.dumps(result, ensure_ascii=False)
                if content_tool_calls:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"Tool result for {tool_name}:\n{result_json}\nUse this result and continue.",
                        }
                    )
                else:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": result_json,
                        }
                    )
        else:
            final_content = "Stopped after reaching the configured turn limit."

        return AgentResult(content=final_content, turns=min(len(self.messages), self.config.max_turns))

    def _chat_with_adaptive_context(self) -> dict[str, Any]:
        while True:
            try:
                return self.client.chat(
                    model=self.config.model,
                    messages=prepare_messages(self.messages, self.config.context_budget_tokens()),
                    tools=self.tools.schemas(),
                    options=self.config.ollama_options(),
                    keep_alive=self.config.keep_alive,
                )
            except OllamaConnectionError as exc:
                message = str(exc).lower()
                if "memory" not in message and "out of" not in message:
                    raise
                if self.config.num_ctx <= self.config.min_num_ctx:
                    raise
                self.config.num_ctx = max(self.config.min_num_ctx, self.config.num_ctx // 2)
                self.messages.append(
                    {
                        "role": "system",
                        "content": (
                            "The local inference server rejected the previous request because of memory pressure. "
                            f"Retrying with num_ctx={self.config.num_ctx}."
                        ),
                    }
                )

    @staticmethod
    def _parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        function = call.get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Tool call did not include a function name: {call}")

        raw_args = function.get("arguments") or {}
        if isinstance(raw_args, str):
            arguments = json.loads(raw_args) if raw_args.strip() else {}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            raise ValueError(f"Tool call arguments must be an object or JSON string: {call}")
        return name, arguments

    @staticmethod
    def _tool_calls_from_content(content: str) -> list[dict[str, Any]]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            try:
                payload, _ = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                return []

        calls: list[dict[str, Any]] = []
        if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
            raw_calls = payload["tool_calls"]
        else:
            raw_calls = [payload]

        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            if "function" in raw_call:
                calls.append(raw_call)
                continue
            name = raw_call.get("name") or raw_call.get("tool_name")
            arguments = raw_call.get("arguments") or raw_call.get("args") or {}
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append({"function": {"name": name, "arguments": arguments}})
        return calls
