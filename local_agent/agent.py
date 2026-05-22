from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .context import prepare_messages
from .ollama_client import OllamaClient, OllamaConnectionError
from .prompts import EDIT_PROMPT, SYSTEM_PROMPT
from .tool_policy import classify_intent, extract_direct_shell_command
from .tools import WorkspaceTools


@dataclass
class AgentResult:
    content: str
    turns: int


class LocalAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OllamaClient(config.ollama_url, timeout=config.ollama_timeout)
        self.tools = WorkspaceTools(config)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(workspace=str(config.workspace))}
        ]
        self.last_written_file: str | None = None

    def run(self, task: str) -> AgentResult:
        self.messages.append({"role": "user", "content": task})
        intent = classify_intent(task)

        command = extract_direct_shell_command(task) or self._followup_run_command(task)
        if command:
            return self._run_shell(command)

        if intent == "hardware":
            context = _format_json("Local hardware profile", self.tools.hardware_profile())
            return self._ask(context)
        if intent == "read":
            return self._ask(self._workspace_context(task))
        if intent == "edit":
            return self._edit(task)
        if intent == "shell":
            return self._ask("The user wants to run something, but no exact command was provided. Ask for the exact command if it is ambiguous.")
        return self._ask()

    def _ask(self, context: str | None = None) -> AgentResult:
        response = self._chat(context)
        message = response.get("message", {})
        content = str(message.get("content", "")).strip()
        self.messages.append({"role": "assistant", "content": content})
        return AgentResult(content=content, turns=1)

    def _edit(self, task: str) -> AgentResult:
        context = self._workspace_context(task)
        response = self._chat(f"{context}\n\n{EDIT_PROMPT}")
        content = str(response.get("message", {}).get("content", "")).strip()
        action = _extract_json_object(content)
        if not isinstance(action, dict):
            self.messages.append({"role": "assistant", "content": content})
            return AgentResult(content=content, turns=1)
        if action.get("action") == "answer":
            reply = str(action.get("message", "")).strip()
            self.messages.append({"role": "assistant", "content": reply})
            return AgentResult(content=reply, turns=1)

        result = self._apply_edit_action(action)
        if result.get("ok") and result.get("path"):
            self.last_written_file = str(result["path"])
        reply = action.get("message") or _format_action_result(action, result)
        self.messages.append({"role": "assistant", "content": reply})
        return AgentResult(content=reply, turns=1)

    def _apply_edit_action(self, action: dict[str, Any]) -> dict[str, Any]:
        kind = action.get("action")
        if kind == "write_file":
            path = str(action.get("path", "")).strip()
            if not path:
                return {"ok": False, "error": "Missing file path."}
            return self.tools.write_file(path, str(action.get("content", "")))
        if kind == "replace_in_file":
            old = str(action.get("old", ""))
            if old == "":
                return {"ok": False, "error": "Replacement target is empty."}
            return self.tools.replace_in_file(
                str(action.get("path", "")),
                old,
                str(action.get("new", "")),
                int(action.get("max_replacements", 1)),
            )
        return {"ok": False, "error": f"Unknown edit action: {kind}"}

    def _run_shell(self, command: str) -> AgentResult:
        result = self.tools.run_shell(command)
        self.messages.append({"role": "assistant", "content": _format_shell_result(result)})
        return AgentResult(content=_format_shell_result(result), turns=1)

    def _followup_run_command(self, task: str) -> str | None:
        text = task.lower()
        if "run" not in text or not self.last_written_file:
            return None
        if "file" not in text and "script" not in text and "it" not in text:
            return None
        path = shlex.quote(self.last_written_file)
        return f"python3 {path}" if self.last_written_file.endswith(".py") else path

    def _workspace_context(self, task: str) -> str:
        listing = self.tools.list_files(max_depth=3, limit=120)
        files = listing.get("files", []) if listing.get("ok") else []
        selected = _select_context_files(task, files)
        parts = [_format_json("Workspace files", listing)]
        for path in selected:
            result = self.tools.read_file(path, max_lines=160)
            if result.get("ok"):
                parts.append(f"File: {path}\n{result['content']}")
        return "\n\n".join(parts)

    def _chat(self, context: str | None = None) -> dict[str, Any]:
        messages = self._messages_with_context(context)
        while True:
            try:
                return self.client.chat(
                    model=self.config.model,
                    messages=prepare_messages(messages, self.config.context_budget_tokens()),
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

    def _messages_with_context(self, context: str | None) -> list[dict[str, Any]]:
        if not context:
            return self.messages
        return [self.messages[0], {"role": "system", "content": context}, *self.messages[1:]]


def _select_context_files(task: str, files: list[str]) -> list[str]:
    lower_task = task.lower()
    defaults = {"readme.md", "pyproject.toml", "package.json", "cargo.toml"}
    selected: list[str] = []
    for path in files:
        name = Path(path).name.lower()
        if name in defaults or name in lower_task or path.lower() in lower_task:
            selected.append(path)
        if len(selected) >= 6:
            break
    return selected


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip(), *_fenced_blocks(text)]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                continue
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            return value
    return None


def _fenced_blocks(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)]


def _format_json(title: str, value: Any) -> str:
    return f"{title}:\n{json.dumps(value, indent=2, ensure_ascii=False)}"


def _format_action_result(action: dict[str, Any], result: dict[str, Any]) -> str:
    if result.get("ok"):
        return f"{action.get('action')} completed: {result.get('path', '')}".strip()
    return str(result.get("error", "Edit action failed."))


def _format_shell_result(result: dict[str, Any]) -> str:
    if not result.get("ok") and result.get("error"):
        return str(result["error"])
    stdout = str(result.get("stdout", "")).strip()
    stderr = str(result.get("stderr", "")).strip()
    returncode = result.get("returncode")
    if stdout and stderr:
        return f"{stdout}\n\nstderr:\n{stderr}"
    if stdout:
        return stdout
    if stderr:
        return f"Command exited with {returncode}.\n{stderr}"
    return f"Command exited with {returncode}."
