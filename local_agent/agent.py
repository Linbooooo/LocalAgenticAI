from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .context import prepare_messages
from .ollama_client import OllamaClient, OllamaConnectionError
from .prompts import ACTION_PROMPT, SYSTEM_PROMPT
from .tool_policy import classify_intent, extract_direct_shell_command
from .tools import WorkspaceTools


@dataclass
class AgentResult:
    content: str
    turns: int


class LocalAgent:
    ACTION_REPAIR_ATTEMPTS = 2

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
            return self._act(task, intent)
        if intent == "edit":
            return self._act(task, intent)
        if intent == "shell":
            return self._act(task, intent)
        return self._ask()

    def _ask(self, context: str | None = None) -> AgentResult:
        response = self._chat(context)
        message = response.get("message", {})
        content = str(message.get("content", "")).strip()
        self.messages.append({"role": "assistant", "content": content})
        return AgentResult(content=content, turns=1)

    def _act(self, task: str, intent: str) -> AgentResult:
        observations: list[dict[str, Any]] = []

        for step in range(1, self.config.max_steps + 1):
            action = self._forced_action(task, intent, observations)
            if action is None:
                action = self._next_action(task, intent, observations)

            kind = str(action.get("action", "")).strip()
            if kind in {"answer", "finish"}:
                forced = self._forced_action(task, intent, observations)
                if forced is not None:
                    action = forced
                    kind = str(action.get("action", "")).strip()
                else:
                    reply = str(action.get("message", "")).strip() or _format_observations(observations)
                    reply = _format_final_reply(reply, observations)
                    self.messages.append({"role": "assistant", "content": reply})
                    return AgentResult(content=reply, turns=step)

            if _is_repeated_write(action, observations):
                forced = self._forced_action(task, intent, observations)
                if forced is not None:
                    action = forced
                else:
                    result = {"ok": False, "error": "Repeated identical write action."}
                    observations.append(
                        {
                            "step": step,
                            "action": _public_action(action),
                            "signature": _action_signature(action),
                            "result": result,
                        }
                    )
                    continue

            kind = str(action.get("action", "")).strip()
            if kind in {"answer", "finish"}:
                reply = str(action.get("message", "")).strip() or _format_observations(observations)
                reply = _format_final_reply(reply, observations)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)

            try:
                result = self._apply_action(action, intent)
            except (TypeError, ValueError, OSError) as exc:
                result = {"ok": False, "error": str(exc)}
            observations.append(
                {
                    "step": step,
                    "action": _public_action(action),
                    "signature": _action_signature(action),
                    "result": result,
                }
            )
            if result.get("ok") and result.get("path"):
                self.last_written_file = str(result["path"])
            if _should_stop_after_success(task, intent, observations):
                reply = _format_final_reply("Completed and verified successfully.", observations)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)
            if not result.get("ok") and _should_stop_after_failure(result):
                reply = _format_observations(observations)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)

        reply = f"Stopped after {self.config.max_steps} action steps.\n\n{_format_observations(observations)}"
        self.messages.append({"role": "assistant", "content": reply})
        return AgentResult(content=reply, turns=self.config.max_steps)

    def _next_action(self, task: str, intent: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
        protocol_error: str | None = None
        last_content = ""

        for _ in range(self.ACTION_REPAIR_ATTEMPTS + 1):
            response = self._chat(self._action_context(task, intent, observations, protocol_error), json_mode=True)
            last_content = str(response.get("message", {}).get("content", "")).strip()
            action = _extract_json_object(last_content)
            if not isinstance(action, dict):
                salvaged = _salvage_action_from_text(task, intent, last_content)
                if salvaged is not None:
                    return salvaged
                protocol_error = _protocol_error("No valid JSON action object was found.", last_content)
                continue

            validation_error = _validate_action(action, intent, observations)
            if validation_error is None:
                return action
            protocol_error = _protocol_error(validation_error, last_content)

        salvaged = _salvage_action_from_text(task, intent, last_content)
        if salvaged is not None and _validate_action(salvaged, intent, observations) is None:
            return salvaged
        return {
            "action": "answer",
            "message": "I could not get a valid local action from the model after retrying the action protocol.",
        }

    def _forced_action(self, task: str, intent: str, observations: list[dict[str, Any]]) -> dict[str, Any] | None:
        if intent != "edit" or not _requests_run(task):
            return None
        if not self.last_written_file or not self.last_written_file.endswith(".py"):
            return None
        if _last_successful_change_step(observations) <= _last_action_step(observations, "run_shell"):
            return None
        return {
            "action": "run_shell",
            "command": f"python3 {shlex.quote(self.last_written_file)}",
            "timeout_seconds": 120,
        }

    def _apply_action(self, action: dict[str, Any], intent: str) -> dict[str, Any]:
        kind = action.get("action")
        if kind == "list_files":
            return self.tools.list_files(
                str(action.get("path", ".")),
                int(action.get("max_depth", 4)),
                int(action.get("limit", 200)),
            )
        if kind == "read_file":
            return self.tools.read_file(
                str(action.get("path", "")),
                int(action.get("start_line", 1)),
                int(action.get("max_lines", 200)),
            )
        if kind == "search_text":
            return self.tools.search_text(
                str(action.get("pattern", "")),
                str(action.get("path", ".")),
                str(action.get("file_glob", "*")),
                _bool_value(action.get("case_sensitive", False)),
            )
        if kind == "write_file":
            if intent != "edit":
                return {"ok": False, "error": "File edits are only allowed for edit requests."}
            path = str(action.get("path", "")).strip()
            if not path:
                return {"ok": False, "error": "Missing file path."}
            return self.tools.write_file(path, str(action.get("content", "")))
        if kind == "replace_in_file":
            if intent != "edit":
                return {"ok": False, "error": "File edits are only allowed for edit requests."}
            old = str(action.get("old", ""))
            if old == "":
                return {"ok": False, "error": "Replacement target is empty."}
            return self.tools.replace_in_file(
                str(action.get("path", "")),
                old,
                str(action.get("new", "")),
                int(action.get("max_replacements", 1)),
            )
        if kind == "run_shell":
            if intent not in {"edit", "shell"}:
                return {"ok": False, "error": "Shell commands are only allowed for shell or edit requests."}
            command = str(action.get("command", "")).strip()
            if not command:
                return {"ok": False, "error": "Missing shell command."}
            return self.tools.run_shell(command, int(action.get("timeout_seconds", 120)))
        return {"ok": False, "error": f"Unknown action: {kind}"}

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

    def _action_context(
        self,
        task: str,
        intent: str,
        observations: list[dict[str, Any]],
        protocol_error: str | None = None,
    ) -> str:
        parts = [
            self._workspace_context(task),
            f"Request intent: {intent}",
            ACTION_PROMPT,
        ]
        if intent == "read":
            parts.append("For this read request, do not edit files or run shell commands.")
        elif intent == "shell":
            parts.append("For this shell request, do not edit files unless the user explicitly asks for edits.")
        if protocol_error:
            parts.append(protocol_error)
        if observations:
            parts.append(_format_json("Previous action results", observations))
        return "\n\n".join(parts)

    def _chat(self, context: str | None = None, *, json_mode: bool = False) -> dict[str, Any]:
        messages = self._messages_with_context(context)
        while True:
            try:
                return self.client.chat(
                    model=self.config.model,
                    messages=prepare_messages(messages, self.config.context_budget_tokens()),
                    options=self.config.ollama_options(),
                    keep_alive=self.config.keep_alive,
                    response_format="json" if json_mode else None,
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


def _validate_action(action: dict[str, Any], intent: str, observations: list[dict[str, Any]]) -> str | None:
    kind = str(action.get("action", "")).strip()
    allowed = {
        "answer",
        "finish",
        "list_files",
        "read_file",
        "replace_in_file",
        "run_shell",
        "search_text",
        "write_file",
    }
    if kind not in allowed:
        return f"Unknown action '{kind}'."
    if kind == "write_file":
        if intent != "edit":
            return "write_file is only valid for edit requests."
        if not str(action.get("path", "")).strip():
            return "write_file requires a non-empty path."
        if "content" not in action:
            return "write_file requires content."
    elif kind == "replace_in_file":
        if intent != "edit":
            return "replace_in_file is only valid for edit requests."
        if not str(action.get("path", "")).strip():
            return "replace_in_file requires a non-empty path."
        if not str(action.get("old", "")):
            return "replace_in_file requires non-empty old text."
        if "new" not in action:
            return "replace_in_file requires new text."
    elif kind == "run_shell":
        if intent not in {"edit", "shell"}:
            return "run_shell is only valid for edit or shell requests."
        if not str(action.get("command", "")).strip():
            return "run_shell requires a non-empty command."
    elif kind == "read_file" and not str(action.get("path", "")).strip():
        return "read_file requires a non-empty path."
    elif kind == "search_text" and not str(action.get("pattern", "")).strip():
        return "search_text requires a non-empty pattern."
    elif kind in {"answer", "finish"}:
        if kind == "finish" and intent == "edit" and not observations:
            return "finish is only valid after completing the requested work."
        if not str(action.get("message", "")).strip():
            return f"{kind} requires a non-empty message."
    return None


def _salvage_action_from_text(task: str, intent: str, text: str) -> dict[str, Any] | None:
    if intent != "edit":
        return None
    path = _filename_from_task(task)
    if not path:
        return None
    code = _python_code_block(text)
    if code is None:
        return None
    return {"action": "write_file", "path": path, "content": code.rstrip() + "\n"}


def _filename_from_task(task: str) -> str | None:
    match = re.search(r"\b[\w./-]+\.py\b", task)
    if not match:
        return None
    path = match.group(0).strip("./")
    if path.startswith("/") or ".." in Path(path).parts:
        return None
    return path


def _python_code_block(text: str) -> str | None:
    for language, content in _fenced_blocks(text, with_language=True):
        if language in {"python", "py"}:
            return content
    for language, content in _fenced_blocks(text, with_language=True):
        if language == "" and _looks_like_python(content):
            return content
    return None


def _looks_like_python(text: str) -> bool:
    return bool(re.search(r"^\s*(def|class|import|from|assert|print)\b", text, re.MULTILINE))


def _protocol_error(reason: str, content: str) -> str:
    return (
        "Protocol correction:\n"
        f"- The previous response was invalid: {reason}\n"
        "- Return exactly one valid JSON action object. No markdown. No prose.\n"
        f"- Previous response excerpt: {_truncate(_one_line(content), 700)}"
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _fenced_blocks(text: str, *, with_language: bool = False):
    blocks = []
    for match in re.finditer(r"```([a-zA-Z0-9_-]*)\s*(.*?)```", text, re.DOTALL):
        language = match.group(1).strip().lower()
        content = match.group(2).strip()
        blocks.append((language, content) if with_language else content)
    return blocks


def _format_json(title: str, value: Any) -> str:
    return f"{title}:\n{json.dumps(value, indent=2, ensure_ascii=False)}"


def _public_action(action: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in action.items() if key != "content"}
    if "content" in action:
        public["content_bytes"] = len(str(action["content"]).encode("utf-8"))
    return public


def _action_signature(action: dict[str, Any]) -> str:
    kind = str(action.get("action", ""))
    if kind == "write_file":
        content_hash = hashlib.sha256(str(action.get("content", "")).encode("utf-8")).hexdigest()
        return f"write_file:{action.get('path', '')}:{content_hash}"
    if kind == "replace_in_file":
        old_hash = hashlib.sha256(str(action.get("old", "")).encode("utf-8")).hexdigest()
        new_hash = hashlib.sha256(str(action.get("new", "")).encode("utf-8")).hexdigest()
        return f"replace_in_file:{action.get('path', '')}:{old_hash}:{new_hash}"
    return json.dumps(_public_action(action), sort_keys=True, ensure_ascii=False)


def _is_repeated_write(action: dict[str, Any], observations: list[dict[str, Any]]) -> bool:
    if action.get("action") not in {"write_file", "replace_in_file"}:
        return False
    signature = _action_signature(action)
    return any(observation.get("signature") == signature for observation in observations)


def _requests_run(task: str) -> bool:
    return any(re.search(rf"\b{word}\b", task, re.IGNORECASE) for word in {"check", "run", "test", "verify"})


def _last_action_step(observations: list[dict[str, Any]], action_name: str) -> int:
    return max(
        (
            int(observation.get("step", 0))
            for observation in observations
            if observation.get("action", {}).get("action") == action_name
        ),
        default=0,
    )


def _last_successful_change_step(observations: list[dict[str, Any]]) -> int:
    return max(
        (
            int(observation.get("step", 0))
            for observation in observations
            if observation.get("action", {}).get("action") in {"write_file", "replace_in_file"}
            and observation.get("result", {}).get("ok")
        ),
        default=0,
    )


def _should_stop_after_failure(result: dict[str, Any]) -> bool:
    error = str(result.get("error", "")).lower()
    return "declined" in error or "blocked" in error or "escapes workspace" in error


def _should_stop_after_success(task: str, intent: str, observations: list[dict[str, Any]]) -> bool:
    if intent != "edit" or not _requests_run(task) or not observations:
        return False
    latest = observations[-1]
    if latest.get("action", {}).get("action") != "run_shell":
        return False
    if not latest.get("result", {}).get("ok"):
        return False
    latest_step = int(latest.get("step", 0))
    return 0 < _last_successful_change_step(observations) < latest_step


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _format_final_reply(reply: str, observations: list[dict[str, Any]]) -> str:
    shell_results = [
        observation
        for observation in observations
        if observation.get("action", {}).get("action") == "run_shell"
    ]
    if not shell_results:
        return reply

    lines = [reply, "", "Command results:"]
    for observation in shell_results:
        action = observation["action"]
        result = observation["result"]
        lines.append(f"$ {action.get('command', '')}")
        lines.append(_format_shell_result(result))
    return "\n".join(line for line in lines if line is not None).strip()


def _format_observations(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return "No actions were completed."

    lines: list[str] = []
    for observation in observations:
        step = observation.get("step")
        action = observation.get("action", {})
        result = observation.get("result", {})
        name = action.get("action", "action")
        if result.get("ok"):
            detail = result.get("path") or result.get("returncode")
            lines.append(f"Step {step}: {name} completed" + (f" ({detail})" if detail is not None else "."))
            if name == "run_shell":
                output = _format_shell_result(result)
                if output:
                    lines.append(output)
        else:
            lines.append(f"Step {step}: {name} failed.")
            lines.append(_format_shell_result(result) if name == "run_shell" else str(result.get("error", "unknown error")))
    return "\n".join(lines).strip()


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
