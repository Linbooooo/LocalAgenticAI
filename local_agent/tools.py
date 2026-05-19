from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import AgentConfig
from .hardware import hardware_report
from .ollama_client import OllamaClient, OllamaConnectionError


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    mutates: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._tools = {
            tool.name: tool
            for tool in [
                Tool(
                    "list_files",
                    "List files under a workspace path.",
                    _object_schema(
                        {
                            "path": {"type": "string", "description": "Workspace-relative path."},
                            "max_depth": {"type": "integer", "minimum": 0, "maximum": 12},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                        }
                    ),
                    self._list_files,
                ),
                Tool(
                    "read_file",
                    "Read a UTF-8 text file from the workspace.",
                    _object_schema(
                        {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "max_lines": {"type": "integer", "minimum": 1, "maximum": 1000},
                        },
                        required=["path"],
                    ),
                    self._read_file,
                ),
                Tool(
                    "search_text",
                    "Search text files in the workspace with a regular expression.",
                    _object_schema(
                        {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                            "file_glob": {"type": "string"},
                            "case_sensitive": {"type": "boolean"},
                            "max_matches": {"type": "integer", "minimum": 1, "maximum": 500},
                        },
                        required=["pattern"],
                    ),
                    self._search_text,
                ),
                Tool(
                    "write_file",
                    "Write a UTF-8 text file inside the workspace, creating parent directories as needed.",
                    _object_schema(
                        {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        required=["path", "content"],
                    ),
                    self._write_file,
                    mutates=True,
                ),
                Tool(
                    "replace_in_file",
                    "Replace exact text inside a workspace file.",
                    _object_schema(
                        {
                            "path": {"type": "string"},
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                            "max_replacements": {"type": "integer", "minimum": 1, "maximum": 100},
                        },
                        required=["path", "old", "new"],
                    ),
                    self._replace_in_file,
                    mutates=True,
                ),
                Tool(
                    "run_shell",
                    "Run a local shell command in the workspace.",
                    _object_schema(
                        {
                            "command": {"type": "string"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                        required=["command"],
                    ),
                    self._run_shell,
                    mutates=True,
                ),
                Tool(
                    "hardware_profile",
                    "Return the detected local hardware profile.",
                    _object_schema({}),
                    self._hardware_profile,
                ),
            ]
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        if tool.mutates and not self._confirm(name, arguments):
            return {"ok": False, "error": "User declined tool execution."}
        try:
            return tool.handler(arguments)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _confirm(self, name: str, arguments: dict[str, Any]) -> bool:
        if self.config.trust == "auto":
            return True
        preview = json.dumps(arguments, ensure_ascii=False)
        if len(preview) > 400:
            preview = preview[:397] + "..."
        answer = input(f"Allow {name} {preview}? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def _resolve(self, raw_path: str | None = None) -> Path:
        raw_path = raw_path or "."
        path = (self.config.workspace / raw_path).resolve()
        if not path.is_relative_to(self.config.workspace):
            raise ValueError(f"Path escapes workspace: {raw_path}")
        return path

    def _list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        root = self._resolve(arguments.get("path"))
        max_depth = int(arguments.get("max_depth", 4))
        limit = int(arguments.get("limit", 200))
        if not root.exists():
            return {"ok": False, "error": f"Path does not exist: {root}"}

        files: list[str] = []
        base_depth = len(root.relative_to(self.config.workspace).parts)
        for current, dirs, names in os.walk(root):
            current_path = Path(current)
            rel_depth = len(current_path.relative_to(self.config.workspace).parts) - base_depth
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "__pycache__"}]
            if rel_depth >= max_depth:
                dirs[:] = []
            for name in sorted(names):
                rel = str((current_path / name).relative_to(self.config.workspace))
                files.append(rel)
                if len(files) >= limit:
                    return {"ok": True, "files": files, "truncated": True}
        return {"ok": True, "files": files, "truncated": False}

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments["path"])
        start_line = int(arguments.get("start_line", 1))
        max_lines = int(arguments.get("max_lines", 200))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start_index = max(start_line - 1, 0)
        selected = lines[start_index : start_index + max_lines]
        numbered = [f"{start_index + index + 1}: {line}" for index, line in enumerate(selected)]
        return {
            "ok": True,
            "path": str(path.relative_to(self.config.workspace)),
            "content": "\n".join(numbered),
            "total_lines": len(lines),
        }

    def _search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        root = self._resolve(arguments.get("path"))
        pattern = arguments["pattern"]
        file_glob = arguments.get("file_glob", "*")
        max_matches = int(arguments.get("max_matches", 100))
        flags = 0 if arguments.get("case_sensitive", False) else re.IGNORECASE
        regex = re.compile(pattern, flags)
        matches: list[dict[str, Any]] = []

        for file_path in _iter_text_files(root, self.config.workspace, file_glob):
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "path": str(file_path.relative_to(self.config.workspace)),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= max_matches:
                        return {"ok": True, "matches": matches, "truncated": True}
        return {"ok": True, "matches": matches, "truncated": False}

    def _write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        return {"ok": True, "path": str(path.relative_to(self.config.workspace))}

    def _replace_in_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments["path"])
        old = arguments["old"]
        new = arguments["new"]
        max_replacements = int(arguments.get("max_replacements", 1))
        content = path.read_text(encoding="utf-8")
        count = content.count(old)
        if count == 0:
            return {"ok": False, "error": "Old text not found."}
        updated = content.replace(old, new, max_replacements)
        path.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path.relative_to(self.config.workspace)),
            "replacements": min(count, max_replacements),
        }

    def _run_shell(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments["command"]
        timeout = int(arguments.get("timeout_seconds", 120))
        blocked = _blocked_command_reason(command, allow_network=self.config.allow_network_tools)
        if blocked:
            return {"ok": False, "error": blocked}

        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.config.workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable="/bin/bash" if os.name == "posix" else None,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }

    def _hardware_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ollama: dict[str, Any]
        try:
            client = OllamaClient(self.config.ollama_url, timeout=self.config.ollama_timeout)
            ollama = {"version": client.version(), "loaded": client.ps().get("models", [])}
        except OllamaConnectionError as exc:
            ollama = {"error": str(exc)}
        return {"ok": True, "hardware": hardware_report(), "ollama": ollama}


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _iter_text_files(root: Path, workspace: Path, file_glob: str):
    if root.is_file():
        if fnmatch.fnmatch(root.name, file_glob):
            yield root
        return

    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "__pycache__"}]
        for name in names:
            file_path = Path(current) / name
            if not fnmatch.fnmatch(name, file_glob):
                continue
            if not file_path.resolve().is_relative_to(workspace):
                continue
            if _looks_binary(file_path):
                continue
            yield file_path


def _looks_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:2048]
    except OSError:
        return True
    return b"\0" in chunk


def _blocked_command_reason(command: str, *, allow_network: bool) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"Could not parse shell command: {exc}"
    lowered = [token.lower() for token in tokens]
    if not allow_network:
        network_tools = {
            "curl",
            "wget",
            "ssh",
            "scp",
            "sftp",
            "rsync",
            "git-clone",
            "git-pull",
            "pip-install",
            "npm-install",
            "pnpm-install",
            "yarn-add",
            "uv-add",
            "uv-pip",
        }
        joined_pairs = {f"{lowered[i]}-{lowered[i + 1]}" for i in range(len(lowered) - 1)}
        singles = {Path(token).name for token in lowered}
        joined_triples = {
            f"{lowered[i]}-{lowered[i + 1]}-{lowered[i + 2]}" for i in range(len(lowered) - 2)
        }
        if network_tools.intersection(singles | joined_pairs | joined_triples):
            return "Blocked by local-only policy. Set allow_network_tools=true only when you intentionally want downloads."

    dangerous = {"git-reset", "mkfs", "dd"}
    joined_pairs = {f"{lowered[i]}-{lowered[i + 1]}" for i in range(len(lowered) - 1)}
    if dangerous.intersection(joined_pairs | set(lowered)):
        return "Blocked because the command looks destructive."
    if "rm" in lowered and any("r" in token and "f" in token for token in lowered if token.startswith("-")):
        return "Blocked because the command looks destructive."
    return None
