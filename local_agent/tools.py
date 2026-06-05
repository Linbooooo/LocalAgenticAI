from __future__ import annotations

import fnmatch
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .hardware import hardware_report
from .ollama_client import OllamaClient, OllamaConnectionError


class WorkspaceTools:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def list_files(self, path: str = ".", max_depth: int = 4, limit: int = 200) -> dict[str, Any]:
        root = self._resolve(path)
        if not root.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}

        files: list[str] = []
        base_depth = len(root.relative_to(self.config.workspace).parts)
        for current, dirs, names in os.walk(root):
            current_path = Path(current)
            rel_depth = len(current_path.relative_to(self.config.workspace).parts) - base_depth
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "__pycache__"}]
            if rel_depth >= max_depth:
                dirs[:] = []
            for name in sorted(names):
                files.append(str((current_path / name).relative_to(self.config.workspace)))
                if len(files) >= limit:
                    return {"ok": True, "files": files, "truncated": True}
        return {"ok": True, "files": files, "truncated": False}

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> dict[str, Any]:
        file_path = self._resolve(path)
        if not file_path.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}
        if file_path.is_dir():
            return {"ok": False, "error": f"Path is a directory: {path}"}
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        start_index = max(start_line - 1, 0)
        selected = lines[start_index : start_index + max_lines]
        numbered = [f"{start_index + index + 1}: {line}" for index, line in enumerate(selected)]
        return {
            "ok": True,
            "path": str(file_path.relative_to(self.config.workspace)),
            "content": "\n".join(numbered),
            "total_lines": len(lines),
        }

    def search_text(
        self,
        pattern: str,
        path: str = ".",
        file_glob: str = "*",
        case_sensitive: bool = False,
        max_matches: int = 100,
    ) -> dict[str, Any]:
        root = self._resolve(path)
        if not root.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return {"ok": False, "error": f"Invalid search pattern: {exc}"}
        matches: list[dict[str, Any]] = []

        for file_path in _iter_text_files(root, self.config.workspace, file_glob):
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
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

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        if not self._confirm("write_file", {"path": path}):
            return {"ok": False, "error": "User declined tool execution."}
        file_path = self._resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(file_path.relative_to(self.config.workspace))}

    def replace_in_file(self, path: str, old: str, new: str, max_replacements: int = 1) -> dict[str, Any]:
        if not self._confirm("replace_in_file", {"path": path, "max_replacements": max_replacements}):
            return {"ok": False, "error": "User declined tool execution."}
        file_path = self._resolve(path)
        if not file_path.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}
        if file_path.is_dir():
            return {"ok": False, "error": f"Path is a directory: {path}"}
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        count = content.count(old)
        if count == 0:
            return {"ok": False, "error": "Old text not found."}
        file_path.write_text(content.replace(old, new, max_replacements), encoding="utf-8")
        return {
            "ok": True,
            "path": str(file_path.relative_to(self.config.workspace)),
            "replacements": min(count, max_replacements),
        }

    def delete_file(self, path: str) -> dict[str, Any]:
        if not self._confirm("delete_file", {"path": path}):
            return {"ok": False, "error": "User declined tool execution."}
        file_path = self._resolve(path)
        if not file_path.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}
        if file_path.is_dir():
            return {"ok": False, "error": f"Path is a directory: {path}"}
        file_path.unlink()
        return {"ok": True, "path": str(file_path.relative_to(self.config.workspace))}

    def run_shell(self, command: str, timeout_seconds: int = 120, stdin: str | None = None) -> dict[str, Any]:
        command = _normalize_shell_command(command)
        blocked = _blocked_command_reason(command, allow_network=self.config.allow_network_tools)
        if blocked:
            return {"ok": False, "error": blocked}
        preview = {"command": command, "timeout_seconds": timeout_seconds}
        if stdin is not None:
            preview["stdin_bytes"] = len(stdin.encode("utf-8"))
        if not self._confirm("run_shell", preview):
            return {"ok": False, "error": "User declined tool execution."}

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.config.workspace,
                capture_output=True,
                input=stdin,
                text=True,
                timeout=timeout_seconds,
                executable="/bin/bash" if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "error": f"Command timed out after {timeout_seconds}s.",
                "stdout": _tail_text(exc.stdout),
                "stderr": _tail_text(exc.stderr),
            }
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": _tail_text(completed.stdout),
            "stderr": _tail_text(completed.stderr),
        }

    def hardware_profile(self) -> dict[str, Any]:
        try:
            client = OllamaClient(self.config.ollama_url, timeout=self.config.ollama_timeout)
            ollama: dict[str, Any] = {"version": client.version(), "loaded": client.ps().get("models", [])}
        except OllamaConnectionError as exc:
            ollama = {"error": str(exc)}
        return {"ok": True, "hardware": hardware_report(), "ollama": ollama}

    def _resolve(self, raw_path: str | None = None) -> Path:
        path = (self.config.workspace / (raw_path or ".")).resolve()
        if not path.is_relative_to(self.config.workspace):
            raise ValueError(f"Path escapes workspace: {raw_path}")
        return path

    def _confirm(self, name: str, arguments: dict[str, Any]) -> bool:
        if self.config.trust == "auto":
            return True
        preview = " ".join(f"{key}={value!r}" for key, value in arguments.items())
        answer = input(f"Allow {name} {preview}? [y/N] ").strip().lower()
        return answer == "yes" or bool(re.fullmatch(r"y+", answer))


def _iter_text_files(root: Path, workspace: Path, file_glob: str):
    if root.is_file():
        if fnmatch.fnmatch(root.name, file_glob) and not _looks_binary(root):
            yield root
        return

    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "__pycache__"}]
        for name in names:
            file_path = Path(current) / name
            if fnmatch.fnmatch(name, file_glob) and file_path.resolve().is_relative_to(workspace) and not _looks_binary(file_path):
                yield file_path


def _looks_binary(path: Path) -> bool:
    try:
        return b"\0" in path.read_bytes()[:2048]
    except OSError:
        return True


def _normalize_shell_command(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if tokens and Path(tokens[0]).name == "python" and shutil.which("python") is None and shutil.which("python3") is not None:
        tokens = ["python3", *tokens[1:]]
    test_command = _unittest_discover_command(tokens)
    if test_command:
        return test_command
    return shlex.join(tokens) if tokens else command


def _unittest_discover_command(tokens: list[str]) -> str | None:
    if len(tokens) != 2 or Path(tokens[0]).name not in {"python", "python3"}:
        return None
    path = Path(tokens[1].replace("\\", "/"))
    if len(path.parts) >= 2 and path.parts[0] == "tests" and path.name.startswith("test_") and path.suffix == ".py":
        return f"python3 -m unittest discover -s tests -p {shlex.quote(path.name)}"
    return None


def _tail_text(value: str | bytes | None, limit: int = 12000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _blocked_command_reason(command: str, *, allow_network: bool) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"Could not parse shell command: {exc}"
    lowered = [token.lower() for token in tokens]

    if not allow_network:
        singles = {Path(token).name for token in lowered}
        pairs = {f"{lowered[i]}-{lowered[i + 1]}" for i in range(len(lowered) - 1)}
        triples = {f"{lowered[i]}-{lowered[i + 1]}-{lowered[i + 2]}" for i in range(len(lowered) - 2)}
        blocked = {
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
        if blocked.intersection(singles | pairs | triples):
            return "Blocked by local-only policy. Set allow_network_tools=true only when you intentionally want downloads."

    pairs = {f"{lowered[i]}-{lowered[i + 1]}" for i in range(len(lowered) - 1)}
    if {"git-reset", "mkfs", "dd"}.intersection(set(lowered) | pairs):
        return "Blocked because the command looks destructive."
    if "rm" in lowered and any("r" in token and "f" in token for token in lowered if token.startswith("-")):
        return "Blocked because the command looks destructive."
    return None
