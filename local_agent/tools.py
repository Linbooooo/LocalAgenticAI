from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import AgentConfig


@dataclass(frozen=True)
class ShellResult:
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return not self.cancelled and not self.error and self.returncode == 0


class WorkspaceShell:
    """Run independent Bash commands from the configured workspace."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def run(self, command: str) -> ShellResult:
        blocked = blocked_command_reason(command, allow_network=self.config.allow_network_tools)
        if blocked:
            return ShellResult(error=blocked)
        if not self._confirm(command):
            return ShellResult(cancelled=True)

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.config.workspace,
                capture_output=True,
                text=True,
                timeout=self.config.shell_timeout,
                executable="/bin/bash" if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return ShellResult(
                stdout=tail_text(exc.stdout),
                stderr=tail_text(exc.stderr),
                error=f"Command timed out after {self.config.shell_timeout}s.",
            )
        return ShellResult(
            returncode=completed.returncode,
            stdout=tail_text(completed.stdout),
            stderr=tail_text(completed.stderr),
        )

    def _confirm(self, command: str) -> bool:
        if self.config.trust == "auto":
            return True
        answer = input(f"Allow shell command:\n{command}\n[y/N] ").strip().lower()
        return answer == "yes" or bool(re.fullmatch(r"y+", answer))


def blocked_command_reason(command: str, *, allow_network: bool) -> str | None:
    try:
        tokens = [token.lower() for token in shlex.split(command)]
    except ValueError as exc:
        return f"Could not parse shell command: {exc}"

    names = {Path(token).name for token in tokens}
    pairs = {f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)}

    if not allow_network:
        network = {"curl", "wget", "ssh", "scp", "sftp", "rsync"}
        installs = {"git clone", "git pull", "pip install", "pip3 install", "npm install", "uv add", "uv pip"}
        if names.intersection(network) or pairs.intersection(installs):
            return "Blocked by the local-only network policy."

    if names.intersection({"mkfs", "dd"}) or pairs.intersection({"git reset"}):
        return "Blocked because the command is destructive."
    if "rm" in names and any(token.startswith("-") and "r" in token and "f" in token for token in tokens):
        return "Blocked because recursive forced deletion is destructive."
    return None


def tail_text(value: str | bytes | None, limit: int = 16000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:].strip()
