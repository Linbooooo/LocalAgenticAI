from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig
from .context import prepare_messages
from .ollama_client import ModelMetrics, OllamaClient
from .tools import ShellResult, WorkspaceShell


SYSTEM_PROMPT = """You are a local coding agent working in:
{workspace}

Use the shell to inspect files, edit code, and run verification. To take an action,
respond with exactly one fenced bash block:

```bash
command
```

The command runs from the workspace root. Its stdout, stderr, and exit code will be
returned to you. Continue until the request is complete, then answer in plain text
without a bash block. Base claims on observed command results. Keep changes focused.
Use non-interactive commands and do not access the network unless the user requested it.
This environment provides `python3`; `python` may not exist. Inspect relevant files before
editing. After a failed command, diagnose the observed error and try a correction instead
of assuming the required tool is unavailable. Never describe a future action in a final
answer: issue its Bash block now. For a requested code change, inspect `git diff` and do
not finish with an empty diff.
"""

SHELL_BLOCK = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class AgentResult:
    content: str
    turns: int
    commands: int
    elapsed_ms: float
    time_to_first_shell_ms: float | None
    model_metrics: list[ModelMetrics] = field(default_factory=list)


class LocalAgent:
    """A linear model -> shell -> observation loop."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client: OllamaClient | None = None,
        shell: WorkspaceShell | None = None,
    ) -> None:
        self.config = config
        self.client = client or OllamaClient(config.ollama_url, timeout=config.ollama_timeout)
        self.shell = shell or WorkspaceShell(config)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(workspace=config.workspace)}
        ]

    def run(self, task: str) -> AgentResult:
        started = time.perf_counter()
        first_shell_at: float | None = None
        command_count = 0
        metrics: list[ModelMetrics] = []
        self.messages.append({"role": "user", "content": task})

        for turn in range(1, self.config.max_steps + 1):
            response, model_metrics = self.client.chat_stream(
                model=self.config.model,
                messages=prepare_messages(self.messages, self.config.context_budget_tokens()),
                options=self.config.ollama_options(),
                keep_alive=self.config.keep_alive,
            )
            metrics.append(model_metrics)
            content = str(response.get("message", {}).get("content", "")).strip()
            self.messages.append({"role": "assistant", "content": content})

            command = extract_shell_command(content)
            if command is None:
                return self._result(content, turn, command_count, started, first_shell_at, metrics)

            if first_shell_at is None:
                first_shell_at = time.perf_counter()

            command_count += 1
            result = self.shell.run(command)
            self.messages.append({"role": "user", "content": format_shell_observation(command, result)})
            if result.cancelled:
                message = "Shell command was not run because approval was declined."
                self.messages.append({"role": "assistant", "content": message})
                return self._result(message, turn, command_count, started, first_shell_at, metrics)

        message = f"Stopped after {self.config.max_steps} model turns without a final answer."
        self.messages.append({"role": "assistant", "content": message})
        return self._result(
            message,
            self.config.max_steps,
            command_count,
            started,
            first_shell_at,
            metrics,
        )

    @staticmethod
    def _result(
        content: str,
        turns: int,
        commands: int,
        started: float,
        first_shell_at: float | None,
        metrics: list[ModelMetrics],
    ) -> AgentResult:
        return AgentResult(
            content=content,
            turns=turns,
            commands=commands,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            time_to_first_shell_ms=None if first_shell_at is None else (first_shell_at - started) * 1000,
            model_metrics=metrics,
        )


def extract_shell_command(content: str) -> str | None:
    match = SHELL_BLOCK.search(content)
    return match.group(1).strip() if match and match.group(1).strip() else None


def format_shell_observation(command: str, result: ShellResult) -> str:
    if result.cancelled:
        return (
            "<shell_result>\n"
            f"command: {command}\n"
            "status: declined by user\n"
            "</shell_result>"
        )

    lines = [
        "<shell_result>",
        f"command: {command}",
        f"exit_code: {result.returncode}",
    ]
    if result.stdout:
        lines.extend(["stdout:", result.stdout])
    if result.stderr:
        lines.extend(["stderr:", result.stderr])
    if result.error:
        lines.extend(["error:", result.error])
    lines.extend(
        [
            "</shell_result>",
            "Continue working if needed. If the task is complete, answer in plain text without a bash block.",
        ]
    )
    return "\n".join(lines)
