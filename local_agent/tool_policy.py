from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


READ_TOOLS = frozenset({"list_files", "read_file", "search_text"})
HARDWARE_TOOLS = frozenset({"hardware_profile"})
MUTATING_TOOLS = frozenset({"write_file", "replace_in_file", "run_shell"})
ALL_TOOLS = READ_TOOLS | HARDWARE_TOOLS | MUTATING_TOOLS


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    allowed_tools: frozenset[str]

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


def classify_tool_policy(task: str) -> ToolPolicy:
    text = task.lower()
    if _has_edit_intent(text):
        return ToolPolicy("edit", ALL_TOOLS)
    if _has_shell_intent(text):
        return ToolPolicy("shell", READ_TOOLS | HARDWARE_TOOLS | frozenset({"run_shell"}))
    if _has_hardware_intent(text):
        return ToolPolicy("hardware", HARDWARE_TOOLS)
    if _has_read_intent(text):
        return ToolPolicy("read", READ_TOOLS)
    return ToolPolicy("chat", frozenset())


def extract_direct_shell_command(task: str) -> str | None:
    text = task.strip()
    quoted = re.match(r"^(?:run|execute)\s+(?:the\s+command\s+)?(?:`([^`]+)`|'([^']+)'|\"([^\"]+)\")\s*\.?$", text, re.IGNORECASE)
    if quoted:
        return next(group for group in quoted.groups() if group)

    unquoted = re.match(r"^(?:run|execute)\s+(.+?)\s*\.?$", text, re.IGNORECASE)
    if not unquoted:
        return None
    candidate = unquoted.group(1).strip()
    try:
        tokens = shlex.split(candidate)
    except ValueError:
        return None
    if not tokens:
        return None
    first = tokens[0]
    if first in {"it", "that", "the", "this"}:
        return None
    if _looks_like_command(first):
        return candidate
    return None


def _has_edit_intent(text: str) -> bool:
    edit_words = {
        "add",
        "change",
        "create",
        "delete",
        "edit",
        "fix",
        "implement",
        "modify",
        "patch",
        "refactor",
        "remove",
        "rename",
        "replace",
        "save",
        "update",
        "write",
    }
    return _contains_word(text, edit_words)


def _has_shell_intent(text: str) -> bool:
    shell_words = {
        "build",
        "check",
        "compile",
        "doctor",
        "execute",
        "install",
        "lint",
        "run",
        "shell",
        "start",
        "test",
        "terminal",
        "verify",
    }
    return _contains_word(text, shell_words)


def _has_hardware_intent(text: str) -> bool:
    hardware_words = {"cpu", "gpu", "hardware", "memory", "ollama", "ram", "vram"}
    return _contains_word(text, hardware_words)


def _has_read_intent(text: str) -> bool:
    read_words = {
        "analyze",
        "codebase",
        "explain",
        "file",
        "files",
        "find",
        "inspect",
        "list",
        "look",
        "read",
        "repo",
        "repository",
        "review",
        "search",
        "show",
        "summarize",
        "tell",
        "what",
        "where",
    }
    return _contains_word(text, read_words)


def _contains_word(text: str, words: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _looks_like_command(first_token: str) -> bool:
    command_names = {
        "cat",
        "docker",
        "git",
        "ls",
        "make",
        "node",
        "npm",
        "nvidia-smi",
        "python",
        "python3",
        "pytest",
    }
    return first_token in command_names or "/" in first_token or "." in first_token or "-" in first_token
