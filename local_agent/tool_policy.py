from __future__ import annotations

import re
import shlex


def classify_intent(task: str) -> str:
    text = task.lower()
    if _has_any(text, {"add", "change", "create", "delete", "edit", "fix", "implement", "modify", "patch", "refactor", "remove", "rename", "replace", "save", "update", "write"}):
        return "edit"
    if _has_any(text, {"build", "check", "compile", "doctor", "execute", "install", "lint", "run", "shell", "start", "test", "terminal", "verify"}):
        return "shell"
    if _has_any(text, {"cpu", "gpu", "hardware", "memory", "ollama", "ram", "vram"}):
        return "hardware"
    if _has_any(text, {"analyze", "codebase", "display", "explain", "file", "files", "find", "inspect", "list", "look", "read", "repo", "repository", "review", "search", "show", "summarize", "tell", "what", "where"}):
        return "read"
    return "chat"


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
    if not tokens or tokens[0] in {"it", "that", "the", "this"}:
        return None
    return candidate if _looks_like_command(tokens[0]) else None


def _has_any(text: str, words: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _looks_like_command(first_token: str) -> bool:
    known = {"cat", "docker", "git", "ls", "make", "node", "npm", "nvidia-smi", "python", "python3", "pytest"}
    return first_token in known or "/" in first_token or "." in first_token or "-" in first_token
