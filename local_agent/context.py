from __future__ import annotations

from typing import Any


def prepare_messages(messages: list[dict[str, Any]], token_budget: int) -> list[dict[str, Any]]:
    """Keep the system prompt and newest complete messages within a rough token budget."""
    if estimate_messages(messages) <= token_budget or len(messages) <= 2:
        return messages

    system = messages[0]
    kept: list[dict[str, Any]] = []
    used = estimate_message(system) + 20
    for message in reversed(messages[1:]):
        cost = estimate_message(message)
        if kept and used + cost > token_budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    omitted = len(messages) - len(kept) - 1
    notice = {
        "role": "system",
        "content": (
            f"{omitted} earlier messages were omitted to fit the local context window. "
            "Re-inspect files when exact prior state matters."
        ),
    }
    return [system, notice, *kept]


def estimate_messages(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message(message) for message in messages)


def estimate_message(message: dict[str, Any]) -> int:
    return 8 + max(1, len(str(message.get("content", ""))) // 4)
