from __future__ import annotations

from typing import Any


def prepare_messages(messages: list[dict[str, Any]], token_budget: int) -> list[dict[str, Any]]:
    if _estimate_messages(messages) <= token_budget:
        return messages
    if not messages:
        return messages

    head = messages[0]
    remaining_budget = max(512, token_budget - _estimate_message(head))

    recent: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages[1:]):
        cost = _estimate_message(message)
        if recent and used + cost > remaining_budget:
            break
        recent.append(message)
        used += cost
    recent.reverse()

    omitted_count = max(0, len(messages) - 1 - len(recent))
    if omitted_count == 0:
        return _fit_messages_to_budget([head, *recent], token_budget)

    omitted = messages[1 : 1 + omitted_count]
    summary = _summarize_messages(omitted)
    summary_message = {
        "role": "system",
        "content": (
            "Condensed earlier context. Use this as background, and re-read files with tools when exact "
            f"details matter.\n{summary}"
        ),
    }

    packed = [head, summary_message, *recent]
    while len(packed) > 3 and _estimate_messages(packed) > token_budget:
        packed.pop(2)
    while _estimate_messages(packed) > token_budget and len(summary_message["content"]) > 400:
        summary_message["content"] = _truncate(summary_message["content"], len(summary_message["content"]) // 2)
    if _estimate_messages(packed) > token_budget and recent:
        return _fit_messages_to_budget([head, recent[-1]], token_budget)
    return _fit_messages_to_budget(packed, token_budget)


def _fit_messages_to_budget(messages: list[dict[str, Any]], token_budget: int) -> list[dict[str, Any]]:
    fitted = [dict(message) for message in messages]
    while _estimate_messages(fitted) > token_budget:
        candidates = [
            (index, len(str(message.get("content", ""))))
            for index, message in enumerate(fitted)
            if index > 0 and len(str(message.get("content", ""))) > 400
        ]
        if not candidates:
            break
        index, length = max(candidates, key=lambda item: item[1])
        over_tokens = _estimate_messages(fitted) - token_budget
        target_length = max(400, length - max(400, over_tokens * 4 + 64))
        fitted[index]["content"] = _truncate_middle(str(fitted[index].get("content", "")), target_length)
    return fitted


def _summarize_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages[-16:]:
        role = message.get("role", "message")
        content = _one_line(str(message.get("content", "")))
        if content:
            lines.append(f"- {role}: {_truncate(content, 240)}")
    if not lines:
        return "- Earlier context omitted because it contained no useful text."
    omitted = max(0, len(messages) - 16)
    if omitted:
        lines.insert(0, f"- {omitted} older messages omitted.")
    return _truncate("\n".join(lines), 1200)


def _estimate_messages(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_message(message) for message in messages)


def _estimate_message(message: dict[str, Any]) -> int:
    return 8 + _estimate_tokens(str(message.get("role", ""))) + _estimate_tokens(str(message.get("content", "")))


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[middle context omitted]...\n"
    available = max(0, limit - len(marker))
    head_length = int(available * 0.6)
    tail_length = available - head_length
    return f"{text[:head_length]}{marker}{text[-tail_length:] if tail_length else ''}"
