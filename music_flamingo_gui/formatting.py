from __future__ import annotations

import re
from pathlib import Path


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def split_reasoning(text: str) -> tuple[str, str]:
    cleaned = text.strip()
    match = THINK_PATTERN.search(cleaned)
    if not match:
        return "", cleaned

    reasoning = match.group(1).strip()
    answer = THINK_PATTERN.sub("", cleaned).strip()
    return reasoning, answer or cleaned


def summarize_user_message(prompt: str, audio_path: str | None) -> str:
    parts: list[str] = []

    if prompt.strip():
        parts.append(prompt.strip())

    if audio_path:
        parts.append(f"Hangfajl: {Path(audio_path).name}")

    if not parts:
        return "Ures kerdes"

    return "\n\n".join(parts)
