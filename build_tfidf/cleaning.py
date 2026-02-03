"""Content cleaning helpers."""

from __future__ import annotations

import re

FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def strip_front_matter(text: str) -> str:
    return FRONT_MATTER_RE.sub("", text, count=1)


def strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str, remove_code: bool = False) -> str:
    text = strip_front_matter(text)
    if remove_code:
        text = strip_code_fences(text)
    return normalize_whitespace(text)
