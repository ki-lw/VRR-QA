# -*- coding: utf-8 -*-
"""Response parsing + deterministic ordering helpers (anti-position-bias)."""

import re
from typing import List, Optional


def options_block_and_letters(item: dict):
    """Render the options block and return (block, 'A, B, ...', [letters])."""
    opts = item.get("options") or {}
    letters = sorted(opts.keys())
    block = "\n".join(f"{L}. {opts[L]}" for L in letters)
    return block, ", ".join(letters), letters


def parse_letter(text: str, valid_letters) -> Optional[str]:
    """Pull the chosen option letter out of a model reply."""
    if not isinstance(text, str) or not text.strip():
        return None
    valid = {c.upper() for c in (valid_letters or [])}
    if not valid:
        return None
    chars = "".join(sorted(valid))
    m = re.search(r"ANSWER\s*[:：]\s*\**\s*([A-Za-z])\b", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(rf"\*\*\s*([{chars}])\s*\**", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(rf"\b([{chars}])\b", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    return None


def extract_reasoning(raw: str) -> str:
    """Extract the decisive part (CRITICAL_OBSERVATION..end) from a judge reply."""
    if not isinstance(raw, str) or not raw.strip():
        return "(no reasoning available)"
    m = re.search(r"CRITICAL_OBSERVATION\s*:", raw)
    if m:
        return raw[m.start():].strip()
    m = re.search(r"ANSWER\s*[:：]", raw)
    if m:
        return raw[m.start():].strip()
    return raw.strip()[-800:]


def expert_order(qid: str) -> List[str]:
    """Stable-per-qid permutation of the three experts to reduce position bias.

    Labels: 'gem' (direct), 'b3' (category-guided), 'b4' (conservative).
    """
    perms = [
        ("b3", "b4", "gem"), ("b3", "gem", "b4"),
        ("b4", "b3", "gem"), ("b4", "gem", "b3"),
        ("gem", "b3", "b4"), ("gem", "b4", "b3"),
    ]
    return list(perms[(hash(qid) if qid is not None else 0) % 6])


def arbiter_order(qid: str) -> List[str]:
    """Stable-per-qid order for the two judges in the final-choose stage."""
    return ["gemini", "gpt"] if (hash(qid) % 2 == 0) else ["gpt", "gemini"]
