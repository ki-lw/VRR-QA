# -*- coding: utf-8 -*-
"""Question classifier (Stage 0).

A high-precision rule-based classifier handles the clear cases; ambiguous
questions fall back to an LLM. We used Qwen3-4B for the fallback during the
competition, but any chat model works -- by default we route the fallback
through the Gemini text client (set CLASSIFIER_MODEL / GEMINI_API_KEY).

Output: an integer category id in 1..9 (see prompts.CATEGORY_NAMES).
"""

import re
from typing import Optional

from . import prompts

# --- rule regexes ---------------------------------------------------------
RE_COUNT = re.compile(r"\bhow many\b", re.I)
RE_CAUSAL = re.compile(
    r"^\s*why\b|\bwhat caused\b|\bcaused (the|a|an)\b|"
    r"\b(actions?|action) (caused|cause|that caused)\b|"
    r"\bmainly responsible for\b|\bresponsible for\b", re.I)
RE_RELATIONSHIP = re.compile(
    r"\brelationship between\b|\bwhat is the relationship\b", re.I)
RE_WHO_IDENT = re.compile(
    r"^\s*who (gives|gave|talks?|gets|sees|does (the|a)|pass(es)? the)\b|"
    r"^\s*who (is|are) (?!the (furthest|closest|nearest|first|second|third|"
    r"tallest|shortest|biggest|largest|smallest|fastest|slowest|"
    r"farthest|loudest))", re.I)
RE_VIEW = re.compile(
    r"\b(point of view|camera('s)? perspective|from the (camera|perspective)|"
    r"able to see|can [\w \-'/]{0,60}?(see|read)\b|"
    r"is the [\w \-]+ visible|is the [\w \-]+ able to|"
    r"pointing (towards|at|to)|from .* perspective)\b", re.I)
RE_LOOKING = re.compile(
    r"\bare .* looking at\b|\bwhat is .* (looking at|pointing)\b", re.I)
RE_DIRECTION = re.compile(r"\bin (what|which) direction\b", re.I)
RE_MOTION_VERB = re.compile(
    r"\b(walking|walks?|running|runs?|moving|moves?|move|"
    r"turning|turns?|turn|driving|drives?|drive|driven|"
    r"flying|flies|fly|jumping|jumps?|sliding|slides?|"
    r"throwing|throws?|throw|spinning|spins?|spin|"
    r"rolling|rolls?|roll|sailing|sails?|carousel)\b", re.I)
RE_VERT_OPT = re.compile(
    r"\b(above|below|on top|underneath|higher than|lower than|"
    r"at the same height|directly above|directly below|"
    r"at (his|her|their|its|the) (level|height)|same level)\b", re.I)
RE_DEPTH_OPT = re.compile(
    r"\b(in front|behind|closer|farther|further away|further from|"
    r"same distance|directly toward|directly towards|directly away|"
    r"right next to|next to (him|her|them))\b", re.I)
RE_LATERAL_OPT = re.compile(
    r"\b(adjacent side|opposite side|same side|side of the (table|street)|"
    r"to (his|her|their|its|the) (left|right)|"
    r"on (his|her|their|its|the) (left|right)|"
    r"from left to right|from right to left|perpendicular|side by side|"
    r"facing (the same direction|directly (toward|away)))\b", re.I)
RE_BARE_LR = re.compile(r"^\s*(to the )?(left|right)\s*$", re.I)
RE_FACING = re.compile(
    r"\bfacing\b[^.?!]*\brelative\b|\borientation\b[^.?!]*\brelative\b", re.I)
RE_SUPERLATIVE_DEPTH = re.compile(r"\b(furthest|farthest|closest|nearest)\b", re.I)
RE_WHICH_ID = re.compile(
    r"^\s*which (car|vehicle|character|animal|species|box|building|sign|"
    r"color|shirt|two|colors?|person|character'?s?|character's|two characters)\b", re.I)
RE_ORDER = re.compile(
    r"\b(order of (appearance|the animals|the vehicles)|arrange in order|"
    r"comes first in chronological|in chronological order|first to last|"
    r"second (distinct|to)|order of the)\b", re.I)
RE_HOW_DOES = re.compile(r"^\s*how (do|does)\b", re.I)


def _options_lateral_only(options: dict) -> bool:
    if not options:
        return False
    return all(RE_BARE_LR.match(str(v).strip()) for v in options.values())


def classify_rule(question: str, options: dict) -> Optional[int]:
    """Return a category id 1..9 if a high-confidence rule fires, else None."""
    q = (question or "").strip()
    opt_text = " ".join(options.values()) if options else ""

    if RE_COUNT.search(q):
        return 6
    if RE_CAUSAL.search(q):
        return 8
    if RE_RELATIONSHIP.search(q):
        return 9
    if RE_WHO_IDENT.search(q):
        return 9
    if re.match(r"^\s*who (is|are)\b", q, re.I) and RE_SUPERLATIVE_DEPTH.search(q):
        return 3
    if RE_VIEW.search(q) or RE_LOOKING.search(q):
        return 5
    if RE_DIRECTION.search(q) and RE_MOTION_VERB.search(q):
        return 4
    if RE_MOTION_VERB.search(q) and re.search(r"\bdirection\b", q, re.I):
        return 4
    if RE_VERT_OPT.search(opt_text):
        return 2
    if RE_DEPTH_OPT.search(opt_text):
        return 3
    if RE_LATERAL_OPT.search(opt_text):
        return 1
    if _options_lateral_only(options):
        return 1
    if RE_FACING.search(q):
        return 1
    if RE_DIRECTION.search(q):
        return 4
    if RE_ORDER.search(q):
        return 7
    if RE_WHICH_ID.search(q):
        return 7
    if RE_HOW_DOES.search(q):
        return 8
    return None


def classify_llm(question: str, options: dict, gemini_client) -> int:
    """LLM fallback. Returns 1..9 (defaults to 7 if the reply is unparseable)."""
    sys = prompts.build_classifier_system_prompt()
    usr = prompts.build_classifier_user_prompt(question, options)
    try:
        out = gemini_client.generate_text(usr, system=sys)
        m = re.search(r"[1-9]", out or "")
        if m:
            return int(m.group(0))
    except Exception:
        pass
    return 7


def classify(question: str, options: dict, gemini_client=None) -> int:
    """Rule-based first, LLM fallback for ambiguous items."""
    cid = classify_rule(question, options)
    if cid is not None:
        return cid
    if gemini_client is not None:
        return classify_llm(question, options, gemini_client)
    return 7  # safe default category when no LLM client is provided
