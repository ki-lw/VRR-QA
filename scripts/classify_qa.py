#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_qa.py
==============

Stage 0 of the pipeline: assign each test question to one of the 9 VRR-QA
reasoning categories. A high-precision rule-based classifier handles the clear
cases; ambiguous questions fall back to an LLM (Qwen3-4B in the competition;
here, by default, the Gemini text client).

This reproduces ``test_qa_with_category.json`` from ``test_qa.json``.

Usage:
  # rule-based only (no API key needed)
  python scripts/classify_qa.py --test-qa test_qa.json --out test_qa_with_category.json

  # rule-based + LLM fallback for ambiguous items (needs GEMINI_API_KEY)
  python scripts/classify_qa.py --test-qa test_qa.json --out test_qa_with_category.json --use-llm
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classify import classify          # noqa: E402
from src.prompts import CATEGORY_NAMES     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-qa", default="test_qa.json")
    ap.add_argument("--out", default="test_qa_with_category.json")
    ap.add_argument("--use-llm", action="store_true",
                    help="enable the LLM fallback for ambiguous questions")
    args = ap.parse_args()

    gem = None
    if args.use_llm:
        from src.api import GeminiClient
        gem = GeminiClient()

    with open(args.test_qa, "r", encoding="utf-8") as f:
        items = json.load(f)

    out = []
    for it in items:
        cid = classify(it.get("question_text", ""), it.get("options", {}), gem)
        rec = dict(it)
        rec["category_id"] = cid
        rec["category"] = CATEGORY_NAMES.get(cid)
        out.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[classify] wrote {len(out)} items -> {args.out}")


if __name__ == "__main__":
    main()
