#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_work.py
=============

Merge the per-stage work logs of the pipeline into a single, clean
``work.jsonl`` that records the full reasoning trace for every test question,
following the paper's three-stage scheme:

  Stage 1  (multi-branch generation)  : direct / category-guided / conservative
  Stage 2  (cross-model recheck)      : Gemini 3.1 Pro judge + GPT-5.5 Pro judge
  Stage 3  (final arbitration)        : Gemini 3.5 Flash chooses between the two

Inputs (per-stage logs produced by ``run_pipeline.py``):
  --ori    stage1 DIRECT branch answers    (one record per question; 172)
  --v6     stage1 multi-branch log          (b3=category, b4=conservative; 172)
  --v7     stage2 Gemini 3.1 Pro judge     (hard subset only; ~70)
  --v8     stage2 GPT-5.5 Pro judge        (hard subset only; ~70)
  --final  stage3 final-choose log         (hard subset only; ~70)
  --submission  the final merged submission json (authoritative answers; 172)

Note on the Stage-1 branches: the "direct" branch is the original Gemini direct
run (``--ori``); the "category_guided" and "conservative" branches come from the
multi-branch log (``--v6``: branch_3_category / branch_4_conservative). All
three branches are kept for every question.

The merge also strips any machine-local fields (e.g. absolute ``video_path``)
so the result is safe to publish.

Usage:
  python scripts/merge_work.py \
      --ori    raw/stage1_direct.jsonl \
      --v7     raw/stage2_gemini.jsonl \
      --v8     raw/stage2_gpt.jsonl \
      --final  raw/stage3_final_choose.jsonl \
      --submission submission.json \
      --out    work.jsonl
"""

import os
import json
import argparse
from typing import Dict, Optional


def load_latest(path: str) -> Dict[str, dict]:
    """Read a jsonl, keep the last record per question_id."""
    out: Dict[str, dict] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = r.get("question_id")
            if qid:
                out[qid] = r
    return out


def _branch(rec: Optional[dict]) -> Optional[dict]:
    """Compact {pred_letter, raw_response} for a Stage-1 branch."""
    if not rec:
        return None
    return {"pred_letter": rec.get("pred_letter"),
            "raw_response": rec.get("raw_response")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ori",    required=True, help="stage1 direct branch (172)")
    ap.add_argument("--v6",     required=True, help="stage1 multi-branch log (172)")
    ap.add_argument("--v7",     required=True, help="stage2 Gemini judge (~70)")
    ap.add_argument("--v8",     required=True, help="stage2 GPT judge (~70)")
    ap.add_argument("--final",  required=True, help="stage3 final-choose (~70)")
    ap.add_argument("--submission", required=True, help="final merged submission json")
    ap.add_argument("--test-qa", default=None,
                    help="optional test_qa.json to source clean metadata")
    ap.add_argument("--out", default="work.jsonl")
    args = ap.parse_args()

    ori = load_latest(args.ori)
    v6 = load_latest(args.v6)
    v7 = load_latest(args.v7)
    v8 = load_latest(args.v8)
    fc = load_latest(args.final)

    with open(args.submission, "r", encoding="utf-8") as f:
        final_ans = {r["question_id"]: r.get("answer_choice", "")
                     for r in json.load(f)}

    meta = {}
    if args.test_qa and os.path.exists(args.test_qa):
        with open(args.test_qa, "r", encoding="utf-8") as f:
            meta = {r["question_id"]: r for r in json.load(f)}

    # iterate over the full question set (prefer test_qa order, else ori order)
    qids = list(meta.keys()) or list(ori.keys())

    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for qid in qids:
            o = ori.get(qid, {})
            v = v6.get(qid, {})
            m = meta.get(qid, {})
            r7 = v7.get(qid)
            r8 = v8.get(qid)
            f = fc.get(qid)

            rec = {
                "question_id": qid,
                "video_id": o.get("video_id") or v.get("video_id") or m.get("video_id"),
                "video_url": o.get("video_url") or v.get("video_url") or m.get("video_url"),
                "question_text": (o.get("question_text") or v.get("question_text")
                                  or m.get("question_text")),
                "options": o.get("options") or v.get("options") or m.get("options"),
                "category_id": v.get("category_id") or (r7 or r8 or {}).get("category_id"),
                "category": v.get("category") or (r7 or r8 or {}).get("category"),
                "final_answer": final_ans.get(qid, ""),
            }

            # ---- Stage 1: the three answer branches (all 172 questions) ----
            #   direct          = original Gemini direct run        (--ori)
            #   category_guided = multi-branch b3 (category-guided)  (--v6)
            #   conservative    = multi-branch b4 (conservative)     (--v6)
            branches = {
                "direct": _branch(o),
                "category_guided": _branch(v.get("branch_3_category")),
                "conservative": _branch(v.get("branch_4_conservative")),
            }
            is_hard = qid in fc
            rec["stage1"] = {
                "branches": {k: b for k, b in branches.items() if b},
                "all_agree": (not is_hard),
            }

            # ---- Stage 2: cross-model recheck (hard only) ----
            if r7 or r8:
                rec["stage2_recheck"] = {
                    "gemini_3_1_pro": None if not r7 else {
                        "pred_letter": r7.get("pred_letter"),
                        "raw_response": (r7.get("judge") or {}).get("raw_response"),
                        "video_duration": r7.get("video_duration"),
                        "fps_used": r7.get("fps_used"),
                    },
                    "gpt_5_5_pro": None if not r8 else {
                        "pred_letter": r8.get("pred_letter"),
                        "raw_response": (r8.get("judge") or {}).get("raw_response"),
                        "video_duration": r8.get("video_duration"),
                        "fps_used": r8.get("fps_used"),
                        "n_frames": r8.get("n_frames"),
                    },
                }
            else:
                rec["stage2_recheck"] = None

            # ---- Stage 3: final choose (hard only) ----
            if f:
                rec["stage3_final_choose"] = {
                    "v7_letter": f.get("v7_letter"),
                    "v8_letter": f.get("v8_letter"),
                    "decision": f.get("decision"),
                    "final_letter": f.get("final_letter"),
                    "fps": f.get("fps"),
                    "arbiter_raw_response": (f.get("arbiter") or {}).get("raw_response"),
                }
            else:
                rec["stage3_final_choose"] = None

            # ---- where was this question resolved ----
            if not is_hard:
                rec["resolved_at"] = "stage1_consensus"
            elif (f or {}).get("decision") == "agree":
                rec["resolved_at"] = "stage2_agreement"
            else:
                rec["resolved_at"] = "stage3_arbitration"

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[merge] wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
