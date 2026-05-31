# -*- coding: utf-8 -*-
"""End-to-end driver for the VRR-QA disagreement-gated cascade.

For each test question:
  0. classify into one of 9 categories (rule-based + LLM fallback)
  1. run the three Stage-1 branches (Gemini, 1 FPS)
        -> if all three agree, accept the consensus answer (done)
  2. otherwise run the two Stage-2 judges (Gemini + GPT, adaptive FPS)
        -> if both judges agree, accept that answer (done)
  3. otherwise run the Stage-3 arbiter (Gemini 3.5 Flash, adaptive FPS)

Writes:
  outputs/submission.json   -- {question_id, answer_choice} for every question
  outputs/work.jsonl        -- full per-question reasoning trace

Run:
  python -m src.run_pipeline --test-qa test_qa.json --video-dir data/videos
"""

import os
import json
import argparse
from typing import Dict, Optional

from . import config
from .classify import classify
from .stage1_branches import run_stage1
from .stage2_recheck import run_stage2
from .stage3_final_choose import run_stage3
from .prompts import CATEGORY_NAMES


def _video_path(video_dir: str, item: dict) -> Optional[str]:
    vid = item.get("video_id")
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        p = os.path.join(video_dir, f"{vid}{ext}")
        if os.path.exists(p):
            return p
    return None


def _load_done(path: str) -> Dict[str, dict]:
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["question_id"]] = r
    return done


def process_one(gem, gpt, item, video_path) -> dict:
    qid = item.get("question_id")
    cid = classify(item.get("question_text", ""), item.get("options", {}), gem)

    rec = {
        "question_id": qid,
        "video_id": item.get("video_id"),
        "video_url": item.get("video_url"),
        "question_text": item.get("question_text"),
        "options": item.get("options"),
        "category_id": cid,
        "category": CATEGORY_NAMES.get(cid),
        "stage2_recheck": None,
        "stage3_final_choose": None,
    }

    # Stage 1
    s1 = run_stage1(gem, item, video_path, cid)
    rec["stage1"] = {"branches": s1["branches"], "all_agree": s1["all_agree"]}
    if s1["all_agree"]:
        rec["final_answer"] = s1["consensus_letter"]
        rec["resolved_at"] = "stage1_consensus"
        return rec

    # Stage 2
    cache = os.path.join(config.FRAME_CACHE_DIR, str(item.get("video_id")))
    s2 = run_stage2(gem, gpt, item, video_path, s1["branches"], frame_cache_dir=cache)
    rec["stage2_recheck"] = s2
    if s2["agree"]:
        rec["final_answer"] = s2["gemini_3_1_pro"]["pred_letter"]
        rec["resolved_at"] = "stage2_agreement"
        return rec

    # Stage 3
    s3 = run_stage3(gem, item, video_path, s2)
    rec["stage3_final_choose"] = s3
    rec["final_answer"] = s3["final_letter"]
    rec["resolved_at"] = "stage3_arbitration"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-qa", default=config.TEST_QA_PATH)
    ap.add_argument("--video-dir", default=config.VIDEO_DIR)
    ap.add_argument("--out-dir", default=config.OUTPUT_DIR)
    ap.add_argument("--resume", action="store_true",
                    help="skip questions already present in work.jsonl")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work_path = os.path.join(args.out_dir, "work.jsonl")
    sub_path = os.path.join(args.out_dir, "submission.json")

    from .api import GeminiClient, OpenAIClient
    gem = GeminiClient()
    gpt = OpenAIClient()

    with open(args.test_qa, "r", encoding="utf-8") as f:
        items = json.load(f)

    done = _load_done(work_path) if args.resume else {}
    mode = "a" if (args.resume and done) else "w"

    with open(work_path, mode, encoding="utf-8") as wf:
        for item in items:
            qid = item.get("question_id")
            if qid in done:
                continue
            vp = _video_path(args.video_dir, item)
            if vp is None:
                print(f"[skip] no video for {item.get('video_id')} (qid={qid})")
                continue
            try:
                rec = process_one(gem, gpt, item, vp)
            except Exception as e:  # keep going on per-item failure
                print(f"[error] qid={qid}: {e}")
                continue
            done[qid] = rec
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wf.flush()
            print(f"[ok] {qid} -> {rec.get('final_answer')} ({rec.get('resolved_at')})")

    submission = [{"question_id": q, "answer_choice": r.get("final_answer", "")}
                  for q, r in done.items()]
    with open(sub_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(submission)} answers -> {sub_path}")


if __name__ == "__main__":
    main()
