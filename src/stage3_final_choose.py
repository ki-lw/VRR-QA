# -*- coding: utf-8 -*-
"""Stage 3: final arbitration (Gemini 3.5 Flash) for residual disagreements.

When the two Stage-2 judges still disagree, a third model watches the clip
itself (adaptive FPS, native video) and is forced to choose between exactly the
two conflicting candidate answers, grounded in the judges' rationales.
"""

from typing import Dict

from . import config, prompts
from .parsing import (options_block_and_letters, parse_letter,
                      extract_reasoning, arbiter_order)
from .video_utils import video_duration, adaptive_fps


def run_stage3(gem, item: dict, video_path: str, stage2: Dict) -> Dict:
    """Arbitrate between the Gemini and GPT judge answers."""
    qid = item.get("question_id")
    block, _letters, _valid = options_block_and_letters(item)

    judges = {
        "gemini": {"letter": stage2["gemini_3_1_pro"]["pred_letter"],
                   "reason": extract_reasoning(stage2["gemini_3_1_pro"]["raw_response"])},
        "gpt": {"letter": stage2["gpt_5_5_pro"]["pred_letter"],
                "reason": extract_reasoning(stage2["gpt_5_5_pro"]["raw_response"])},
    }
    order = arbiter_order(qid)
    a, b = order
    cand1, cand2 = judges[a]["letter"], judges[b]["letter"]

    prompt = prompts.FINAL_CHOOSE_PROMPT.format(
        question=item.get("question_text", "").strip(),
        options_block=block,
        cand1=cand1 or "?", cand2=cand2 or "?",
        ans1=cand1 or "?", reason1=judges[a]["reason"],
        ans2=cand2 or "?", reason2=judges[b]["reason"],
    )

    dur = video_duration(video_path)
    fps = adaptive_fps(dur)
    raw = gem.generate_video(video_path, prompt, fps=fps,
                             model=config.GEMINI_ARBITER_MODEL)

    # The arbiter must pick one of the two candidates.
    valid_two = [c for c in (cand1, cand2) if c]
    final_letter = parse_letter(raw, valid_two) or cand1 or cand2

    return {
        "arbiter_order": order,
        "fps": fps,
        "gemini_letter": judges["gemini"]["letter"],
        "gpt_letter": judges["gpt"]["letter"],
        "final_letter": final_letter,
        "arbiter_raw_response": raw,
    }
