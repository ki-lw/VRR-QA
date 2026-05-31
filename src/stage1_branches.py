# -*- coding: utf-8 -*-
"""Stage 1: multi-branch answer generation (Gemini, fixed 1 FPS).

Three independent answer branches watch the same clip:
  * direct        -- plain QA, no guidance
  * category_guided -- inject the micro-guideline matching the question's category
  * conservative  -- anti-hallucination, only answer what is directly supported

If all three branches agree, the answer is accepted as high-confidence
(consensus). Otherwise the question is "hard" and escalated to Stage 2.
"""

from typing import Dict

from . import config, prompts
from .parsing import options_block_and_letters, parse_letter


def _run_branch(gem, item, prompt_text, video_path):
    block, letters, valid = options_block_and_letters(item)
    raw = gem.generate_video(
        video_path,
        prompt_text.format(question=item.get("question_text", "").strip(),
                            options_block=block, letters=letters,
                            guideline_block=item.get("_guideline_block", "")),
        fps=config.BRANCH_FPS,
        model=config.GEMINI_BRANCH_MODEL,
    )
    return {"raw_response": raw, "pred_letter": parse_letter(raw, valid)}


def run_stage1(gem, item: dict, video_path: str, category_id: int) -> Dict:
    """Return the three branch outputs + a consensus flag."""
    guideline = prompts.CATEGORY_GUIDELINES.get(category_id, "")
    item = dict(item)
    item["_guideline_block"] = (guideline + "\n\n") if guideline else ""

    direct = _run_branch(gem, item, prompts.DIRECT_PROMPT, video_path)
    category = _run_branch(gem, item, prompts.CATEGORY_PROMPT, video_path)
    conservative = _run_branch(gem, item, prompts.CONSERVATIVE_PROMPT, video_path)

    letters = {direct["pred_letter"], category["pred_letter"],
               conservative["pred_letter"]}
    all_agree = (len(letters) == 1 and None not in letters)

    return {
        "branches": {
            "direct": direct,             # 'gem'
            "category_guided": category,  # 'b3'
            "conservative": conservative,  # 'b4'
        },
        "all_agree": all_agree,
        "consensus_letter": direct["pred_letter"] if all_agree else None,
    }
