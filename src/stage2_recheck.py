# -*- coding: utf-8 -*-
"""Stage 2: cross-model evidence-grounded recheck (hard questions only).

Two independent judges re-watch the clip (adaptive FPS) and audit the three
Stage-1 expert rationales against the video before answering:

  * Gemini judge (same family as Stage 1)  -- native video input
  * GPT judge     (different family)        -- frame/image input

Using a *different* model family for the second judge avoids replaying the same
blind spots / hallucinations that produced the Stage-1 disagreement.
The two judges share the identical JUDGE_PROMPT and expert ordering.
"""

from typing import Dict

from . import config, prompts
from .parsing import options_block_and_letters, parse_letter, expert_order
from .video_utils import video_duration, adaptive_fps, extract_frames


def _build_judge_prompt(item, branches, order):
    block, letters, valid = options_block_and_letters(item)
    src = {
        "gem": (branches["direct"]["raw_response"] or "(no answer)").strip(),
        "b3":  (branches["category_guided"]["raw_response"] or "(no answer)").strip(),
        "b4":  (branches["conservative"]["raw_response"] or "(no answer)").strip(),
    }
    a, b, c = order
    prompt = prompts.JUDGE_PROMPT.format(
        question=item.get("question_text", "").strip(),
        options_block=block, letters=letters,
        expert_a_text=src[a], expert_b_text=src[b], expert_c_text=src[c],
    )
    return prompt, valid


def run_stage2(gem, gpt, item: dict, video_path: str, branches: Dict,
               frame_cache_dir: str = None) -> Dict:
    """Run both judges; return their answers + the FPS used."""
    qid = item.get("question_id")
    _, _, valid = options_block_and_letters(item)
    order = expert_order(qid)
    prompt, _ = _build_judge_prompt(item, branches, order)

    dur = video_duration(video_path)
    fps = adaptive_fps(dur)

    # Gemini judge -- native video.
    gem_raw = gem.generate_video(video_path, prompt, fps=fps,
                                 model=config.GEMINI_JUDGE_MODEL)
    gem_letter = parse_letter(gem_raw, valid)

    # GPT judge -- frames at the same FPS.
    frames = extract_frames(video_path, fps=fps, cache_dir=frame_cache_dir)
    gpt_raw = gpt.generate_frames(frames, prompt, model=config.GPT_JUDGE_MODEL)
    gpt_letter = parse_letter(gpt_raw, valid)

    return {
        "expert_order": order,
        "video_duration": dur,
        "fps_used": fps,
        "gemini_3_1_pro": {"pred_letter": gem_letter, "raw_response": gem_raw},
        "gpt_5_5_pro": {"pred_letter": gpt_letter, "raw_response": gpt_raw,
                        "n_frames": len(frames)},
        "agree": (gem_letter is not None and gem_letter == gpt_letter),
    }
