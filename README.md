# VRR-QA: A Disagreement-Gated Multi-Model Cascade for Implicit Video Relational Reasoning

This repository contains the reproduction code, data manifest, final submission,
and full reasoning traces for our entry to the **Implicit Video Relational
Reasoning (VRR-QA)** challenge (benchmark introduced in
[arXiv:2506.21742](https://arxiv.org/abs/2506.21742)).

The system is a **disagreement-gated cascade**: most questions are settled cheaply
by a consensus of three prompting strategies, and only the genuinely hard
questions are escalated to progressively more expensive, cross-model
verification stages.

---

## Pipeline overview

```
                 ┌─────────────────────────────────────────────┐
 test question → │ Stage 0: classify into 1 of 9 VRR categories │
                 └─────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────────────────────────────────────┐
        │ Stage 1 — three answer branches (Gemini, 1 FPS)            │
        │   • direct          • category-guided     • conservative   │
        └───────────────────────────────────────────────────────────┘
                                    │
                all 3 agree ───────────────────→ accept (consensus)
                                    │ disagree (hard)
        ┌───────────────────────────────────────────────────────────┐
        │ Stage 2 — cross-model recheck (adaptive FPS)               │
        │   • Gemini judge (native video)                            │
        │   • GPT judge     (frames)      ← different model family   │
        │   both audit the 3 branch rationales vs. the video         │
        └───────────────────────────────────────────────────────────┘
                                    │
                both agree ────────────────────→ accept
                                    │ disagree
        ┌───────────────────────────────────────────────────────────┐
        │ Stage 3 — final arbitration (Gemini 3.5 Flash, adaptive)   │
        │   watch the video, choose between the 2 judge answers      │
        └───────────────────────────────────────────────────────────┘
```

On the 172-question test set this routed **102** questions through Stage-1
consensus, **45** more were resolved when the two judges agreed in Stage 2, and
the remaining **25** were settled by Stage-3 arbitration.

**Why two different model families at Stage 2?** A single model that already
produced the Stage-1 disagreement is likely to repeat the same blind spots when
asked to recheck. Pairing the Gemini judge with a GPT judge makes the
verification decorrelated, so an error from one family can be caught by the
other.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── test_qa.json                 # the 172 official test questions (no answers)
├── test_qa_with_category.json   # test_qa.json + our Stage-0 category labels
├── submission.json              # our final answers ({question_id, answer_choice})
├── work.jsonl                   # full per-question reasoning trace (all stages)
├── src/
│   ├── config.py                # models, FPS tiers, paths, API keys (from env)
│   ├── api.py                   # public Gemini (google-genai) + OpenAI wrappers
│   ├── prompts.py               # every prompt + category guidelines
│   ├── classify.py              # Stage 0: rule-based + LLM-fallback classifier
│   ├── video_utils.py           # duration, adaptive FPS, frame extraction
│   ├── parsing.py               # answer parsing + anti-position-bias ordering
│   ├── stage1_branches.py       # Stage 1: direct / category / conservative
│   ├── stage2_recheck.py        # Stage 2: Gemini + GPT cross-model judges
│   ├── stage3_final_choose.py   # Stage 3: final-choose arbiter
│   └── run_pipeline.py          # end-to-end driver
└── scripts/
    ├── classify_qa.py           # Stage 0: produces test_qa_with_category.json
    └── merge_work.py            # merges per-stage logs into work.jsonl
```

---

## Setup

```bash
pip install -r requirements.txt
```

### API keys

All API calls use the **public** Gemini and OpenAI SDKs. No keys are bundled.
Export your own before running:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
# optional, only if you use an OpenAI-compatible gateway:
# export OPENAI_BASE_URL="https://your-endpoint/v1"
```

### Model names

Defaults live in `src/config.py` and can be overridden via environment
variables:

| Stage | Env var | Default |
|-------|---------|---------|
| Stage 1 branches | `GEMINI_BRANCH_MODEL` | `gemini-3.1-pro` |
| Stage 2 Gemini judge | `GEMINI_JUDGE_MODEL` | `gemini-3.1-pro` |
| Stage 2 GPT judge | `GPT_JUDGE_MODEL` | `gpt-5.5-pro` |
| Stage 3 arbiter | `GEMINI_ARBITER_MODEL` | `gemini-3.5-flash` |
| Classifier fallback | `CLASSIFIER_MODEL` | `gemini-3.1-pro` (originally Qwen3-4B) |

### Videos

The official test clips are **not** redistributed here. Download them from the
challenge and place one file per `video_id` under `data/videos/`:

```
data/videos/<video_id>.mp4
```

(`.mkv`, `.webm`, `.mov` are also accepted.) The `video_url` field in
`test_qa.json` points to each source clip.

---

## Running

Stage 0 only (regenerate the category labels):

```bash
python scripts/classify_qa.py --test-qa test_qa.json \
    --out test_qa_with_category.json --use-llm   # drop --use-llm for rule-based only
```

End-to-end (classification → branches → recheck → arbitration):

```bash
python -m src.run_pipeline \
    --test-qa test_qa.json \
    --video-dir data/videos \
    --out-dir outputs \
    --resume
```

This writes `outputs/submission.json` and `outputs/work.jsonl`. The GPT judge
caches extracted frames under `data/frame_cache/<video_id>/` to speed up reruns.

**FPS strategy.** The Stage-1 branches sample at a fixed **1 FPS**; the Stage-2
judges and Stage-3 arbiter use a denser **adaptive** schedule (`<8s → 8 FPS`,
`<20s → 2 FPS`, otherwise `1 FPS`), so short clips get more temporal detail
exactly where the verification happens.

---

## Files shipped with the repo

* **`test_qa.json`** — the 172 official test questions (`video_id`, `video_url`,
  `question_text`, `options`, timing).
* **`test_qa_with_category.json`** — the same questions plus our Stage-0
  `category_id` / `category`. Classification is part of our method, so we ship
  the labels; regenerate them with `python scripts/classify_qa.py`.
* **`submission.json`** — our final answers, one `{question_id, answer_choice}`
  per question (172, all filled).
* **`work.jsonl`** — one JSON object per question with the complete trace:
  the Stage-1 branch rationales, the two Stage-2 judge audits + answers, the
  Stage-3 arbitration, the `final_answer`, and a `resolved_at` field
  (`stage1_consensus` / `stage2_agreement` / `stage3_arbitration`).

`work.jsonl` is produced from the four per-stage logs by `scripts/merge_work.py`:

```bash
python scripts/merge_work.py \
    --ori   stage1_direct.jsonl \
    --v6    stage1_multibranch.jsonl \
    --v7    stage2_gemini.jsonl \
    --v8    stage2_gpt.jsonl \
    --final stage3_final_choose.jsonl \
    --submission submission.json \
    --test-qa test_qa.json \
    --out work.jsonl
```

The merge also strips machine-local fields so the trace is safe to publish.

---

## Notes

* The released `work.jsonl` records all **three** Stage-1 branches
  (`direct`, `category_guided`, `conservative`) for **every** question. The
  `direct` branch is the original Gemini direct run; `category_guided` /
  `conservative` come from the multi-branch log. For the escalated questions
  these are byte-identical to the rationales actually fed to the judges.
* All paths are repo-relative; nothing machine-specific is hard-coded.
* Generation is run at temperature 0 for determinism; per-question expert /
  judge orderings are fixed by a hash of the `question_id` to reduce position
  bias while staying reproducible.
