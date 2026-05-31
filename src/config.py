# -*- coding: utf-8 -*-
"""Central configuration for the VRR-QA reproduction pipeline.

All API keys are read from environment variables and default to empty strings.
Fill in your own keys (see README) before running anything that hits an API.
"""

import os

# ---------------------------------------------------------------------------
# API credentials (fill these via environment variables -- see README)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Optional: point OpenAI-compatible clients at a custom endpoint.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# ---------------------------------------------------------------------------
# Models used at each stage of the pipeline (override via env if you like)
# ---------------------------------------------------------------------------
# Stage 1: three answer branches (direct / category-guided / conservative)
GEMINI_BRANCH_MODEL = os.environ.get("GEMINI_BRANCH_MODEL", "gemini-3.1-pro")
# Stage 2: cross-model recheck -- two independent judges
GEMINI_JUDGE_MODEL  = os.environ.get("GEMINI_JUDGE_MODEL",  "gemini-3.1-pro")
GPT_JUDGE_MODEL     = os.environ.get("GPT_JUDGE_MODEL",     "gpt-5.5-pro")
# Stage 3: final arbitration between the two judges
GEMINI_ARBITER_MODEL = os.environ.get("GEMINI_ARBITER_MODEL", "gemini-3.5-flash")
# Optional LLM fallback for the question classifier (Stage 0).
# Originally Qwen3-4B; any chat model works. Empty -> rule-based only.
CLASSIFIER_MODEL = os.environ.get("CLASSIFIER_MODEL", "gemini-3.1-pro")

# ---------------------------------------------------------------------------
# Adaptive FPS strategy
#   Stage 1 (three branches)        : fixed 1 FPS  (BRANCH_FPS)
#   Stage 2/3 (recheck + arbitrate) : adaptive, denser FPS on short clips
# ---------------------------------------------------------------------------
BRANCH_FPS = 1
# (duration_threshold_seconds, fps) -- first matching tier wins.
FPS_TIERS = [
    (8.0, 8),    # clips  < 8s  -> 8 FPS
    (20.0, 2),   # clips  < 20s -> 2 FPS
]
FPS_DEFAULT = 1  # clips >= 20s -> 1 FPS

# Frame extraction (only needed for the GPT judge, which consumes images).
MAX_FRAMES = 256
FRAME_MAX_SIZE = 512          # resize so the long edge <= this many px
FRAME_JPEG_QUALITY = 90

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4096
HTTP_TIMEOUT = 180

# ---------------------------------------------------------------------------
# Default paths (all relative to the repo root -- no machine-specific paths)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_QA_PATH = os.path.join(REPO_ROOT, "test_qa.json")
# Put the official test clips here (not shipped in the repo). One file per
# video_id, e.g. data/videos/<video_id>.mp4  -- see README.
VIDEO_DIR = os.environ.get("VRR_VIDEO_DIR", os.path.join(REPO_ROOT, "data", "videos"))
FRAME_CACHE_DIR = os.environ.get("VRR_FRAME_CACHE", os.path.join(REPO_ROOT, "data", "frame_cache"))
OUTPUT_DIR = os.environ.get("VRR_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
