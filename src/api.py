# -*- coding: utf-8 -*-
"""Thin wrappers around the *public* Gemini and OpenAI SDKs.

This replaces the internal/company routing layer used during the competition
with standard, publicly documented API calls. Supply your own API keys via the
environment variables described in the README; nothing here is hard-coded.

  * GeminiClient  -- native video input with a controllable sampling FPS
                     (used by the Stage-1 branches, the Gemini judge, and the
                     Stage-3 arbiter), plus a plain text call for classification.
  * OpenAIClient  -- multi-image (frame) input for the GPT judge.
"""

import time
from typing import List, Optional

from . import config


# ===========================================================================
# Gemini (google-genai) -- native video, FPS-controllable
# ===========================================================================
class GeminiClient:
    """Wraps google-genai. Install with `pip install google-genai`."""

    def __init__(self, api_key: Optional[str] = None,
                 timeout: int = config.HTTP_TIMEOUT):
        from google import genai  # noqa: F401  (lazy import)
        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is empty. Export it before running (see README)."
            )
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.timeout = timeout

    def _upload_video(self, video_path: str):
        """Upload a local video file and wait until it is ACTIVE."""
        f = self.client.files.upload(file=video_path)
        # Files must finish server-side processing before use.
        while getattr(f.state, "name", str(f.state)) == "PROCESSING":
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        if getattr(f.state, "name", str(f.state)) == "FAILED":
            raise RuntimeError(f"Gemini file processing failed: {video_path}")
        return f

    def generate_video(self, video_path: str, prompt: str, fps: int,
                       model: Optional[str] = None,
                       temperature: float = config.TEMPERATURE,
                       max_output_tokens: int = config.MAX_OUTPUT_TOKENS) -> str:
        """Answer `prompt` while watching `video_path`, sampled at `fps`."""
        from google.genai import types
        model = model or config.GEMINI_BRANCH_MODEL
        f = self._upload_video(video_path)
        part = types.Part(
            file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
            video_metadata=types.VideoMetadata(fps=fps),
        )
        resp = self.client.models.generate_content(
            model=model,
            contents=types.Content(role="user", parts=[part, types.Part(text=prompt)]),
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return (resp.text or "").strip()

    def generate_text(self, prompt: str, system: Optional[str] = None,
                      model: Optional[str] = None,
                      temperature: float = 0.0) -> str:
        """Plain text completion (used by the question classifier fallback)."""
        from google.genai import types
        model = model or config.CLASSIFIER_MODEL
        cfg = types.GenerateContentConfig(temperature=temperature)
        if system:
            cfg.system_instruction = system
        resp = self.client.models.generate_content(
            model=model, contents=prompt, config=cfg,
        )
        return (resp.text or "").strip()


# ===========================================================================
# OpenAI -- GPT judge consumes a sequence of frames as images
# ===========================================================================
class OpenAIClient:
    """Wraps the openai SDK. Install with `pip install openai`."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = config.HTTP_TIMEOUT):
        from openai import OpenAI
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is empty. Export it before running (see README)."
            )
        kwargs = {"api_key": key, "timeout": timeout}
        base = base_url or config.OPENAI_BASE_URL
        if base:
            kwargs["base_url"] = base
        self.client = OpenAI(**kwargs)

    def generate_frames(self, frames_data_urls: List[str], prompt: str,
                        model: Optional[str] = None,
                        temperature: float = config.TEMPERATURE,
                        max_output_tokens: int = config.MAX_OUTPUT_TOKENS) -> str:
        """Answer `prompt` given a list of base64 data-URL frames."""
        model = model or config.GPT_JUDGE_MODEL
        content = [{"type": "text", "text": prompt}]
        for url in frames_data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
