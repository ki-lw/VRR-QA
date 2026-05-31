# -*- coding: utf-8 -*-
"""Video helpers: duration, adaptive FPS, and frame extraction for the GPT judge."""

import os
import math
import json
import base64
from typing import List, Optional

from . import config


def video_duration(video_path: str) -> float:
    """Return the clip duration in seconds (0.0 on failure)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n / fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def adaptive_fps(duration_s: float) -> int:
    """Denser sampling on short clips; see config.FPS_TIERS."""
    if not duration_s or duration_s <= 0:
        return config.FPS_DEFAULT
    for thr, fps in config.FPS_TIERS:
        if duration_s < thr:
            return fps
    return config.FPS_DEFAULT


def _uniform_indices(total_frames: int, desired: int) -> List[int]:
    """Pick `desired` evenly-spaced frame indices (segment midpoints)."""
    if desired <= 0 or total_frames <= 0:
        return []
    if desired >= total_frames:
        return list(range(total_frames))
    step = total_frames / desired
    return [min(total_frames - 1, int((i + 0.5) * step)) for i in range(desired)]


def extract_frames(video_path: str, fps: int,
                   max_frames: int = config.MAX_FRAMES,
                   max_size: int = config.FRAME_MAX_SIZE,
                   cache_dir: Optional[str] = None) -> List[str]:
    """Sample frames at `fps`, resize, JPEG-encode and return base64 data URLs.

    Used only by the GPT judge, which takes images rather than native video.
    If `cache_dir` is given, decoded frames + a meta.json are written there and
    reused on subsequent runs with the same parameters.
    """
    import cv2

    if cache_dir:
        cached = _load_cache(cache_dir, fps, max_frames, max_size)
        if cached is not None:
            return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / src_fps if src_fps > 0 else 0.0

    desired = min(max_frames, max(1, math.floor(duration * fps)))
    want = set(_uniform_indices(total, desired))
    max_want = max(want) if want else -1

    data_urls: List[str] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            h, w = frame.shape[:2]
            scale = max_size / float(max(h, w)) if max(h, w) > max_size else 1.0
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.FRAME_JPEG_QUALITY])
            if ok2:
                b64 = base64.b64encode(buf).decode("utf-8")
                data_urls.append(f"data:image/jpeg;base64,{b64}")
        idx += 1
        if idx > max_want:
            break
    cap.release()

    if cache_dir:
        _save_cache(cache_dir, fps, max_frames, max_size, data_urls)
    return data_urls


def _meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "meta.json")


def _load_cache(cache_dir: str, fps: int, max_frames: int, max_size: int):
    try:
        with open(_meta_path(cache_dir), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (meta.get("fps") == fps and meta.get("max_frames") == max_frames
                and meta.get("max_size") == max_size):
            return meta.get("frames")
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _save_cache(cache_dir: str, fps: int, max_frames: int, max_size: int,
                data_urls: List[str]):
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(_meta_path(cache_dir), "w", encoding="utf-8") as f:
            json.dump({"fps": fps, "max_frames": max_frames,
                       "max_size": max_size, "n_frames": len(data_urls),
                       "frames": data_urls}, f)
    except OSError:
        pass
