from __future__ import annotations
import os
import random
import subprocess
from collections import defaultdict
from src.pipeline.csv_parser import FeedingEvent


def extract_clip(video_source: str, start_s: float, end_s: float, output_path: str) -> None:
    """Extract [start_s, end_s] from video_source into output_path via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-to", str(end_s),
        "-i", video_source,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")


def _find_video(video_dir: str, video_id: str) -> str:
    for ext in (".mp4", ".MP4", ".mov", ".MOV"):
        p = os.path.join(video_dir, video_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No video found for {video_id} in {video_dir}")


def _video_duration(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def build_positive_clips(
    events: list[FeedingEvent],
    video_dir: str,
    output_dir: str,
    pre_s: float = 1.0,
    post_s: float = 4.0,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, event in enumerate(events):
        video_path = _find_video(video_dir, event.video_id)
        start = max(0.0, event.timestamp_s - pre_s)
        end = event.timestamp_s + post_s
        out = os.path.join(output_dir, f"{event.video_id}_feed_{i:04d}.mp4")
        extract_clip(video_path, start, end, out)
        paths.append(out)
    return paths


def _boundary_candidates(
    feeding_timestamps: list[float],
    duration: float,
    clip_duration_s: float,
    pre_s: float,
    post_s: float,
) -> list[tuple[float, float]]:
    """Windows just outside each feeding event, filtered to avoid other events."""
    candidates = []
    for ts in feeding_timestamps:
        after_start = ts + post_s
        if after_start + clip_duration_s <= duration:
            candidates.append((after_start, after_start + clip_duration_s))
        before_end = ts - pre_s
        if before_end - clip_duration_s >= 0:
            candidates.append((before_end - clip_duration_s, before_end))
    # Drop any window whose midpoint lands inside another feeding window
    safe = []
    for start, end in candidates:
        mid = (start + end) / 2
        if not any(abs(mid - ft) < clip_duration_s / 2.0 for ft in feeding_timestamps):
            safe.append((start, end))
    return safe


def build_negative_clips(
    events: list[FeedingEvent],
    video_dir: str,
    output_dir: str,
    clips_per_video: int = 4,
    min_gap_s: float = 20.0,
    clip_duration_s: float = 5.0,
    pre_s: float = 1.0,
    post_s: float = 4.0,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    by_video: dict[str, list[float]] = defaultdict(list)
    for e in events:
        by_video[e.video_id].append(e.timestamp_s)

    paths = []
    rng = random.Random(42)
    for video_id, feeding_timestamps in by_video.items():
        try:
            video_path = _find_video(video_dir, video_id)
        except FileNotFoundError:
            continue
        duration = _video_duration(video_path)
        if duration < clip_duration_s:
            continue

        n_boundary = clips_per_video // 2
        n_random = clips_per_video - n_boundary
        found = 0

        # Boundary negatives (hard examples adjacent to feeding windows)
        candidates = _boundary_candidates(feeding_timestamps, duration, clip_duration_s, pre_s, post_s)
        rng.shuffle(candidates)
        for start, end in candidates[:n_boundary]:
            out = os.path.join(output_dir, f"{video_id}_normal_{found:04d}.mp4")
            extract_clip(video_path, start, end, out)
            paths.append(out)
            found += 1

        # Random negatives (easy background examples ≥ min_gap_s from any event)
        attempts = 0
        while found < clips_per_video and attempts < 200:
            attempts += 1
            start = rng.uniform(0, duration - clip_duration_s)
            mid = start + clip_duration_s / 2
            if any(abs(mid - ft) < min_gap_s for ft in feeding_timestamps):
                continue
            out = os.path.join(output_dir, f"{video_id}_normal_{found:04d}.mp4")
            extract_clip(video_path, start, start + clip_duration_s, out)
            paths.append(out)
            found += 1
    return paths
