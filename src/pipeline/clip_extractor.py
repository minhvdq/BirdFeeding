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
    pre_s: float = 2.0,
    post_s: float = 8.0,
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


def build_negative_clips(
    events: list[FeedingEvent],
    video_dir: str,
    output_dir: str,
    clips_per_video: int = 4,
    min_gap_s: float = 20.0,
    clip_duration_s: float = 10.0,
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
        if duration < clip_duration_s + min_gap_s:
            continue

        attempts, found = 0, 0
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
