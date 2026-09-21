from __future__ import annotations
import cv2
import numpy as np


def sample_frames(clip_path: str, sample_fps: float = 2.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(clip_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(src_fps / sample_fps))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def letterbox_resize(
    frame: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize full frame to target_size with black letterbox padding."""
    h, w = frame.shape[:2]
    th, tw = target_size
    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    pad_y = (th - nh) // 2
    pad_x = (tw - nw) // 2
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out


def process_clip(
    clip_path: str,
    sample_fps: float = 2.0,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    raw_frames = sample_frames(clip_path, sample_fps)
    if not raw_frames:
        return np.zeros((0, *target_size, 3), dtype=np.uint8)
    processed = [letterbox_resize(f, target_size=target_size) for f in raw_frames]
    return np.stack(processed, axis=0)  # (N, 224, 224, 3)
