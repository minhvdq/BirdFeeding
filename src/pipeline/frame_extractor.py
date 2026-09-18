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


def motion_crop(
    frame: np.ndarray,
    reference: np.ndarray,
    padding: int = 30,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    diff = cv2.absdiff(frame, reference)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    if contours:
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + bw + padding), min(h, y + bh + padding)
        crop = frame[y1:y2, x1:x2]
    else:
        # No motion: return center crop
        size = min(h, w)
        cy, cx = h // 2, w // 2
        half = size // 2
        crop = frame[max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]

    if crop.size == 0:
        crop = frame

    return cv2.resize(crop, target_size)


def process_clip(
    clip_path: str,
    sample_fps: float = 2.0,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    raw_frames = sample_frames(clip_path, sample_fps)
    if not raw_frames:
        return np.zeros((0, *target_size, 3), dtype=np.uint8)

    reference = raw_frames[0]
    processed = [motion_crop(f, reference, target_size=target_size) for f in raw_frames]
    return np.stack(processed, axis=0)  # (N, 224, 224, 3)
