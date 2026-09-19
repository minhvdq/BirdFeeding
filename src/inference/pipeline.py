from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from src.pipeline.frame_extractor import motion_crop

_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class FeedingDetection:
    start_s: float
    end_s: float
    confidence: float


def _scan_motion(video_path: str, scan_fps: float = 1.0, min_area: int = 500) -> list[float]:
    """Return timestamps (seconds) where significant motion is detected."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(src_fps / scan_fps))
    motion_timestamps, idx, prev = [], 0, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ts = idx / src_fps
            if prev is not None:
                diff = cv2.absdiff(frame, prev)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                if cv2.countNonZero(mask) >= min_area:
                    motion_timestamps.append(ts)
            prev = frame.copy()
        idx += 1
    cap.release()
    return motion_timestamps


def _cluster_timestamps(timestamps: list[float], gap_s: float = 5.0) -> list[float]:
    """Merge nearby motion timestamps into single candidate midpoints."""
    if not timestamps:
        return []
    clusters, current = [], [timestamps[0]]
    for t in timestamps[1:]:
        if t - current[-1] <= gap_s:
            current.append(t)
        else:
            clusters.append(sum(current) / len(current))
            current = [t]
    clusters.append(sum(current) / len(current))
    return clusters


def _extract_clip_frames(
    video_path: str, center_s: float, clip_duration_s: float = 10.0, sample_fps: float = 2.0
) -> np.ndarray:
    """Extract and preprocess 20 frames around center_s. Returns (T, 3, 224, 224) float32."""
    start_s = max(0.0, center_s - clip_duration_s / 2)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src_fps))
    interval = max(1, round(src_fps / sample_fps))
    n_frames = int(clip_duration_s * sample_fps)
    frames, idx, reference = [], 0, None
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if reference is None:
            reference = frame.copy()
        if idx % interval == 0:
            crop = motion_crop(frame, reference, target_size=(224, 224))
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            frames.append(_IMAGENET_TRANSFORM(pil).numpy())
        idx += 1
    cap.release()
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((3, 224, 224), dtype=np.float32))
    return np.stack(frames[:n_frames], axis=0).astype(np.float32)  # (T, 3, 224, 224)


class FeedingDetector:
    def __init__(self, feature_model_path: str, classifier_model_path: str):
        self._feat_sess = ort.InferenceSession(feature_model_path)
        self._cls_sess = ort.InferenceSession(classifier_model_path)
        self._feat_input = self._feat_sess.get_inputs()[0].name
        self._cls_input = self._cls_sess.get_inputs()[0].name

    def _embed_clip(self, frames: np.ndarray) -> np.ndarray:
        """frames: (T, 3, 224, 224) -> features: (1, T, 1280)"""
        feats = self._feat_sess.run(None, {self._feat_input: frames})[0]  # (T, 1280)
        return feats[np.newaxis]  # (1, T, 1280)

    def _classify(self, features: np.ndarray) -> float:
        """features: (1, T, 1280) -> probability float"""
        return float(self._cls_sess.run(None, {self._cls_input: features})[0][0][0])

    def detect(
        self,
        video_path: str,
        threshold: float = 0.6,
        progress_cb: Callable[[float], None] | None = None,
    ) -> list[FeedingDetection]:
        motion_ts = _scan_motion(video_path)
        candidates = _cluster_timestamps(motion_ts)
        if not candidates:
            return []

        detections: list[FeedingDetection] = []
        for i, center in enumerate(candidates):
            if progress_cb:
                progress_cb((i + 1) / len(candidates))
            frames = _extract_clip_frames(video_path, center)
            features = self._embed_clip(frames)
            prob = self._classify(features)
            if prob >= threshold:
                half = 5.0
                detections.append(FeedingDetection(
                    start_s=max(0.0, center - half),
                    end_s=center + half,
                    confidence=round(prob, 3),
                ))

        # Merge adjacent detections within 3s gap, keeping max confidence
        if not detections:
            return []
        detections.sort(key=lambda d: d.start_s)
        merged = [detections[0]]
        for d in detections[1:]:
            prev = merged[-1]
            if d.start_s <= prev.end_s + 3.0:
                merged[-1] = FeedingDetection(
                    start_s=prev.start_s,
                    end_s=max(prev.end_s, d.end_s),
                    confidence=max(prev.confidence, d.confidence),
                )
            else:
                merged.append(d)
        return merged
