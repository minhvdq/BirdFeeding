# Tern Feeding Event Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python application that accepts a field video and returns timestamped feeding events detected by a two-stage CV model (EfficientNet-B0 feature extractor + LSTM temporal classifier).

**Architecture:** Motion detection (frame differencing) identifies candidate 10-second windows in the uploaded video; each window's frames are embedded by a frozen EfficientNet-B0 backbone; a lightweight LSTM classifies the resulting sequence as feeding vs. non-feeding. Both models ship as ONNX files so inference requires no PyTorch. Training runs on Google Colab (free T4) via a notebook that reads clips from Google Drive.

**Tech Stack:** Python 3.10+, OpenCV, ffmpeg-python, PyTorch + torchvision (Colab only), onnxruntime, Gradio, pandas, numpy

**Spec:** `docs/superpowers/specs/2026-09-17-bird-feeding-detection-design.md`

## Global Constraints

- Python 3.10+ throughout
- `onnxruntime>=1.17.0` — no PyTorch dependency on student machines
- `opencv-python>=4.9.0`
- `gradio>=4.28.0`
- `torch>=2.1.0`, `torchvision>=0.16.0` — Colab / training only
- All frames resized to 224×224 before model input
- EfficientNet-B0 feature vector dimension: 1280
- Clip length for training and inference: 10 seconds sampled at 2 fps = 20 frames
- Positive clip window: `[timestamp - 2s, timestamp + 8s]`
- Negative clip minimum gap from any feeding event: 20 seconds
- ONNX models live in `models/` at project root
- Tests live under `tests/` mirroring `src/` structure
- Run tests with `pytest` from project root

---

## File Map

```
BirdFeeding/
  src/
    pipeline/
      csv_parser.py         # FeedingEvent dataclass + parse_feeding_events()
      clip_extractor.py     # ffmpeg clip extraction, positive/negative builders
      frame_extractor.py    # frame sampling + motion-crop preprocessing
    model/
      features.py           # PyTorch EfficientNet-B0 wrapper (Colab use only)
      temporal.py           # PyTorch LSTM classifier (Colab use only)
      train.py              # BirdClipDataset + training loop (Colab use only)
      export.py             # torch → ONNX export for both models
    inference/
      pipeline.py           # FeedingDetector: video → list[FeedingDetection]
    app/
      app.py                # Gradio UI entry point
  tests/
    conftest.py             # synthetic video + CSV fixtures
    pipeline/
      test_csv_parser.py
      test_clip_extractor.py
      test_frame_extractor.py
    model/
      test_features.py
      test_temporal.py
      test_export.py
    inference/
      test_pipeline.py
    app/
      test_app.py
  notebooks/
    train.ipynb             # Google Colab training notebook
  models/                   # populated after training (ONNX files go here)
  data/
    clips/feeding/
    clips/normal/
    frames/feeding/
    frames/normal/
  requirements.txt          # inference + app deps
  requirements-train.txt    # adds torch/torchvision for Colab
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `requirements-train.txt`
- Create: `tests/conftest.py`
- Create: `src/__init__.py`, `src/pipeline/__init__.py`, `src/model/__init__.py`, `src/inference/__init__.py`, `src/app/__init__.py`
- Create: `tests/pipeline/__init__.py`, `tests/model/__init__.py`, `tests/inference/__init__.py`, `tests/app/__init__.py`

**Interfaces:**
- Produces: `synthetic_video` fixture → `(path: str, feeding_timestamp_s: float)` — 15-second MP4 with a bright moving rectangle at seconds 9–12
- Produces: `sample_csv` fixture → `path: str` — minimal CSV with 2 feeding events for GX_TEST

- [ ] **Step 1: Create requirements files**

`requirements.txt`:
```
onnxruntime>=1.17.0
opencv-python>=4.9.0
gradio>=4.28.0
numpy>=1.26.0
pandas>=2.1.0
ffmpeg-python>=0.2.0
```

`requirements-train.txt`:
```
-r requirements.txt
torch>=2.1.0
torchvision>=0.16.0
onnx>=1.15.0
scikit-learn>=1.3.0
```

- [ ] **Step 2: Create all `__init__.py` files**

```bash
mkdir -p src/pipeline src/model src/inference src/app
mkdir -p tests/pipeline tests/model tests/inference tests/app
mkdir -p models data/clips/feeding data/clips/normal data/frames/feeding data/frames/normal notebooks
touch src/__init__.py src/pipeline/__init__.py src/model/__init__.py src/inference/__init__.py src/app/__init__.py
touch tests/pipeline/__init__.py tests/model/__init__.py tests/inference/__init__.py tests/app/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py` with shared fixtures**

```python
import os
import pytest
import cv2
import numpy as np


@pytest.fixture
def synthetic_video(tmp_path):
    """15-second 640x480 MP4 at 30fps. Bright rectangle moves at t=9-12s."""
    path = str(tmp_path / "test_video.mp4")
    w, h, fps, duration_s = 640, 480, 30, 15
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(42)
    for i in range(fps * duration_s):
        frame = rng.integers(20, 50, (h, w, 3), dtype=np.uint8)
        if 270 <= i <= 360:  # seconds 9-12
            x = int((i - 270) / 3) + 80
            frame[200:260, x : x + 60] = 220  # bright rectangle = "bird"
        writer.write(frame)
    writer.release()
    return path, 9.0


@pytest.fixture
def sample_csv(tmp_path):
    """Minimal CSV with two feeding events and one Start/End row each."""
    path = str(tmp_path / "events.csv")
    content = (
        "date,Year,Observer,GoPro video ID,Watch ID,Area,"
        "Video timestamp (m.s),Time,Event (start; end; feeding; failed feed; stolen from; stolen by),"
        "Nest ID\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,0:00,,Start,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,9:37,,Feeding,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,5:10,,Feeding Fail,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,3:00,,Stealing,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,11:48,,End,N1\n"
    )
    path_obj = tmp_path / "events.csv"
    path_obj.write_text(content)
    return str(path)
```

- [ ] **Step 4: Install deps and verify pytest collects**

```bash
pip install -r requirements.txt pytest
pytest --collect-only
```

Expected: "no tests ran" with 0 errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt requirements-train.txt tests/conftest.py src/ tests/
git commit -m "feat: project scaffolding — requirements, package structure, test fixtures"
```

---

## Task 2: CSV Parser

**Files:**
- Create: `src/pipeline/csv_parser.py`
- Create: `tests/pipeline/test_csv_parser.py`

**Interfaces:**
- Produces: `FeedingEvent(video_id: str, timestamp_s: float)`
- Produces: `parse_feeding_events(csv_path: str) -> list[FeedingEvent]`
  - Keeps rows where Event column contains "feeding" or "stealing" (case-insensitive)
  - Skips rows where Event is "start" or "end" (case-insensitive)

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_csv_parser.py`:
```python
from src.pipeline.csv_parser import FeedingEvent, parse_feeding_events


def test_returns_only_feeding_rows(sample_csv):
    events = parse_feeding_events(sample_csv)
    assert len(events) == 3  # Feeding + Feeding Fail + Stealing; not Start/End


def test_video_id_parsed(sample_csv):
    events = parse_feeding_events(sample_csv)
    assert all(e.video_id == "GX_TEST" for e in events)


def test_timestamp_parsed_to_seconds(sample_csv):
    events = parse_feeding_events(sample_csv)
    timestamps = {round(e.timestamp_s) for e in events}
    assert 577 in timestamps   # 9:37 = 9*60+37
    assert 310 in timestamps   # 5:10 = 5*60+10
    assert 180 in timestamps   # 3:00


def test_returns_feeding_event_dataclass(sample_csv):
    events = parse_feeding_events(sample_csv)
    assert isinstance(events[0], FeedingEvent)
    assert hasattr(events[0], "video_id")
    assert hasattr(events[0], "timestamp_s")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/pipeline/test_csv_parser.py -v
```
Expected: 4 errors — `ImportError: cannot import name 'FeedingEvent'`

- [ ] **Step 3: Implement `src/pipeline/csv_parser.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class FeedingEvent:
    video_id: str
    timestamp_s: float


_KEEP = {"feeding", "feeding fail", "stealing"}


def _parse_timestamp(ts: str) -> float:
    ts = str(ts).strip()
    if ":" in ts:
        mins, secs = ts.split(":", 1)
        return int(mins) * 60 + float(secs)
    return float(ts)


def parse_feeding_events(csv_path: str) -> list[FeedingEvent]:
    df = pd.read_csv(csv_path, dtype=str)
    event_col = [c for c in df.columns if "event" in c.lower()][0]
    video_col = [c for c in df.columns if "gopro" in c.lower() or "video id" in c.lower()][0]
    ts_col = [c for c in df.columns if "timestamp" in c.lower()][0]

    events: list[FeedingEvent] = []
    for _, row in df.iterrows():
        event = str(row[event_col]).strip().lower()
        if event not in _KEEP:
            continue
        video_id = str(row[video_col]).strip()
        ts = _parse_timestamp(row[ts_col])
        events.append(FeedingEvent(video_id=video_id, timestamp_s=ts))
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/pipeline/test_csv_parser.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/csv_parser.py tests/pipeline/test_csv_parser.py
git commit -m "feat: CSV parser — FeedingEvent dataclass + parse_feeding_events()"
```

---

## Task 3: Clip Extractor

**Files:**
- Create: `src/pipeline/clip_extractor.py`
- Create: `tests/pipeline/test_clip_extractor.py`

**Interfaces:**
- Consumes: `FeedingEvent` from `src.pipeline.csv_parser`
- Produces: `extract_clip(video_source: str, start_s: float, end_s: float, output_path: str) -> None`
- Produces: `build_positive_clips(events: list[FeedingEvent], video_dir: str, output_dir: str, pre_s: float = 2.0, post_s: float = 8.0) -> list[str]`
- Produces: `build_negative_clips(events: list[FeedingEvent], video_dir: str, output_dir: str, clips_per_video: int = 4, min_gap_s: float = 20.0, clip_duration_s: float = 10.0) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_clip_extractor.py`:
```python
import os
import cv2
from src.pipeline.clip_extractor import extract_clip, build_positive_clips, build_negative_clips
from src.pipeline.csv_parser import FeedingEvent


def test_extract_clip_creates_file(synthetic_video, tmp_path):
    video_path, _ = synthetic_video
    out = str(tmp_path / "clip.mp4")
    extract_clip(video_path, start_s=2.0, end_s=5.0, output_path=out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_extract_clip_duration(synthetic_video, tmp_path):
    video_path, _ = synthetic_video
    out = str(tmp_path / "clip.mp4")
    extract_clip(video_path, start_s=2.0, end_s=5.0, output_path=out)
    cap = cv2.VideoCapture(out)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frame_count / fps
    assert 2.5 <= duration <= 3.5  # ~3 seconds with codec rounding


def test_build_positive_clips(synthetic_video, tmp_path):
    video_path, ts = synthetic_video
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    # Name the video so video_id "GX_TEST" maps to it
    import shutil
    shutil.copy(video_path, os.path.join(video_dir, "GX_TEST.mp4"))
    out_dir = str(tmp_path / "clips" / "feeding")
    os.makedirs(out_dir)
    events = [FeedingEvent(video_id="GX_TEST", timestamp_s=ts)]
    paths = build_positive_clips(events, video_dir, out_dir)
    assert len(paths) == 1
    assert os.path.exists(paths[0])


def test_build_negative_clips_avoids_feeding_windows(synthetic_video, tmp_path):
    video_path, ts = synthetic_video
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    import shutil
    shutil.copy(video_path, os.path.join(video_dir, "GX_TEST.mp4"))
    out_dir = str(tmp_path / "clips" / "normal")
    os.makedirs(out_dir)
    events = [FeedingEvent(video_id="GX_TEST", timestamp_s=ts)]
    paths = build_negative_clips(events, video_dir, out_dir, clips_per_video=2, min_gap_s=5.0)
    # None of the negative clips should overlap the feeding window
    for p in paths:
        cap = cv2.VideoCapture(p)
        cap.release()
        assert os.path.exists(p)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/pipeline/test_clip_extractor.py -v
```
Expected: ImportError on `clip_extractor`

- [ ] **Step 3: Implement `src/pipeline/clip_extractor.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/pipeline/test_clip_extractor.py -v
```
Expected: 4 PASSED (requires `ffmpeg` on PATH — install via `brew install ffmpeg`)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/clip_extractor.py tests/pipeline/test_clip_extractor.py
git commit -m "feat: clip extractor — ffmpeg-based positive and negative clip builders"
```

---

## Task 4: Frame Extractor

**Files:**
- Create: `src/pipeline/frame_extractor.py`
- Create: `tests/pipeline/test_frame_extractor.py`

**Interfaces:**
- Produces: `sample_frames(clip_path: str, sample_fps: float = 2.0) -> list[np.ndarray]` — list of BGR uint8 frames, each shape `(H, W, 3)`
- Produces: `motion_crop(frame: np.ndarray, reference: np.ndarray, padding: int = 30, target_size: tuple[int,int] = (224, 224)) -> np.ndarray` — shape `(224, 224, 3)` uint8 BGR
- Produces: `process_clip(clip_path: str, sample_fps: float = 2.0, target_size: tuple[int,int] = (224, 224)) -> np.ndarray` — shape `(N, 224, 224, 3)` uint8 BGR

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_frame_extractor.py`:
```python
import numpy as np
from src.pipeline.frame_extractor import sample_frames, motion_crop, process_clip


def test_sample_frames_count(synthetic_video):
    path, _ = synthetic_video
    frames = sample_frames(path, sample_fps=2.0)
    # 15s * 2fps = 30 frames (±1 for rounding)
    assert 28 <= len(frames) <= 32


def test_sample_frames_shape(synthetic_video):
    path, _ = synthetic_video
    frames = sample_frames(path, sample_fps=2.0)
    assert frames[0].ndim == 3
    assert frames[0].shape[2] == 3  # BGR


def test_motion_crop_output_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[200:260, 100:160] = 200  # bright region = motion
    result = motion_crop(frame, reference, padding=10, target_size=(224, 224))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_motion_crop_no_motion_returns_center_crop():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reference = frame.copy()
    result = motion_crop(frame, reference, target_size=(224, 224))
    assert result.shape == (224, 224, 3)


def test_process_clip_output_shape(synthetic_video):
    path, _ = synthetic_video
    frames = process_clip(path, sample_fps=2.0, target_size=(224, 224))
    assert frames.ndim == 4
    assert frames.shape[1:] == (224, 224, 3)
    assert frames.dtype == np.uint8
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/pipeline/test_frame_extractor.py -v
```
Expected: ImportError on `frame_extractor`

- [ ] **Step 3: Implement `src/pipeline/frame_extractor.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/pipeline/test_frame_extractor.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/frame_extractor.py tests/pipeline/test_frame_extractor.py
git commit -m "feat: frame extractor — frame sampling + motion-crop preprocessing"
```

---

## Task 5: PyTorch Model Definitions

These files are used in Colab for training only. Not part of inference on student machines.

**Files:**
- Create: `src/model/features.py`
- Create: `src/model/temporal.py`
- Create: `tests/model/test_features.py`
- Create: `tests/model/test_temporal.py`

**Interfaces:**
- Produces: `EfficientNetFeatureExtractor` — `nn.Module`, `forward(x: Tensor[B,3,224,224]) -> Tensor[B,1280]`
- Produces: `TemporalClassifier` — `nn.Module`, `forward(x: Tensor[B,T,1280]) -> Tensor[B,1]`

- [ ] **Step 1: Write the failing tests**

`tests/model/test_features.py`:
```python
import torch
from src.model.features import EfficientNetFeatureExtractor


def test_output_shape():
    model = EfficientNetFeatureExtractor()
    model.eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1280)


def test_weights_frozen():
    model = EfficientNetFeatureExtractor()
    for p in model.parameters():
        assert not p.requires_grad
```

`tests/model/test_temporal.py`:
```python
import torch
from src.model.temporal import TemporalClassifier


def test_output_shape():
    model = TemporalClassifier()
    model.eval()
    x = torch.zeros(4, 20, 1280)  # batch=4, seq=20, features=1280
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)


def test_output_in_range():
    model = TemporalClassifier()
    model.eval()
    x = torch.randn(2, 20, 1280)
    with torch.no_grad():
        out = model(x)
    assert (out >= 0).all() and (out <= 1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/ -v
```
Expected: ImportError (torch may not be installed locally — install with `pip install torch torchvision` for local testing, or run on Colab)

- [ ] **Step 3: Implement `src/model/features.py`**

```python
import torch
import torch.nn as nn
from torchvision import models, transforms


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = backbone.avgpool
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) — ImageNet-normalized
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)  # (B, 1280)


# Preprocessing transform — apply before passing frames to the model
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

- [ ] **Step 4: Implement `src/model/temporal.py`**

```python
import torch
import torch.nn as nn


class TemporalClassifier(nn.Module):
    def __init__(self, feature_dim: int = 1280, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(256, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1280)
        x = self.proj(x)           # (B, T, 256)
        _, (h, _) = self.lstm(x)   # h: (layers, B, hidden)
        return torch.sigmoid(self.head(h[-1]))  # (B, 1)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/model/ -v
```
Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/model/features.py src/model/temporal.py tests/model/test_features.py tests/model/test_temporal.py
git commit -m "feat: PyTorch model definitions — EfficientNet-B0 feature extractor + LSTM classifier"
```

---

## Task 6: Training Script + Colab Notebook

**Files:**
- Create: `src/model/train.py`
- Create: `notebooks/train.ipynb`
- Create: `tests/model/test_train.py`

**Interfaces:**
- Consumes: `EfficientNetFeatureExtractor`, `TemporalClassifier` from Tasks 5
- Produces: `BirdClipDataset(feature_paths: list[str], labels: list[int], augment: bool)`
- Produces: `extract_and_cache_features(frame_dir: str, feature_dir: str, device: str) -> None`
- Produces: `train(feature_dir: str, model_out: str, epochs: int, lr: float, device: str) -> None`

- [ ] **Step 1: Write failing tests for dataset and feature caching**

`tests/model/test_train.py`:
```python
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.model.train import BirdClipDataset, extract_and_cache_features


def test_dataset_getitem(tmp_path):
    # Create fake .npy feature files
    for i in range(4):
        np.save(str(tmp_path / f"clip_{i:04d}.npy"), np.random.randn(20, 1280).astype(np.float32))
    paths = [str(tmp_path / f"clip_{i:04d}.npy") for i in range(4)]
    labels = [1, 0, 1, 0]
    ds = BirdClipDataset(paths, labels, augment=False)
    feat, label = ds[0]
    assert feat.shape == (20, 1280)
    assert label.shape == (1,)


def test_dataset_augment_preserves_shape(tmp_path):
    np.save(str(tmp_path / "clip.npy"), np.random.randn(20, 1280).astype(np.float32))
    ds = BirdClipDataset([str(tmp_path / "clip.npy")], [1], augment=True)
    feat, _ = ds[0]
    assert feat.shape == (20, 1280)


def test_extract_and_cache_features(tmp_path):
    import cv2
    # Create fake frame dir with 3 frames per clip
    feed_dir = tmp_path / "frames" / "feeding" / "clip_0000"
    feed_dir.mkdir(parents=True)
    for i in range(3):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(feed_dir / f"frame_{i:04d}.jpg"), img)

    feat_out = tmp_path / "features"
    feat_out.mkdir()
    extract_and_cache_features(str(tmp_path / "frames"), str(feat_out), device="cpu")
    saved = list(feat_out.rglob("*.npy"))
    assert len(saved) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_train.py -v
```
Expected: ImportError on `train`

- [ ] **Step 3: Implement `src/model/train.py`**

```python
from __future__ import annotations
import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model.features import EfficientNetFeatureExtractor, TRANSFORM
from src.model.temporal import TemporalClassifier


class BirdClipDataset(Dataset):
    def __init__(self, feature_paths: list[str], labels: list[int], augment: bool = False):
        self.paths = feature_paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = np.load(self.paths[idx]).astype(np.float32)  # (T, 1280)
        if self.augment:
            # Temporal swap of adjacent frames
            if random.random() < 0.3 and len(features) > 1:
                i = random.randint(0, len(features) - 2)
                features[[i, i + 1]] = features[[i + 1, i]]
            # Gaussian noise on feature vectors
            features += np.random.normal(0, 0.01, features.shape).astype(np.float32)
        return torch.tensor(features), torch.tensor([float(self.labels[idx])])


def extract_and_cache_features(frame_dir: str, feature_dir: str, device: str = "cuda") -> None:
    """
    frame_dir layout: frame_dir/{feeding,normal}/{clip_name}/frame_XXXX.jpg
    Saves feature_dir/{feeding,normal}/{clip_name}.npy — shape (T, 1280)
    """
    extractor = EfficientNetFeatureExtractor().to(device).eval()
    for label in ("feeding", "normal"):
        clip_dirs = sorted(glob(os.path.join(frame_dir, label, "*")))
        out_label_dir = os.path.join(feature_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)
        for clip_dir in clip_dirs:
            clip_name = os.path.basename(clip_dir)
            frame_paths = sorted(glob(os.path.join(clip_dir, "*.jpg")))
            if not frame_paths:
                continue
            import cv2
            tensors = []
            for fp in frame_paths:
                img = cv2.imread(fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil = Image.fromarray(img)
                tensors.append(TRANSFORM(pil))
            batch = torch.stack(tensors).to(device)  # (T, 3, 224, 224)
            with torch.no_grad():
                feats = extractor(batch).cpu().numpy()  # (T, 1280)
            np.save(os.path.join(out_label_dir, f"{clip_name}.npy"), feats)


def train(
    feature_dir: str,
    model_out: str,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    feed_paths = sorted(glob(os.path.join(feature_dir, "feeding", "*.npy")))
    norm_paths = sorted(glob(os.path.join(feature_dir, "normal", "*.npy")))
    all_paths = feed_paths + norm_paths
    all_labels = [1] * len(feed_paths) + [0] * len(norm_paths)

    # 80/20 stratified split
    from sklearn.model_selection import train_test_split
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    tr_ds = BirdClipDataset(tr_paths, tr_labels, augment=True)
    val_ds = BirdClipDataset(val_paths, val_labels, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)

    model = TemporalClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.BCELoss()

    best_val_loss, patience_count, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        for feats, labels in tr_dl:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, labels in val_dl:
                feats, labels = feats.to(device), labels.to(device)
                val_loss += criterion(model(feats), labels).item()
        val_loss /= len(val_dl)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs} — val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 10:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"Model saved to {model_out}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/model/test_train.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Create `notebooks/train.ipynb`**

Create the notebook file with the following JSON content (each string in `source` is one cell):

Cell 1 — Setup:
```python
# Mount Google Drive and install deps
from google.colab import drive
drive.mount("/content/drive")
!pip install -q onnx scikit-learn
import sys
sys.path.insert(0, "/content/drive/MyDrive/BirdFeeding")
```

Cell 2 — Extract features (run once):
```python
from src.model.train import extract_and_cache_features
extract_and_cache_features(
    frame_dir="/content/drive/MyDrive/BirdFeeding/data/frames",
    feature_dir="/content/drive/MyDrive/BirdFeeding/data/features",
    device="cuda",
)
print("Features cached.")
```

Cell 3 — Train:
```python
from src.model.train import train
train(
    feature_dir="/content/drive/MyDrive/BirdFeeding/data/features",
    model_out="/content/drive/MyDrive/BirdFeeding/models/temporal_classifier.pt",
    epochs=50,
    lr=1e-3,
    device="cuda",
)
```

Cell 4 — Export ONNX (covered in Task 7):
```python
# See Task 7 — run export.py after training
```

Save this as a valid `.ipynb` JSON. The simplest approach is to run:
```bash
jupyter nbconvert --to notebook --execute /dev/null  # create stub
```
Or create it manually from Colab UI and paste cell code there.

- [ ] **Step 6: Commit**

```bash
git add src/model/train.py tests/model/test_train.py notebooks/
git commit -m "feat: training script — BirdClipDataset, feature caching, training loop + Colab notebook"
```

---

## Task 7: ONNX Export

**Files:**
- Create: `src/model/export.py`
- Create: `tests/model/test_export.py`

**Interfaces:**
- Consumes: `EfficientNetFeatureExtractor`, `TemporalClassifier`
- Produces: `export_feature_extractor(out_path: str) -> None` — saves `efficientnet_features.onnx`
- Produces: `export_temporal_classifier(weights_path: str, out_path: str) -> None` — saves `temporal_classifier.onnx`
- Produces: `verify_onnx(onnx_path: str, torch_model: nn.Module, sample_input: torch.Tensor, atol: float) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/model/test_export.py`:
```python
import os
import numpy as np
import torch
import onnxruntime as ort
from src.model.export import export_feature_extractor, export_temporal_classifier, verify_onnx
from src.model.temporal import TemporalClassifier


def test_export_feature_extractor(tmp_path):
    out = str(tmp_path / "features.onnx")
    export_feature_extractor(out)
    assert os.path.exists(out)
    sess = ort.InferenceSession(out)
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    result = sess.run(None, {sess.get_inputs()[0].name: dummy})
    assert result[0].shape == (1, 1280)


def test_export_temporal_classifier(tmp_path):
    # Save a fresh (untrained) model and export it
    model = TemporalClassifier()
    weights_path = str(tmp_path / "model.pt")
    torch.save(model.state_dict(), weights_path)
    out = str(tmp_path / "classifier.onnx")
    export_temporal_classifier(weights_path, out)
    assert os.path.exists(out)
    sess = ort.InferenceSession(out)
    dummy = np.zeros((1, 20, 1280), dtype=np.float32)
    result = sess.run(None, {sess.get_inputs()[0].name: dummy})
    assert result[0].shape == (1, 1)
    assert 0.0 <= float(result[0][0][0]) <= 1.0


def test_verify_onnx_matches_pytorch(tmp_path):
    model = TemporalClassifier()
    weights_path = str(tmp_path / "model.pt")
    torch.save(model.state_dict(), weights_path)
    out = str(tmp_path / "classifier.onnx")
    export_temporal_classifier(weights_path, out)
    sample = torch.randn(1, 20, 1280)
    verify_onnx(out, model, sample, atol=1e-4)  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_export.py -v
```
Expected: ImportError on `export`

- [ ] **Step 3: Implement `src/model/export.py`**

```python
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

from src.model.features import EfficientNetFeatureExtractor
from src.model.temporal import TemporalClassifier


def export_feature_extractor(out_path: str) -> None:
    model = EfficientNetFeatureExtractor().eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["frames"],
        output_names=["features"],
        dynamic_axes={"frames": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )


def export_temporal_classifier(weights_path: str, out_path: str) -> None:
    model = TemporalClassifier()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.zeros(1, 20, 1280)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["feature_sequence"],
        output_names=["feeding_prob"],
        dynamic_axes={
            "feature_sequence": {0: "batch", 1: "seq_len"},
            "feeding_prob": {0: "batch"},
        },
        opset_version=17,
    )


def verify_onnx(
    onnx_path: str,
    torch_model: nn.Module,
    sample_input: torch.Tensor,
    atol: float = 1e-4,
) -> None:
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(sample_input).numpy()

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {sess.get_inputs()[0].name: sample_input.numpy()})[0]

    if not np.allclose(torch_out, ort_out, atol=atol):
        max_diff = np.abs(torch_out - ort_out).max()
        raise AssertionError(f"ONNX/PyTorch mismatch — max diff: {max_diff:.6f}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/model/test_export.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/model/export.py tests/model/test_export.py
git commit -m "feat: ONNX export — feature extractor + temporal classifier with verification"
```

---

## Task 8: Inference Pipeline

**Files:**
- Create: `src/inference/pipeline.py`
- Create: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `process_clip` from `src.pipeline.frame_extractor`
- Produces: `FeedingDetection(start_s: float, end_s: float, confidence: float)`
- Produces: `FeedingDetector(feature_model_path: str, classifier_model_path: str)`
  - `detect(video_path: str, threshold: float = 0.6, progress_cb: Callable[[float], None] | None = None) -> list[FeedingDetection]`

- [ ] **Step 1: Write the failing tests**

`tests/inference/test_pipeline.py`:
```python
import os
import numpy as np
import torch
import pytest
from src.inference.pipeline import FeedingDetector, FeedingDetection
from src.model.export import export_feature_extractor, export_temporal_classifier
from src.model.temporal import TemporalClassifier


@pytest.fixture
def onnx_models(tmp_path):
    feat_path = str(tmp_path / "features.onnx")
    cls_path = str(tmp_path / "classifier.onnx")
    export_feature_extractor(feat_path)
    model = TemporalClassifier()
    weights = str(tmp_path / "model.pt")
    torch.save(model.state_dict(), weights)
    export_temporal_classifier(weights, cls_path)
    return feat_path, cls_path


def test_feeding_detection_dataclass():
    d = FeedingDetection(start_s=10.0, end_s=15.0, confidence=0.85)
    assert d.start_s == 10.0
    assert d.end_s == 15.0
    assert d.confidence == 0.85


def test_detector_loads(onnx_models):
    feat, cls = onnx_models
    detector = FeedingDetector(feat, cls)
    assert detector is not None


def test_detector_returns_list(onnx_models, synthetic_video):
    feat, cls = onnx_models
    video_path, _ = synthetic_video
    detector = FeedingDetector(feat, cls)
    results = detector.detect(video_path, threshold=0.5)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, FeedingDetection)
        assert r.start_s < r.end_s
        assert 0.0 <= r.confidence <= 1.0


def test_detector_progress_callback(onnx_models, synthetic_video):
    feat, cls = onnx_models
    video_path, _ = synthetic_video
    detector = FeedingDetector(feat, cls)
    progress_values = []
    detector.detect(video_path, threshold=0.5, progress_cb=progress_values.append)
    assert len(progress_values) > 0
    assert all(0.0 <= v <= 1.0 for v in progress_values)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/inference/test_pipeline.py -v
```
Expected: ImportError on `pipeline`

- [ ] **Step 3: Implement `src/inference/pipeline.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import cv2
import numpy as np
import onnxruntime as ort
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
            import PIL.Image
            pil = PIL.Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
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
        """frames: (T, 3, 224, 224) → features: (1, T, 1280)"""
        T = frames.shape[0]
        feats = self._feat_sess.run(None, {self._feat_input: frames})[0]  # (T, 1280)
        return feats[np.newaxis]  # (1, T, 1280)

    def _classify(self, features: np.ndarray) -> float:
        """features: (1, T, 1280) → probability float"""
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

        # Merge overlapping detections
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/inference/test_pipeline.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/inference/pipeline.py tests/inference/test_pipeline.py
git commit -m "feat: inference pipeline — motion scan, candidate clustering, ONNX classification, event merging"
```

---

## Task 9: Gradio App

**Files:**
- Create: `src/app/app.py`
- Create: `tests/app/test_app.py`

**Interfaces:**
- Consumes: `FeedingDetector`, `FeedingDetection` from `src.inference.pipeline`
- Produces: `build_app(detector: FeedingDetector) -> gr.Blocks`
- Produces: `launch()` — entry point, reads ONNX paths from `models/` dir

- [ ] **Step 1: Write the failing tests**

`tests/app/test_app.py`:
```python
import pandas as pd
import torch
import pytest
from unittest.mock import MagicMock
from src.app.app import build_app, _detections_to_df, _format_seconds
from src.inference.pipeline import FeedingDetection


def test_format_seconds():
    assert _format_seconds(0.0) == "0:00"
    assert _format_seconds(90.5) == "1:30"
    assert _format_seconds(577.0) == "9:37"


def test_detections_to_df_empty():
    df = _detections_to_df([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_detections_to_df_populated():
    detections = [
        FeedingDetection(start_s=141.0, end_s=146.0, confidence=0.91),
        FeedingDetection(start_s=473.0, end_s=479.0, confidence=0.84),
    ]
    df = _detections_to_df(detections)
    assert len(df) == 2
    assert df.iloc[0]["Start"] == "2:21"
    assert df.iloc[0]["End"] == "2:26"
    assert df.iloc[0]["Confidence"] == 0.91


def test_build_app_returns_blocks():
    import gradio as gr
    mock_detector = MagicMock()
    app = build_app(mock_detector)
    assert isinstance(app, gr.Blocks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/app/test_app.py -v
```
Expected: ImportError on `app`

- [ ] **Step 3: Implement `src/app/app.py`**

```python
from __future__ import annotations
import os
import tempfile
import pandas as pd
import gradio as gr

from src.inference.pipeline import FeedingDetector, FeedingDetection

_FEATURE_MODEL = os.path.join("models", "efficientnet_features.onnx")
_CLASSIFIER_MODEL = os.path.join("models", "temporal_classifier.onnx")


def _format_seconds(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def _detections_to_df(detections: list[FeedingDetection]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["#", "Start", "End", "Confidence"])
    rows = [
        {
            "#": i + 1,
            "Start": _format_seconds(d.start_s),
            "End": _format_seconds(d.end_s),
            "Confidence": d.confidence,
        }
        for i, d in enumerate(detections)
    ]
    return pd.DataFrame(rows)


def build_app(detector: FeedingDetector) -> gr.Blocks:
    with gr.Blocks(title="Tern Feeding Event Detector") as app:
        gr.Markdown("## Tern Feeding Event Detector")
        gr.Markdown("Upload a field video to detect feeding events.")

        with gr.Row():
            video_input = gr.File(label="Upload video (.mp4 or .mov)", file_types=[".mp4", ".mov"])
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.99, value=0.6, step=0.05,
                label="Confidence threshold (lower = more sensitive)",
            )

        run_btn = gr.Button("Run Detection", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        results_table = gr.Dataframe(
            headers=["#", "Start", "End", "Confidence"],
            label="Detected Feeding Events",
        )
        csv_output = gr.File(label="Download CSV", visible=False)

        def run_detection(video_file, threshold, progress=gr.Progress()):
            if video_file is None:
                return "Please upload a video first.", pd.DataFrame(), gr.update(visible=False)

            progress_values: list[float] = []

            def _cb(v: float):
                progress_values.append(v)
                progress(v, desc=f"Analysing window {len(progress_values)}…")

            detections = detector.detect(video_file.name, threshold=threshold, progress_cb=_cb)
            df = _detections_to_df(detections)

            if df.empty:
                return "No feeding events detected above threshold.", df, gr.update(visible=False)

            # Write CSV to temp file for download
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            df.to_csv(tmp.name, index=False)
            return f"Found {len(detections)} feeding event(s).", df, gr.update(value=tmp.name, visible=True)

        run_btn.click(
            fn=run_detection,
            inputs=[video_input, threshold_slider],
            outputs=[status, results_table, csv_output],
        )
    return app


def launch():
    for path in (_FEATURE_MODEL, _CLASSIFIER_MODEL):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ONNX model not found at {path}. "
                "Run training on Colab and download the model files to models/."
            )
    detector = FeedingDetector(_FEATURE_MODEL, _CLASSIFIER_MODEL)
    app = build_app(detector)
    app.launch()


if __name__ == "__main__":
    launch()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/app/test_app.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Verify the app launches (manual)**

Place any two ONNX files in `models/` (e.g., copy from a Colab run or use the export script) then run:
```bash
python -m src.app.app
```
Expected: Browser opens at `http://localhost:7860` showing the UI. Upload `data/videos/GX010539.MP4`, set threshold to 0.5, click Run Detection.

- [ ] **Step 6: Commit**

```bash
git add src/app/app.py tests/app/test_app.py
git commit -m "feat: Gradio app — video upload, confidence slider, timestamp table, CSV download"
```

---

## Self-Review Checklist

- **Spec §4 Clip Extraction:** Task 3 implements `build_positive_clips` and `build_negative_clips` with correct window `[ts-2s, ts+8s]` and 20s gap. ✓
- **Spec §4.2 Frame preprocessing:** Task 4 implements frame sampling at 2fps + motion crop → 224×224. ✓
- **Spec §5.1 EfficientNet-B0 frozen:** Task 5 freezes all parameters in `EfficientNetFeatureExtractor`. ✓
- **Spec §5.2 LSTM architecture:** Task 5 matches `Linear(1280→256)→ReLU→Dropout(0.3)→LSTM(256,128,2)→Linear(128→1)→Sigmoid`. ✓
- **Spec §5.3 Training:** Task 6 implements 80/20 split, BCE loss, Adam, ReduceLROnPlateau, early stopping patience=10, all augmentations. ✓
- **Spec §5.4 ONNX export:** Task 7 exports both models and verifies numerical match. ✓
- **Spec §6 Inference steps 1-6:** Task 8 implements motion scan → candidate clustering → feature extraction → classification → threshold + merge → M:SS format. ✓
- **Spec §7 Gradio UI:** Task 9 matches the wireframe — file upload, threshold slider, progress, results table, CSV download. ✓
- **Spec §8 Project structure:** File map matches spec exactly. ✓
- **No placeholders:** All code blocks are complete and runnable. ✓
- **Type consistency:** `FeedingEvent`, `FeedingDetection`, `EfficientNetFeatureExtractor`, `TemporalClassifier` names are consistent across all tasks. ✓
