# Pipeline Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace motion-crop preprocessing with full-frame letterbox resize, add boundary hard negatives to the training data pipeline, and switch model training from clip-level to video-level train/val split.

**Architecture:** Three independent changes to existing files: (1) `frame_extractor.py` and `inference/pipeline.py` drop `motion_crop` in favour of `letterbox_resize`; (2) `clip_extractor.py` adds a boundary-negative tier to `build_negative_clips`; (3) `train.py` replaces sklearn's clip-level `train_test_split` with a manual video-level split. No new files. No interface changes visible to `collect_data.py` or the Gradio app.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, PyTorch (training only), ONNX Runtime (inference)

**Spec:** `docs/superpowers/specs/2026-09-17-bird-feeding-detection-design.md`

## Global Constraints

- Python ≥ 3.10
- `onnxruntime ≥ 1.17.0`, `opencv-python ≥ 4.9.0`, `numpy ≥ 1.26.0`
- No PyTorch or torchvision in inference path (`src/inference/`, `src/app/`)
- `process_clip` signature unchanged: `(clip_path, sample_fps=2.0, target_size=(224,224)) -> np.ndarray` returning `(N,224,224,3)` uint8
- `build_negative_clips` signature backward-compatible: all new parameters have defaults matching current behaviour
- All 27 existing tests must pass after each task
- Run tests with: `source venv/bin/activate && pytest tests/ -v` from repo root
- Do not add Co-Authored-By trailers to commits

---

## File Map

| File | Change |
|---|---|
| `src/pipeline/frame_extractor.py` | Replace `motion_crop` with `letterbox_resize`; update `process_clip` |
| `src/inference/pipeline.py` | Swap `motion_crop` import for `letterbox_resize`; update `_extract_clip_frames` |
| `tests/pipeline/test_frame_extractor.py` | Replace `motion_crop` tests with `letterbox_resize` tests |
| `src/pipeline/clip_extractor.py` | Add `_boundary_candidates` helper; add boundary tier to `build_negative_clips` |
| `tests/pipeline/test_clip_extractor.py` | Add test for boundary negative generation |
| `src/model/train.py` | Add `_video_id_from_path` + `_video_level_split`; call from `train()` |
| `tests/model/test_train.py` | Add test asserting no video appears on both sides of the split |

---

## Task 1: Full-Frame Letterbox Preprocessing

**Files:**
- Modify: `src/pipeline/frame_extractor.py`
- Modify: `src/inference/pipeline.py`
- Modify: `tests/pipeline/test_frame_extractor.py`

**Interfaces:**
- Produces: `letterbox_resize(frame: np.ndarray, target_size: tuple[int,int] = (224,224)) -> np.ndarray` — returned array is `(224,224,3)` uint8
- Produces: `process_clip` (signature unchanged) — now uses `letterbox_resize` internally instead of `motion_crop`
- `motion_crop` is deleted entirely; nothing outside this task depends on it after this task

- [ ] **Step 1: Write failing tests for `letterbox_resize`**

Replace the two `motion_crop` tests in `tests/pipeline/test_frame_extractor.py` with three `letterbox_resize` tests. Keep the two `sample_frames` tests and the `process_clip` test unchanged.

```python
# tests/pipeline/test_frame_extractor.py
import numpy as np
from src.pipeline.frame_extractor import sample_frames, letterbox_resize, process_clip


def test_sample_frames_count(synthetic_video):
    path, _ = synthetic_video
    frames = sample_frames(path, sample_fps=2.0)
    assert 28 <= len(frames) <= 32


def test_sample_frames_shape(synthetic_video):
    path, _ = synthetic_video
    frames = sample_frames(path, sample_fps=2.0)
    assert frames[0].ndim == 3
    assert frames[0].shape[2] == 3  # BGR


def test_letterbox_resize_output_shape():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = letterbox_resize(frame, target_size=(224, 224))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_letterbox_resize_wide_frame_pads_top_bottom():
    # 2:1 wide frame → top and bottom rows should be black padding
    frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
    result = letterbox_resize(frame, target_size=(224, 224))
    assert result.shape == (224, 224, 3)
    # Content fills the horizontal centre strip; top row is padding
    assert result[0, 112].sum() == 0       # top padding is black
    assert result[112, 112].sum() > 0      # centre has content


def test_letterbox_resize_square_frame_no_padding():
    frame = np.ones((480, 480, 3), dtype=np.uint8) * 100
    result = letterbox_resize(frame, target_size=(224, 224))
    assert result.shape == (224, 224, 3)
    # No padding: every pixel should be non-zero
    assert result.min() > 0


def test_process_clip_output_shape(synthetic_video):
    path, _ = synthetic_video
    frames = process_clip(path, sample_fps=2.0, target_size=(224, 224))
    assert frames.ndim == 4
    assert frames.shape[1:] == (224, 224, 3)
    assert frames.dtype == np.uint8
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source venv/bin/activate && pytest tests/pipeline/test_frame_extractor.py -v
```

Expected: `test_letterbox_resize_*` → FAIL (`ImportError: cannot import name 'letterbox_resize'`)

- [ ] **Step 3: Implement `letterbox_resize` and update `process_clip` in `frame_extractor.py`**

Replace the entire file:

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
```

- [ ] **Step 4: Update `_extract_clip_frames` in `src/inference/pipeline.py`**

Change the import at line 8 from `motion_crop` to `letterbox_resize`, and remove the reference frame / `motion_crop` call inside `_extract_clip_frames`:

```python
# Line 8 — change this:
from src.pipeline.frame_extractor import motion_crop
# to this:
from src.pipeline.frame_extractor import letterbox_resize
```

Replace `_extract_clip_frames` (lines 65–89):

```python
def _extract_clip_frames(
    video_path: str, center_s: float, clip_duration_s: float = 10.0, sample_fps: float = 2.0
) -> np.ndarray:
    """Extract and preprocess frames around center_s. Returns (T, 3, 224, 224) float32."""
    start_s = max(0.0, center_s - clip_duration_s / 2)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src_fps))
    interval = max(1, round(src_fps / sample_fps))
    n_frames = int(clip_duration_s * sample_fps)
    frames, idx = [], 0
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            resized = letterbox_resize(frame, target_size=(224, 224))
            frames.append(_normalize_frame(resized))
        idx += 1
    cap.release()
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((3, 224, 224), dtype=np.float32))
    return np.stack(frames[:n_frames], axis=0).astype(np.float32)  # (T, 3, 224, 224)
```

- [ ] **Step 5: Run all tests**

```bash
source venv/bin/activate && pytest tests/ -v
```

Expected: all tests pass (27 total)

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/frame_extractor.py src/inference/pipeline.py tests/pipeline/test_frame_extractor.py
git commit -m "refactor: replace motion_crop with full-frame letterbox resize"
```

---

## Task 2: Boundary Negative Sampling

**Files:**
- Modify: `src/pipeline/clip_extractor.py`
- Modify: `tests/pipeline/test_clip_extractor.py`

**Interfaces:**
- Produces: `_boundary_candidates(feeding_timestamps, duration, clip_duration_s, pre_s, post_s) -> list[tuple[float,float]]` — internal helper, not imported externally
- `build_negative_clips` signature gains two new keyword-only parameters with defaults: `pre_s: float = 1.0`, `post_s: float = 4.0`. Existing callers without these args are unaffected.

- [ ] **Step 1: Write failing test for boundary negative generation**

Add one test to `tests/pipeline/test_clip_extractor.py` (keep all existing tests):

```python
def test_build_negative_clips_includes_boundary_windows(synthetic_video, tmp_path):
    """With a feeding event near the middle of a 15s video, boundary windows
    just after the event must be generated."""
    video_path, ts = synthetic_video  # ts is around 7.5s in a 15s video
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    import shutil
    shutil.copy(video_path, os.path.join(video_dir, "GX_TEST.mp4"))
    out_dir = str(tmp_path / "clips" / "normal_boundary")
    events = [FeedingEvent(video_id="GX_TEST", timestamp_s=ts)]
    paths = build_negative_clips(
        events, video_dir, out_dir,
        clips_per_video=2, min_gap_s=20.0,
        clip_duration_s=5.0, pre_s=1.0, post_s=4.0,
    )
    # At least one clip must have been created
    assert len(paths) >= 1
    for p in paths:
        assert os.path.exists(p)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
source venv/bin/activate && pytest tests/pipeline/test_clip_extractor.py::test_build_negative_clips_includes_boundary_windows -v
```

Expected: FAIL (`TypeError: build_negative_clips() got an unexpected keyword argument 'pre_s'`)

- [ ] **Step 3: Implement `_boundary_candidates` and update `build_negative_clips`**

Replace `build_negative_clips` (and add helper) in `src/pipeline/clip_extractor.py`:

```python
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
        if not any(abs(mid - ft) < clip_duration_s for ft in feeding_timestamps):
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
        if duration < clip_duration_s + min_gap_s:
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
```

- [ ] **Step 4: Run all tests**

```bash
source venv/bin/activate && pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/clip_extractor.py tests/pipeline/test_clip_extractor.py
git commit -m "feat: add boundary hard negatives to build_negative_clips"
```

---

## Task 3: Video-Level Train/Val Split

**Files:**
- Modify: `src/model/train.py`
- Modify: `tests/model/test_train.py`

**Interfaces:**
- Produces: `_video_id_from_path(path: str) -> str` — extracts video ID from feature path (`GX010539_feed_0000.npy` → `GX010539`)
- Produces: `_video_level_split(feature_dir: str) -> tuple[list[str], list[str], list[int], list[int]]` — returns `(tr_paths, val_paths, tr_labels, val_labels)`; `train()` calls this instead of sklearn

- [ ] **Step 1: Write failing test for video-level split**

Add to `tests/model/test_train.py` (keep all existing tests):

```python
def test_video_level_split_no_video_in_both_sets(tmp_path):
    """No video ID should appear in both train and validation paths."""
    import numpy as np
    from src.model.train import _video_id_from_path, _video_level_split

    feat_dir = tmp_path / "features"
    for label in ("feeding", "normal"):
        (feat_dir / label).mkdir(parents=True)
        for vid_num in range(5):
            vid_id = f"GX{vid_num:06d}"
            suffix = "feed" if label == "feeding" else "normal"
            for clip_num in range(2):
                p = feat_dir / label / f"{vid_id}_{suffix}_{clip_num:04d}.npy"
                np.save(str(p), np.random.randn(10, 1280).astype(np.float32))

    tr_paths, val_paths, tr_labels, val_labels = _video_level_split(str(feat_dir))

    tr_ids = {_video_id_from_path(p) for p in tr_paths}
    val_ids = {_video_id_from_path(p) for p in val_paths}

    # Strict: no video on both sides
    assert tr_ids.isdisjoint(val_ids), f"Overlap: {tr_ids & val_ids}"
    # Together they cover all 5 videos
    assert len(tr_ids | val_ids) == 5
    # Labels are binary
    assert set(tr_labels) <= {0, 1}
    assert set(val_labels) <= {0, 1}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
source venv/bin/activate && pytest tests/model/test_train.py::test_video_level_split_no_video_in_both_sets -v
```

Expected: FAIL (`ImportError: cannot import name '_video_id_from_path'`)

- [ ] **Step 3: Add `_video_id_from_path`, `_video_level_split` and update `train()`**

Add these two functions above `train()` in `src/model/train.py`, and replace the split block inside `train()`:

```python
def _video_id_from_path(path: str) -> str:
    """GX010539_feed_0000.npy  →  GX010539"""
    return os.path.basename(path).split("_")[0]


def _video_level_split(
    feature_dir: str,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Split clips into train/val ensuring no video appears on both sides."""
    from collections import defaultdict

    feed_paths = sorted(glob(os.path.join(feature_dir, "feeding", "*.npy")))
    norm_paths = sorted(glob(os.path.join(feature_dir, "normal", "*.npy")))
    all_paths = feed_paths + norm_paths
    all_labels = [1] * len(feed_paths) + [0] * len(norm_paths)

    by_video: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, label in zip(all_paths, all_labels):
        by_video[_video_id_from_path(path)].append((path, label))

    video_ids = sorted(by_video.keys())
    rng = random.Random(42)
    rng.shuffle(video_ids)
    n_val = max(1, len(video_ids) // 5)
    val_ids = set(video_ids[-n_val:])

    tr_paths, tr_labels, val_paths, val_labels = [], [], [], []
    for vid in video_ids:
        for path, label in by_video[vid]:
            if vid in val_ids:
                val_paths.append(path)
                val_labels.append(label)
            else:
                tr_paths.append(path)
                tr_labels.append(label)
    return tr_paths, val_paths, tr_labels, val_labels
```

Inside `train()`, replace the split block (currently lines 77–81):

```python
    # Replace:
    from sklearn.model_selection import train_test_split
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    # With:
    tr_paths, val_paths, tr_labels, val_labels = _video_level_split(feature_dir)
```

Also remove the now-unused top-of-function variables `all_paths` and `all_labels` from `train()` since `_video_level_split` builds them internally. The function signature of `train()` is unchanged.

- [ ] **Step 4: Run all tests**

```bash
source venv/bin/activate && pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/model/train.py tests/model/test_train.py
git commit -m "refactor: video-level train/val split to prevent data leakage"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Full-frame letterbox to 224×224 (§4.2) | Task 1 |
| MOG2 kept for candidate generation only, not cropping (§4.2 rationale) | Task 1 — `_scan_motion` unchanged |
| Boundary negatives alongside random negatives (§4.1) | Task 2 |
| Video-level 80/20 split (§5.3) | Task 3 |
| `collect_data.py` calls unchanged | All tasks — verified: `process_clip` and `build_negative_clips` signatures backward-compatible |

**Placeholder scan:** None found.

**Type consistency:**
- `letterbox_resize` returns `np.ndarray` (uint8) — matches `process_clip` expectation ✓
- `_extract_clip_frames` passes `letterbox_resize` output to `_normalize_frame` which expects BGR uint8 ✓
- `_video_level_split` returns `(list[str], list[str], list[int], list[int])` — matches what `train()` feeds into `BirdClipDataset` ✓
- `_boundary_candidates` returns `list[tuple[float,float]]` — iterated as `(start, end)` in `build_negative_clips` ✓
