# Tern Feeding Behavior Detection — Design Spec

**Date:** 2026-09-17  
**Project:** Environmental Study Lab — Bird Feeding CV Application

---

## 1. Overview & Goals

Lab students currently watch 10–20 minute field videos manually and log feeding events into a spreadsheet. The goal is an application that accepts a video and automatically returns timestamps of feeding events, replacing the manual review step.

**Feeding event definition:** Any of — successful feed, failed feed, or prey stolen — all treated as one class. Events are 3–10 seconds long and always involve an adult tern flying into frame carrying prey, approaching a chick, and transferring (or attempting to transfer) the prey.

**Success criteria:**
- Detects feeding events with reasonable recall (missing fewer events than a distracted student)
- Acceptable false positive rate (non-feeding bird arrivals not flagged)
- Handles partial occlusion (feeding bird facing away from camera)
- Processes a 10–15 minute video in under 10 minutes on a MacBook Air M1

---

## 2. System Architecture

Two phases: build-time (run once by the developer) and run-time (used by lab students).

```
BUILD TIME
──────────
CSV + Videos (SharePoint / local)
        │
        ▼
  Clip Extractor          parse CSV timestamps → extract 10s clips per event
        │                 + sample negative clips from non-feeding windows
        ▼
  Google Drive            clips stored here (~100MB total)
        │
        ▼
  Colab Training Notebook
        ├─ Stage A: EfficientNet-B0 (frozen) → per-frame feature vectors
        └─ Stage B: LSTM temporal classifier → feeding / not feeding
        │
        ▼
  efficientnet_features.onnx + temporal_classifier.onnx  ← downloaded to local machine

RUN TIME
────────
Student uploads video → Gradio App
        │
        ▼
  Motion Detector         background subtraction (MOG2) → candidate windows
        │
        ▼
  Clip Sampler            10s clips around each candidate window
        │
        ▼
  Feature Extractor       EfficientNet-B0 ONNX → 20 × 1280-dim vectors per clip
        │
        ▼
  Temporal Classifier     LSTM ONNX → feeding probability per clip
        │
        ▼
  Event Merger            smooth + merge adjacent positives → (start, end) pairs
        │
        ▼
  Gradio UI               timestamp table + CSV download
```

---

## 3. Tech Stack

| Concern | Tool |
|---|---|
| Video processing | `ffmpeg`, `opencv-python` |
| Feature extraction | `torchvision` EfficientNet-B0 (pretrained ImageNet) |
| Temporal model | PyTorch LSTM |
| Model export | ONNX (`torch.onnx`) |
| Inference runtime | `onnxruntime` (CPU, no PyTorch needed on student machines) |
| Training environment | Google Colab (free T4 GPU) |
| Data bridge | Google Drive (mounted in Colab) |
| UI | Gradio |
| Language | Python 3.10+ |

---

## 4. Data Pipeline (Build-Time, runs on MacBook)

### 4.1 Clip Extraction

Input: `Feeding Data(Sheet1).csv` + videos (local folder or SharePoint direct URLs)

For each feeding event row in the CSV:
- Parse `Video timestamp` column → convert `M:SS` to seconds
- Extract `[timestamp - 1s, timestamp + 4s]` → **positive clip** (5s window)

For each video, sample negative clips in two tiers (equal split):
- **Random negatives:** random windows ≥ 20s from any feeding timestamp → easy background examples
- **Boundary negatives:** windows starting just after `timestamp + post_s` or ending just before `timestamp - pre_s` — hard examples capturing the adult arriving or departing without a feeding transfer

Both tiers use the same 5s duration.

If videos are on SharePoint with direct download URLs, use ffmpeg's remote seek to avoid downloading the full file:
```bash
ffmpeg -ss <start> -to <end> -i "<sharepoint-url>" -c copy clip.mp4
```
If SharePoint URLs are not directly accessible, the script processes one local video at a time (download → extract clips → delete video).

**Output:**
```
data/clips/
  feeding/    ← ~78 clips
  normal/     ← ~80–100 clips
```

### 4.2 Frame Extraction & Preprocessing

For each clip → sample at 2fps → 10 frames per clip (5s × 2fps).

For each frame:
1. Letterbox-resize the **full frame** to 480×480 (scale so the longer dimension = 480, pad shorter dimension with black). 224×224 was too small for 1080p GoPro footage — a bird occupying 10% of frame height became only 12px tall, losing too much spatial detail for EfficientNet to extract useful features.
2. Save as JPEG

**Rationale:** Motion-crop (absdiff → largest contour → bbox) was replaced because the largest contour frequently corresponds to background birds, grass, or water rather than the feeding interaction. Full-frame input forces the model to learn behavior-based patterns (adult flies in, approaches chick, transfers prey) that generalise across camera angles and backgrounds, which is the correct inductive bias for this task.

The MOG2 motion scan is still used at inference time for **candidate generation** (1fps coarse pass to find windows worth classifying), but motion information is no longer used to decide how to crop individual frames.

**Output:**
```
data/frames/
  feeding/    ← clips × 10 frames per clip
  normal/     ← clips × 10 frames per clip
```

---

## 5. Model Architecture (Training on Google Colab)

### 5.1 Stage A — Frame Feature Extraction (frozen, no training)

- Model: EfficientNet-B0 pretrained on ImageNet (via `torchvision.models`)
- Remove final classification layer → output 1280-dim feature vector per frame
- Run all training frames through once → cache feature vectors to disk
- This step is free: no training, just a forward pass

### 5.2 Stage B — Temporal Classifier (trained)

Input: sequence of 20 × 1280-dim vectors (one 10s clip)  
Output: scalar feeding probability (0–1)

Architecture:
```
Linear(1280 → 256) → ReLU → Dropout(0.3)
LSTM(256, hidden=128, layers=2, dropout=0.2)
Linear(128 → 1) → Sigmoid
```

Small enough to train in minutes on Colab T4 with ~160 clips.

### 5.3 Training

- **Split: by source video, not by clip.** Group all clips by their video ID prefix (e.g. `GX010539_feed_0000` → video `GX010539`). Split the set of video IDs 80/20. Every clip from a given video lands entirely in train or entirely in validation — never both. This prevents optimistic validation scores caused by shared background, lighting, and bird appearance across clips from the same video. The held-out validation videos are never seen during training.
- Loss: Binary cross-entropy
- Optimizer: Adam, lr=1e-3, reduce on plateau
- Epochs: up to 50 with early stopping (patience=10)
- Augmentation per clip:
  - Random horizontal flip (applied consistently across all frames in a clip)
  - Random brightness/contrast jitter (±20%)
  - Temporal jitter: randomly drop 1–2 frames and duplicate adjacent frames
  - Gaussian noise on feature vectors (σ=0.01)

### 5.4 Export

Both Stage A and Stage B exported as ONNX:
- `models/efficientnet_features.onnx`
- `models/temporal_classifier.onnx`

Inference on CPU via `onnxruntime` — no PyTorch dependency on student machines.

---

## 6. Inference Pipeline (Run-Time, on MacBook)

When a student uploads a video:

**Step 1 — Motion scan (1 fps)**  
Sample 1 frame per second. Run MOG2 background subtraction on each. Record timestamps where motion area exceeds a minimum threshold (filters out camera sensor noise).

**Step 2 — Candidate window generation**  
Cluster motion timestamps that are within 5s of each other → one candidate window per cluster. Extract a 10s clip centered on each cluster midpoint.

**Step 3 — Feature extraction**  
For each candidate clip: sample frames at 2fps → letterbox-resize each to 480×480 → run through `efficientnet_features.onnx` → T × 1280-dim sequence.

**Step 4 — Temporal classification**  
Feed sequence into `temporal_classifier.onnx` → feeding probability score.

**Step 5 — Threshold & merge**  
- Apply confidence threshold (default 0.6, tunable in UI)
- Candidate windows above threshold → feeding events
- Merge events whose time ranges overlap or are within 3s of each other
- Output: list of `(start_seconds, end_seconds, confidence)` tuples

**Step 6 — Format output**  
Convert seconds to `M:SS` format. Display in Gradio table. Offer CSV download.

---

## 7. Gradio Application

Single `app.py` file. Runs locally at `localhost:7860`, opens automatically in browser.

**UI layout:**
```
┌─────────────────────────────────────┐
│  Tern Feeding Event Detector        │
│                                     │
│  [  Upload video file  ]            │
│                                     │
│  Confidence threshold: [--●------]  │
│  (default 0.6)                      │
│                                     │
│  [ Run Detection ]                  │
│                                     │
│  Progress: ████████░░ 78%           │
│                                     │
│  Results:                           │
│  ┌──┬───────┬───────┬────────────┐  │
│  │# │ Start │  End  │ Confidence │  │
│  ├──┼───────┼───────┼────────────┤  │
│  │1 │  2:21 │  2:25 │   0.91     │  │
│  │2 │  7:53 │  7:59 │   0.84     │  │
│  └──┴───────┴───────┴────────────┘  │
│                                     │
│  [ Download CSV ]                   │
└─────────────────────────────────────┘
```

Notes:
- Accepts MP4, MOV (standard GoPro formats)
- Progress bar updates per candidate window processed
- If no events detected, shows "No feeding events detected above threshold"
- CSV output matches existing spreadsheet column format where possible

---

## 8. Project Structure

```
BirdFeeding/
  data/
    Feeding Data(Sheet1).csv
    videos/           ← temporary local video storage
    clips/
      feeding/
      normal/
    frames/           ← intermediate (can delete after training)
  src/
    pipeline/
      extract_clips.py      ← CSV + video → clips
      extract_frames.py     ← clips → cropped frames
    model/
      features.py           ← EfficientNet-B0 wrapper
      temporal.py           ← LSTM classifier definition
      export.py             ← PyTorch → ONNX
    inference/
      motion.py             ← MOG2 background subtraction
      pipeline.py           ← full inference pipeline
    app/
      app.py                ← Gradio UI
  models/
    efficientnet_features.onnx
    temporal_classifier.onnx
  notebooks/
    train.ipynb             ← Google Colab training notebook
  docs/
    superpowers/
      specs/
        2026-09-17-bird-feeding-detection-design.md
  requirements.txt          ← inference deps (onnxruntime, gradio, opencv)
  requirements-train.txt    ← training deps (torch, torchvision)
```

---

## 9. Out of Scope

- Individual bird identification (which adult fed which chick)
- Prey species classification (hake vs herring vs sand lance)
- Real-time video stream processing
- Multi-user authentication
- Cloud hosting (deferred — local Gradio app can be hosted later without architecture changes)
- Bounding box visualization overlaid on video
