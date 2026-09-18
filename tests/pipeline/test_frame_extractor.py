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
