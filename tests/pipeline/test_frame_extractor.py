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
