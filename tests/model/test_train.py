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


def test_video_level_split_no_video_in_both_sets(tmp_path):
    """No video ID should appear in both train and validation paths."""
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
