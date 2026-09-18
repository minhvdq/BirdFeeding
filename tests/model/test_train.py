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
