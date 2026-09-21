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
            from PIL import Image
            tensors = []
            for fp in frame_paths:
                img = cv2.imread(fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img)
                tensors.append(TRANSFORM(pil))
            batch = torch.stack(tensors).to(device)  # (T, 3, 224, 224)
            with torch.no_grad():
                feats = extractor(batch).cpu().numpy()  # (T, 1280)
            np.save(os.path.join(out_label_dir, f"{clip_name}.npy"), feats)


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


def train(
    feature_dir: str,
    model_out: str,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    # Video-level split
    tr_paths, val_paths, tr_labels, val_labels = _video_level_split(feature_dir)

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
