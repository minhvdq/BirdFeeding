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
