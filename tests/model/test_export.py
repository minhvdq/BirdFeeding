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
