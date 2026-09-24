from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

from src.model.features import EfficientNetFeatureExtractor
from src.model.temporal import TemporalClassifier


def export_feature_extractor(out_path: str) -> None:
    model = EfficientNetFeatureExtractor().eval()
    dummy = torch.zeros(1, 3, 480, 480)
    torch.onnx.export(
        model, dummy, out_path,
        dynamo=False,
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
        dynamo=False,
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
