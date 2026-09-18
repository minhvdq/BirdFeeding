import torch
from src.model.features import EfficientNetFeatureExtractor


def test_output_shape():
    model = EfficientNetFeatureExtractor()
    model.eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1280)


def test_weights_frozen():
    model = EfficientNetFeatureExtractor()
    for p in model.parameters():
        assert not p.requires_grad
