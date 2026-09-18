import torch
from src.model.temporal import TemporalClassifier


def test_output_shape():
    model = TemporalClassifier()
    model.eval()
    x = torch.zeros(4, 20, 1280)  # batch=4, seq=20, features=1280
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)


def test_output_in_range():
    model = TemporalClassifier()
    model.eval()
    x = torch.randn(2, 20, 1280)
    with torch.no_grad():
        out = model(x)
    assert (out >= 0).all() and (out <= 1).all()
