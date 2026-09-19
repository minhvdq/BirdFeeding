import pandas as pd
import torch
import pytest
from unittest.mock import MagicMock
from src.app.app import build_app, _detections_to_df, _format_seconds
from src.inference.pipeline import FeedingDetection


def test_format_seconds():
    assert _format_seconds(0.0) == "0:00"
    assert _format_seconds(90.5) == "1:30"
    assert _format_seconds(577.0) == "9:37"


def test_detections_to_df_empty():
    df = _detections_to_df([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_detections_to_df_populated():
    detections = [
        FeedingDetection(start_s=141.0, end_s=146.0, confidence=0.91),
        FeedingDetection(start_s=473.0, end_s=479.0, confidence=0.84),
    ]
    df = _detections_to_df(detections)
    assert len(df) == 2
    assert df.iloc[0]["Start"] == "2:21"
    assert df.iloc[0]["End"] == "2:26"
    assert df.iloc[0]["Confidence"] == 0.91


def test_build_app_returns_blocks():
    import gradio as gr
    mock_detector = MagicMock()
    app = build_app(mock_detector)
    assert isinstance(app, gr.Blocks)
