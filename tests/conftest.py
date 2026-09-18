import os
import pytest
import cv2
import numpy as np


@pytest.fixture
def synthetic_video(tmp_path):
    """15-second 640x480 MP4 at 30fps. Bright rectangle moves at t=9-12s."""
    path = str(tmp_path / "test_video.mp4")
    w, h, fps, duration_s = 640, 480, 30, 15
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(42)
    for i in range(fps * duration_s):
        frame = rng.integers(20, 50, (h, w, 3), dtype=np.uint8)
        if 270 <= i <= 360:  # seconds 9-12
            x = int((i - 270) / 3) + 80
            frame[200:260, x : x + 60] = 220  # bright rectangle = "bird"
        writer.write(frame)
    writer.release()
    return path, 9.0


@pytest.fixture
def sample_csv(tmp_path):
    """Minimal CSV with two feeding events and one Start/End row each."""
    path = str(tmp_path / "events.csv")
    content = (
        "date,Year,Observer,GoPro video ID,Watch ID,Area,"
        "Video timestamp (m.s),Time,Event (start; end; feeding; failed feed; stolen from; stolen by),"
        "Nest ID\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,0:00,,Start,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,9:37,,Feeding,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,5:10,,Feeding Fail,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,3:00,,Stealing,N1\n"
        "1-Jul,2023,SV,GX_TEST,W1,Area1,11:48,,End,N1\n"
    )
    path_obj = tmp_path / "events.csv"
    path_obj.write_text(content)
    return str(path)
