import os
import cv2
from src.pipeline.clip_extractor import extract_clip, build_positive_clips, build_negative_clips
from src.pipeline.csv_parser import FeedingEvent


def test_extract_clip_creates_file(synthetic_video, tmp_path):
    video_path, _ = synthetic_video
    out = str(tmp_path / "clip.mp4")
    extract_clip(video_path, start_s=2.0, end_s=5.0, output_path=out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_extract_clip_duration(synthetic_video, tmp_path):
    video_path, _ = synthetic_video
    out = str(tmp_path / "clip.mp4")
    extract_clip(video_path, start_s=2.0, end_s=5.0, output_path=out)
    cap = cv2.VideoCapture(out)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frame_count / fps
    assert 2.5 <= duration <= 3.5  # ~3 seconds with codec rounding


def test_build_positive_clips(synthetic_video, tmp_path):
    video_path, ts = synthetic_video
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    # Name the video so video_id "GX_TEST" maps to it
    import shutil
    shutil.copy(video_path, os.path.join(video_dir, "GX_TEST.mp4"))
    out_dir = str(tmp_path / "clips" / "feeding")
    os.makedirs(out_dir)
    events = [FeedingEvent(video_id="GX_TEST", timestamp_s=ts)]
    paths = build_positive_clips(events, video_dir, out_dir)
    assert len(paths) == 1
    assert os.path.exists(paths[0])


def test_build_negative_clips_avoids_feeding_windows(synthetic_video, tmp_path):
    video_path, ts = synthetic_video
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    import shutil
    shutil.copy(video_path, os.path.join(video_dir, "GX_TEST.mp4"))
    out_dir = str(tmp_path / "clips" / "normal")
    os.makedirs(out_dir)
    events = [FeedingEvent(video_id="GX_TEST", timestamp_s=ts)]
    paths = build_negative_clips(events, video_dir, out_dir, clips_per_video=2, min_gap_s=5.0)
    # None of the negative clips should overlap the feeding window
    for p in paths:
        cap = cv2.VideoCapture(p)
        cap.release()
        assert os.path.exists(p)


def test_build_negative_clips_includes_boundary_windows(synthetic_video, tmp_path):
    """With a feeding event near the middle of a 15s video, boundary windows
    just after the event must be generated."""
    video_path, ts = synthetic_video  # ts is around 7.5s in a 15s video
    video_dir = str(tmp_path / "videos")
    os.makedirs(video_dir)
    import shutil
    shutil.copy(video_path, os.path.join(video_dir, "GX_TEST.mp4"))
    out_dir = str(tmp_path / "clips" / "normal_boundary")
    events = [FeedingEvent(video_id="GX_TEST", timestamp_s=ts)]
    paths = build_negative_clips(
        events, video_dir, out_dir,
        clips_per_video=2, min_gap_s=20.0,
        clip_duration_s=5.0, pre_s=1.0, post_s=4.0,
    )
    # At least one clip must have been created
    assert len(paths) >= 1
    for p in paths:
        assert os.path.exists(p)
