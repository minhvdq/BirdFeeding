from src.pipeline.csv_parser import FeedingEvent, parse_feeding_events


def test_returns_only_feeding_rows(sample_csv):
    events = parse_feeding_events(sample_csv)
    assert len(events) == 3  # Feeding + Feeding Fail + Stealing; not Start/End


def test_video_id_parsed(sample_csv):
    events = parse_feeding_events(sample_csv)
    assert all(e.video_id == "GX_TEST" for e in events)


def test_timestamp_parsed_to_seconds(sample_csv):
    events = parse_feeding_events(sample_csv)
    timestamps = {round(e.timestamp_s) for e in events}
    assert 577 in timestamps   # 9:37 = 9*60+37
    assert 310 in timestamps   # 5:10 = 5*60+10
    assert 180 in timestamps   # 3:00


def test_returns_feeding_event_dataclass(sample_csv):
    events = parse_feeding_events(sample_csv)
    assert isinstance(events[0], FeedingEvent)
    assert hasattr(events[0], "video_id")
    assert hasattr(events[0], "timestamp_s")
