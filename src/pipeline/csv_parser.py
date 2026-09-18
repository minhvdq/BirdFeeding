from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class FeedingEvent:
    video_id: str
    timestamp_s: float


_KEEP = {"feeding", "feeding fail", "stealing"}


def _parse_timestamp(ts: str) -> float:
    ts = str(ts).strip()
    if ":" in ts:
        mins, secs = ts.split(":", 1)
        return int(mins) * 60 + float(secs)
    return float(ts)


def parse_feeding_events(csv_path: str) -> list[FeedingEvent]:
    df = pd.read_csv(csv_path, dtype=str)
    event_col = [c for c in df.columns if "event" in c.lower()][0]
    video_col = [c for c in df.columns if "gopro" in c.lower() or "video id" in c.lower()][0]
    ts_col = [c for c in df.columns if "timestamp" in c.lower()][0]

    events: list[FeedingEvent] = []
    for _, row in df.iterrows():
        event = str(row[event_col]).strip().lower()
        if event not in _KEEP:
            continue
        video_id = str(row[video_col]).strip()
        ts = _parse_timestamp(row[ts_col])
        events.append(FeedingEvent(video_id=video_id, timestamp_s=ts))
    return events
