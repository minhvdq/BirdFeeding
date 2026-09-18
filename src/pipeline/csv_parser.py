from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class FeedingEvent:
    video_id: str
    timestamp_s: float


def _parse_timestamp(ts: str) -> float:
    ts = str(ts).strip()
    if ":" in ts:
        mins, secs = ts.split(":", 1)
        return int(mins) * 60 + float(secs)
    return float(ts)


def _is_feeding_event(event: str) -> bool:
    # Substring match to handle field variations: "Feeding ", "Feeding?",
    # "Failed Feeding?", "Feeding Fail", "Stolen From", "Stealing"
    e = event.strip().lower()
    return "feeding" in e or "steal" in e


def parse_feeding_events(csv_path: str) -> list[FeedingEvent]:
    df = pd.read_csv(csv_path, dtype=str)
    # Use startswith("event") to avoid matching "Time (event start time)"
    event_col = [c for c in df.columns if c.lower().startswith("event")][0]
    video_col = [c for c in df.columns if "gopro" in c.lower() or "video id" in c.lower()][0]
    ts_col = [c for c in df.columns if "timestamp" in c.lower()][0]

    events: list[FeedingEvent] = []
    for _, row in df.iterrows():
        event = str(row[event_col])
        if not _is_feeding_event(event):
            continue
        video_id = str(row[video_col]).strip()
        ts_str = str(row[ts_col]).strip()
        if not ts_str or ts_str.lower() == "nan":
            continue
        events.append(FeedingEvent(video_id=video_id, timestamp_s=_parse_timestamp(ts_str)))
    return events
