#!/usr/bin/env python3
"""
Run after downloading a video from SharePoint to data/videos/.
Extracts clips and frames, uploads frames to Google Drive, then tells
you which video to download next.

Usage:
    python3 collect_data.py                  # extract + upload to Drive
    python3 collect_data.py --no-drive       # extract only, skip Drive
    python3 collect_data.py --dry-run        # preview, no changes
    python3 collect_data.py --status         # show progress only

First-time Drive setup:
    1. Go to https://console.cloud.google.com/
    2. Create a project → Enable "Google Drive API"
    3. Create OAuth credentials (Desktop App) → Download JSON
    4. Save it as:  credentials/client_secrets.json
    A browser window will open on first run to authorize access.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "data", "Feeding Data(Sheet1).csv")
VIDEO_DIR = os.path.join(ROOT, "data", "videos")
CLIPS_FEEDING = os.path.join(ROOT, "data", "clips", "feeding")
CLIPS_NORMAL = os.path.join(ROOT, "data", "clips", "normal")
FRAMES_FEEDING = os.path.join(ROOT, "data", "frames", "feeding")
FRAMES_NORMAL = os.path.join(ROOT, "data", "frames", "normal")

CREDENTIALS_DIR = os.path.join(ROOT, "credentials")
CLIENT_SECRETS = os.path.join(CREDENTIALS_DIR, "client_secrets.json")
TOKEN_FILE = os.path.join(CREDENTIALS_DIR, "drive_token.json")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_ROOT_FOLDER = "BirdFeeding"


# ── Google Drive ──────────────────────────────────────────────────────────────

class DriveUploader:
    def __init__(self) -> None:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, DRIVE_SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, DRIVE_SCOPES)
                creds = flow.run_local_server(port=0)
            os.makedirs(CREDENTIALS_DIR, exist_ok=True)
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())

        self._service = build("drive", "v3", credentials=creds)
        self._folder_cache: dict[str, str] = {}

    def _get_or_create_folder(self, name: str, parent_id: str | None = None) -> str:
        cache_key = f"{parent_id}/{name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = self._service.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])

        if files:
            folder_id = files[0]["id"]
        else:
            meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
            if parent_id:
                meta["parents"] = [parent_id]
            folder_id = self._service.files().create(body=meta, fields="id").execute()["id"]

        self._folder_cache[cache_key] = folder_id
        return folder_id

    def upload_frame_dir(self, local_dir: str, label: str) -> int:
        """Upload all JPEGs in local_dir to Drive under BirdFeeding/frames/<label>/<dirname>/."""
        from googleapiclient.http import MediaFileUpload

        clip_name = os.path.basename(local_dir)
        root_id = self._get_or_create_folder(DRIVE_ROOT_FOLDER)
        frames_id = self._get_or_create_folder("frames", root_id)
        label_id = self._get_or_create_folder(label, frames_id)
        clip_id = self._get_or_create_folder(clip_name, label_id)

        jpgs = sorted(f for f in os.listdir(local_dir) if f.endswith(".jpg"))
        for jpg in jpgs:
            path = os.path.join(local_dir, jpg)
            media = MediaFileUpload(path, mimetype="image/jpeg")
            self._service.files().create(
                body={"name": jpg, "parents": [clip_id]},
                media_body=media,
                fields="id",
            ).execute()
        return len(jpgs)


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _video_ids_in_dir(frames_dir: str) -> set[str]:
    if not os.path.isdir(frames_dir):
        return set()
    ids: set[str] = set()
    for name in os.listdir(frames_dir):
        if os.path.isdir(os.path.join(frames_dir, name)):
            ids.add(name.split("_")[0])
    return ids


def _videos_in_dir(video_dir: str) -> list[str]:
    if not os.path.isdir(video_dir):
        return []
    return [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".mov")) and not f.startswith(".")
    ]


def _video_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].upper()


def _save_frames(clip_path: str, frames_dir: str, dry_run: bool) -> tuple[int, str]:
    """Returns (frame_count, out_dir)."""
    from src.pipeline.frame_extractor import process_clip

    clip_stem = os.path.splitext(os.path.basename(clip_path))[0]
    out_dir = os.path.join(frames_dir, clip_stem)
    if dry_run:
        print(f"    [dry-run] would save frames → {out_dir}/")
        return 10, out_dir
    os.makedirs(out_dir, exist_ok=True)
    frames = process_clip(clip_path)
    if len(frames) == 0:
        print(f"    WARNING: no frames extracted from {clip_stem}")
        return 0, out_dir
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.jpg"), frame)
    return len(frames), out_dir


def process_video(video_path: str, all_events, dry_run: bool, uploader: DriveUploader | None) -> None:
    from src.pipeline.clip_extractor import build_positive_clips, build_negative_clips

    video_id = _video_id_from_path(video_path)
    video_events = [e for e in all_events if e.video_id.upper() == video_id]

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"  Feeding events in CSV: {len(video_events)}")

    if not video_events:
        print("  No feeding events found for this video — skipping.")
        return

    print("  Extracting positive clips...")
    if dry_run:
        pos_clips = [f"[dry] {video_id}_feed_{i:04d}.mp4" for i in range(len(video_events))]
        print(f"    [dry-run] would extract {len(pos_clips)} clip(s)")
    else:
        pos_clips = build_positive_clips(video_events, VIDEO_DIR, CLIPS_FEEDING)
        print(f"    Extracted {len(pos_clips)} feeding clip(s)")

    print("  Extracting negative clips...")
    if dry_run:
        neg_clips = [f"[dry] {video_id}_normal_000{i}.mp4" for i in range(4)]
        print(f"    [dry-run] would extract up to 4 normal clip(s)")
    else:
        neg_clips = build_negative_clips(video_events, VIDEO_DIR, CLIPS_NORMAL)
        print(f"    Extracted {len(neg_clips)} normal clip(s)")

    print("  Extracting + uploading feeding frames...")
    for clip in pos_clips:
        n, out_dir = _save_frames(clip, FRAMES_FEEDING, dry_run)
        label = os.path.basename(out_dir)
        if uploader and not dry_run and n > 0:
            uploaded = uploader.upload_frame_dir(out_dir, "feeding")
            print(f"    {label}: {n} frames saved, {uploaded} uploaded to Drive")
        else:
            print(f"    {label}: {n} frames {'would be saved' if dry_run else 'saved'}")

    print("  Extracting + uploading normal frames...")
    for clip in neg_clips:
        n, out_dir = _save_frames(clip, FRAMES_NORMAL, dry_run)
        label = os.path.basename(out_dir)
        if uploader and not dry_run and n > 0:
            uploaded = uploader.upload_frame_dir(out_dir, "normal")
            print(f"    {label}: {n} frames saved, {uploaded} uploaded to Drive")
        else:
            print(f"    {label}: {n} frames {'would be saved' if dry_run else 'saved'}")

    print(f"\n  Done with {video_id}.")
    if not dry_run:
        print(f"  You can now delete data/videos/{os.path.basename(video_path)}")


# ── Status ────────────────────────────────────────────────────────────────────

def print_status(all_events) -> tuple[set[str], list[str]]:
    from collections import Counter

    all_video_ids = sorted({e.video_id.upper() for e in all_events})
    done_ids = {v.upper() for v in (_video_ids_in_dir(FRAMES_FEEDING) | _video_ids_in_dir(FRAMES_NORMAL))}

    remaining = [vid for vid in all_video_ids if vid not in done_ids]
    done = [vid for vid in all_video_ids if vid in done_ids]
    events_by_video = Counter(e.video_id.upper() for e in all_events)

    print(f"\n{'='*60}")
    print("DATA COLLECTION STATUS")
    print(f"{'='*60}")
    print(f"  Total videos in CSV:  {len(all_video_ids)}")
    print(f"  Already processed:    {len(done)}")
    print(f"  Still needed:         {len(remaining)}")

    if done:
        print(f"\n  Processed ({len(done)}):")
        for vid in done:
            print(f"    [x] {vid}  ({events_by_video[vid]} feeding events)")

    if remaining:
        print(f"\n  Still needed ({len(remaining)}):")
        for vid in remaining:
            print(f"    [ ] {vid}  ({events_by_video[vid]} feeding events)")
        print(f"\n  Next to download:  {remaining[0]}")

    print()
    return done_ids, remaining


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect training data one video at a time.")
    parser.add_argument("--no-drive", action="store_true", help="Skip Google Drive upload")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--status", action="store_true", help="Show progress only")
    args = parser.parse_args()

    sys.path.insert(0, ROOT)

    from src.pipeline.csv_parser import parse_feeding_events

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)

    all_events = parse_feeding_events(CSV_PATH)
    print(f"Loaded {len(all_events)} unique feeding events from CSV.")

    done_ids, remaining = print_status(all_events)

    if args.status:
        return

    use_drive = not args.no_drive and not args.dry_run
    uploader: DriveUploader | None = None
    if use_drive:
        if not os.path.exists(CLIENT_SECRETS):
            print(
                "Drive upload skipped — credentials/client_secrets.json not found.\n"
                "See the setup instructions at the top of this file, or run with --no-drive."
            )
            use_drive = False
        else:
            print("Connecting to Google Drive...")
            uploader = DriveUploader()
            print("Connected.\n")

    available_videos = _videos_in_dir(VIDEO_DIR)
    unprocessed_videos = [v for v in available_videos if _video_id_from_path(v) not in done_ids]

    if not unprocessed_videos:
        if available_videos:
            print("All videos in data/videos/ are already processed.")
        else:
            print("No videos found in data/videos/.")
        if remaining:
            print(f"Next video to download from SharePoint: {remaining[0]}")
        return

    for video_path in unprocessed_videos:
        process_video(video_path, all_events, dry_run=args.dry_run, uploader=uploader)

    print(f"\n{'='*60}")
    print("COLLECTION ROUND COMPLETE")

    done_ids_after = {v.upper() for v in (_video_ids_in_dir(FRAMES_FEEDING) | _video_ids_in_dir(FRAMES_NORMAL))}
    all_video_ids = sorted({e.video_id.upper() for e in all_events})
    remaining_after = [vid for vid in all_video_ids if vid not in done_ids_after]

    feed_count = sum(1 for e in os.scandir(FRAMES_FEEDING) if e.is_dir()) if os.path.isdir(FRAMES_FEEDING) else 0
    norm_count = sum(1 for e in os.scandir(FRAMES_NORMAL) if e.is_dir()) if os.path.isdir(FRAMES_NORMAL) else 0
    print(f"  Feeding clip dirs: {feed_count}")
    print(f"  Normal clip dirs:  {norm_count}")

    if remaining_after:
        print(f"\n  Next video to download:  {remaining_after[0]}  ({len(remaining_after)} remaining)")
    else:
        print("\n  All videos processed! Frames are on Google Drive — ready to train on Colab.")


if __name__ == "__main__":
    main()
