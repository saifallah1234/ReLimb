"""
make_clips.py
=============
Splits every video in data/raw_videos/hf/inside/ and outside/ into
fixed-length clips and writes them back into the same folder structure.

For each source video  e.g.  inside/1_1_1_f.mp4  it produces:
    inside/1_1_1_f_clip_000.mp4
    inside/1_1_1_f_clip_001.mp4
    ...

Each clip folder also gets copies of:
    <stem>_clip_NNN.json          ← same label as the parent video
    <stem>_clip_NNN_annotations.xml  ← same XML as the parent video
      (the XML frame indices are shifted so they correspond to the clip)

After running this script, re-run limb_detection.py + gait_algorithm.py
on the new clip videos as normal — they will produce detected_events.csv
and gait_metrics.json for each clip automatically.

Parameters (edit below)
-----------------------
WINDOW_FRAMES : int   — length of each clip in frames (default 150)
STEP_FRAMES   : int   — how far the window moves between clips (default 100)
                        step < window means clips overlap
                        step == window means no overlap (your "double" idea)
MIN_FRAMES    : int   — clips shorter than this are padded, not skipped (default 30)

Usage
-----
    python src/pose_extraction/make_clips.py
or with custom parameters:
    python src/pose_extraction/make_clips.py --window 150 --step 100
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# ── Project paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw_videos" / "hf"

# ── Default clip parameters ────────────────────────────────────────────────
WINDOW_FRAMES = 150
STEP_FRAMES   = 100   # change to 150 for non-overlapping (your "double" idea)
MIN_FRAMES    = 30    # clips with fewer real frames get zero-padded


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def get_clip_ranges(total_frames: int, window: int, step: int) -> list[tuple[int, int]]:
    """
    Return (start, end) frame ranges for each clip.
    The last clip is always anchored to the end so no frames are discarded.

    Example — 350 frames, window=150, step=100:
        (0, 150), (100, 250), (200, 350)
    """
    ranges = []
    start = 0
    while start + window <= total_frames:
        ranges.append((start, start + window))
        start += step

    # Final clip anchored to end (avoids discarding trailing frames)
    if ranges and ranges[-1][1] < total_frames:
        ranges.append((total_frames - window, total_frames))

    # Short video: single padded clip
    if not ranges:
        ranges.append((0, total_frames))

    return ranges


def write_clip_video(
    cap: cv2.VideoCapture,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    frame_size: tuple[int, int],
    pad_to: int,
) -> int:
    """
    Write frames [start_frame, end_frame) to out_path.
    If the clip is shorter than pad_to, duplicate the last frame to fill.
    Returns the number of real (non-padded) frames written.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_written = 0
    last_frame = None

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        last_frame = frame
        frames_written += 1

    # Pad with last frame if clip is shorter than window
    while frames_written < pad_to and last_frame is not None:
        writer.write(last_frame)
        frames_written += 1

    writer.release()
    return end_frame - start_frame   # real frame count (before padding)


def copy_json_for_clip(src_json: Path, dst_json: Path) -> None:
    """Copy the parent video's JSON label file to the clip's path."""
    shutil.copy2(src_json, dst_json)


def shift_xml_for_clip(src_xml: Path, dst_xml: Path, start_frame: int, window: int) -> None:
    """
    Copy the XML annotation, adjusting all frame= attributes by -start_frame
    so they align with the clip's local frame indices.

    Elements (skeleton or mask) whose shifted frame falls outside [0, window)
    are REMOVED entirely from the track — not just marked hidden — so that
    downstream tools only see exactly the frames that belong to this clip.
    """
    tree = ET.parse(str(src_xml))
    root = tree.getroot()

    for track in root.findall(".//track"):
        # Collect all direct child elements that carry a frame= attribute.
        # CVAT uses <skeleton> for pose tracks and <mask> for segmentation tracks.
        children_to_remove = []

        for child in list(track):
            frame_attr = child.get("frame")
            if frame_attr is None:
                continue  # future element types — leave untouched

            orig = int(frame_attr)
            new_frame = orig - start_frame

            if new_frame < 0 or new_frame >= window:
                # This annotation does not belong to the clip — drop it
                children_to_remove.append(child)
            else:
                child.set("frame", str(new_frame))
                # Mark every kept frame as a keyframe so downstream code
                # does not try to interpolate beyond the clip boundaries
                child.set("keyframe", "1")

        for child in children_to_remove:
            track.remove(child)

    # Update segment metadata to match the clip window
    for seg in root.iter("segment"):
        seg.set("start", "0")
        seg.set("stop", str(window - 1))

    # Update task size in meta so CVAT-style readers know the clip length
    for size_el in root.iter("size"):
        size_el.text = str(window)

    tree.write(str(dst_xml), encoding="utf-8", xml_declaration=True)


# ──────────────────────────────────────────────────────────────────────────
# Core: process one video
# ──────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    window: int,
    step: int,
) -> int:
    """
    Split one video into clips. Returns number of clips produced.
    """
    stem   = video_path.stem                   # e.g. "1_1_1_f"
    folder = video_path.parent                 # e.g. .../inside/

    # Matching annotation files (may not exist for all videos)
    json_src = folder / f"{stem}.json"
    xml_src  = folder / f"{stem}_annotations.xml"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open {video_path.name} — skipping")
        return 0

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < MIN_FRAMES:
        print(f"  ⚠ {video_path.name} only {total_frames} frames — skipping")
        cap.release()
        return 0

    ranges = get_clip_ranges(total_frames, window, step)
    print(f"  {video_path.name}  ({total_frames}f @ {fps:.0f}fps)  → {len(ranges)} clip(s)")

    for clip_idx, (start, end) in enumerate(ranges):
        tag      = f"clip_{clip_idx:03d}"
        clip_stem = f"{stem}_{tag}"

        clip_video = folder / f"{clip_stem}.mp4"
        clip_json  = folder / f"{clip_stem}.json"
        clip_xml   = folder / f"{clip_stem}_annotations.xml"

        # ── Video ─────────────────────────────────────────────────────────
        real_frames = write_clip_video(
            cap, clip_video, start, end, fps, (w, h), pad_to=window
        )
        print(f"    clip {clip_idx:03d}  frames {start}–{end}  → {clip_video.name}")

        # ── JSON label (copy unchanged) ────────────────────────────────────
        if json_src.exists():
            copy_json_for_clip(json_src, clip_json)
        else:
            print(f"    ⚠ No JSON found for {stem} — clip will have no label")

        # ── XML annotations (frame-shifted copy) ──────────────────────────
        if xml_src.exists():
            shift_xml_for_clip(xml_src, clip_xml, start_frame=start, window=window)
        else:
            print(f"    ⚠ No XML found for {stem} — clip has no skeleton annotations")

    cap.release()
    return len(ranges)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main(window: int, step: int) -> None:
    subsets = ["inside", "outside"]
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    total_clips = 0
    total_videos = 0

    for subset in subsets:
        subset_dir = RAW_DIR / subset
        if not subset_dir.exists():
            print(f"⚠ {subset_dir} not found — skipping")
            continue

        # Collect only SOURCE videos — skip any file that already looks like a clip
        videos = sorted([
            p for p in subset_dir.iterdir()
            if p.suffix.lower() in video_exts
            and "_clip_" not in p.stem          # don't re-clip existing clips
        ])

        print(f"\n── {subset.upper()}  ({len(videos)} source videos) ──")

        for vp in videos:
            n = process_video(vp, window, step)
            total_clips  += n
            total_videos += 1

    print(f"\n{'='*55}")
    print(f"Done.  {total_videos} source videos → {total_clips} clips")
    print(f"Window={window} frames, Step={step} frames")
    print(f"\nNext steps:")
    print(f"  1. Run limb_detection.py  — processes clip videos → keypoints.npy + detected_events.csv")
    print(f"  2. Run gait_algorithm.py  — computes gait_metrics.json for each clip session")
    print(f"  3. Re-run train_loocv.py  — now with ~{total_clips} training samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split ProGait videos into fixed-length clips")
    parser.add_argument("--window", type=int, default=WINDOW_FRAMES,
                        help=f"Clip length in frames (default: {WINDOW_FRAMES})")
    parser.add_argument("--step",   type=int, default=STEP_FRAMES,
                        help=f"Step between clip starts in frames (default: {STEP_FRAMES}). "
                             f"Set equal to --window for no overlap.")
    args = parser.parse_args()

    if args.step > args.window:
        print(f"⚠ step ({args.step}) > window ({args.window}) — some frames will be skipped entirely")

    print(f"ReLimb clip maker — window={args.window}, step={args.step}")
    print(f"Source directory : {RAW_DIR}\n")

    main(args.window, args.step)