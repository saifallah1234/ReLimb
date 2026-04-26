"""
xml_loader.py
=============
Parses a ProGait-format CVAT XML annotation file and produces:

  1. keypoints.npy    — shape [N_frames, 66] float32
                        Same flat layout as limb_detection.py:
                        [lm0_x, lm0_y, lm1_x, lm1_y, ..., lm32_x, lm32_y]
                        ProGait's 23 keypoints are remapped to MediaPipe indices.
                        Unmapped slots stay NaN.

  2. detected_events.csv — heel-strike / toe-off events derived from the
                           ankle keypoint trajectories in the XML, using the
                           same signal-processing logic as limb_detection.py.

Usage (standalone):
    python xml_loader.py --xml path/to/annotations.xml \
                         --session_id 1_1_1_f \
                         [--fps 25.0]

Or import and call  load_xml_session()  from other modules.

Notes on the XML format
-----------------------
* The XML contains exactly ONE <track id="0" label="body-foot"> track.
  This track covers the single subject in the recording.
  There is NO bounding-box or person-ID field — the format assumes one
  subject per file.  If multiple people appear in the video, the XML only
  annotates the prosthetic-leg wearer; our code follows the same assumption.

* Keypoint labels are 1-indexed strings ("1" … "23").
  outside="1" means the point is not visible; we store NaN for those.

ProGait label → MediaPipe index mapping
----------------------------------------
ProGait  Body part          MediaPipe
  1      Left ear           7
  2      Right ear          8
  3      Nose               0
  4      Left eye           1
  5      Right eye          2
  6      Left shoulder      11
  7      Right shoulder     12
  8      Left elbow         13
  9      Right elbow        14
 10      (unused / wrist)   15  ← mapped conservatively
 11      Left hip           23
 12      Right hip          24
 13      Left knee          25
 14      Right knee         26
 15      (unused / shin)    — (NaN)
 16      Left ankle         27
 17      Right ankle        28
 18      Left heel          29
 19      Left foot index    31
 20      Left small toe     32
 21      Right heel         30
 22      Right foot index   31  ← shared slot, averaged if both visible
 23      Right small toe    32  ← shared slot, averaged if both visible
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks

# ── Project layout (same convention as limb_detection.py) ────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "data").exists() else SCRIPT_DIR
SESSION_DIR  = PROJECT_ROOT / "data" / "sessions"

# ── ProGait label (1-indexed str) → MediaPipe landmark index ─────────────────
# Values of None mean "no corresponding MediaPipe landmark; leave NaN"
PROGAIT_TO_MEDIAPIPE: dict[str, int | None] = {
    "1":  7,    # left ear
    "2":  8,    # right ear
    "3":  0,    # nose
    "4":  1,    # left eye (inner)
    "5":  2,    # right eye (inner)
    "6":  11,   # left shoulder
    "7":  12,   # right shoulder
    "8":  13,   # left elbow
    "9":  14,   # right elbow
    "10": 15,   # left wrist (approx)
    "11": 23,   # left hip
    "12": 24,   # right hip
    "13": 25,   # left knee
    "14": 26,   # right knee
    "15": None, # no good MediaPipe equivalent (shin mid-point)
    "16": 27,   # left ankle
    "17": 28,   # right ankle
    "18": 29,   # left heel
    "19": 31,   # left foot index
    "20": 32,   # left pinky toe (approx foot index slot)
    "21": 30,   # right heel
    "22": 31,   # right foot index  ← same slot as label 19; averaged
    "23": 32,   # right pinky toe   ← same slot as label 20; averaged
}

# Landmarks used by gait_algorithm.py for the event-detection signals
LM_LEFT_HIP    = 23
LM_LEFT_FOOT   = 31   # foot index
LM_RIGHT_HIP   = 24
LM_RIGHT_FOOT  = 32   # foot index (small-toe slot used as fallback)

# ── Signal-processing constants (mirror limb_detection.py) ───────────────────
BUTTER_ORDER          = 10
NORMALIZED_CUTOFF     = 0.1752
MIN_STEP_SEPARATION_S = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Core parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_xml(xml_path: Path) -> tuple[np.ndarray, int]:
    """
    Parse a ProGait CVAT XML file.

    Returns
    -------
    keypoints : np.ndarray, shape [N_frames, 66], dtype float32
        Flat MediaPipe-layout array. Unobserved / unmapped points = NaN.
    n_frames  : int
        Number of frames found in the XML (max frame index + 1).
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Find the skeleton track (label="body-foot", id="0")
    track = None
    for t in root.findall("track"):
        if t.get("label") == "body-foot":
            track = t
            break
    if track is None:
        raise ValueError(f"No 'body-foot' track found in {xml_path}")

    # First pass: find total frame count
    frames_in_xml = [int(sk.get("frame")) for sk in track.findall("skeleton")]
    if not frames_in_xml:
        raise ValueError(f"Track has no skeleton frames in {xml_path}")
    n_frames = max(frames_in_xml) + 1

    # Allocate output array: [N_frames, 66] initialised to NaN
    keypoints = np.full((n_frames, 66), np.nan, dtype=np.float32)

    # Accumulation buffer for landmarks that may be written by multiple
    # ProGait labels (labels 19+22 both map to MP slot 31, etc.)
    # We average them when both are visible.
    # Shape: [N_frames, 33, 2, 2] → axis-3: [sum, count]
    accum = np.zeros((n_frames, 33, 2, 2), dtype=np.float64)  # [..., 0]=sum, [...,1]=count

    for skeleton in track.findall("skeleton"):
        frame_idx = int(skeleton.get("frame"))
        for pt in skeleton.findall("points"):
            label    = pt.get("label")          # "1" … "23"
            outside  = pt.get("outside", "0")   # "1" = not visible
            if outside == "1":
                continue                        # treat as NaN

            mp_idx = PROGAIT_TO_MEDIAPIPE.get(label)
            if mp_idx is None:
                continue

            coords_str = pt.get("points", "")   # e.g. "957.29,220.11"
            try:
                x_str, y_str = coords_str.split(",")
                x, y = float(x_str), float(y_str)
            except ValueError:
                continue

            accum[frame_idx, mp_idx, 0, 0] += x
            accum[frame_idx, mp_idx, 0, 1] += y
            accum[frame_idx, mp_idx, 1, 0] += 1   # count x
            accum[frame_idx, mp_idx, 1, 1] += 1   # count y

    # Convert accumulation → averaged flat keypoints
    for frame_idx in range(n_frames):
        for mp_idx in range(33):
            cnt = accum[frame_idx, mp_idx, 1, 0]
            if cnt > 0:
                keypoints[frame_idx, mp_idx * 2]     = accum[frame_idx, mp_idx, 0, 0] / cnt  # x
                keypoints[frame_idx, mp_idx * 2 + 1] = accum[frame_idx, mp_idx, 0, 1] / cnt  # y

    return keypoints, n_frames


# ─────────────────────────────────────────────────────────────────────────────
# Event detection (mirrors limb_detection.py signal processing)
# ─────────────────────────────────────────────────────────────────────────────

def _interp_nan(x: np.ndarray) -> np.ndarray:
    n    = len(x)
    idx  = np.arange(n)
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return x
    return np.interp(idx, idx[mask], x[mask])


def _butter_lowpass(x: np.ndarray, order: int, Wn: float) -> np.ndarray:
    b, a = butter(order, Wn, btype="low", analog=False)
    return filtfilt(b, a, x)


def _detect_events(signal: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    min_dist = int(MIN_STEP_SEPARATION_S * fps)
    ht = 0.2 * (np.nanmax(signal) - np.nanmin(signal)) + np.nanmin(signal)
    peaks,   _ = find_peaks( signal, distance=min_dist, height= ht)
    troughs, _ = find_peaks(-signal, distance=min_dist, height=-ht)
    return peaks, troughs


def derive_events(keypoints: np.ndarray, fps: float) -> pd.DataFrame:
    """
    Run the same hip–foot signal processing as limb_detection.py on
    keypoints extracted from XML.

    Returns a DataFrame with columns:
        side, event, frame, time_s
    """
    lh = _interp_nan(keypoints[:, LM_LEFT_HIP  * 2])      # left  hip  x
    lf = _interp_nan(keypoints[:, LM_LEFT_FOOT * 2])      # left  foot x
    rh = _interp_nan(keypoints[:, LM_RIGHT_HIP  * 2])     # right hip  x
    rf = _interp_nan(keypoints[:, LM_RIGHT_FOOT * 2])     # right foot x

    left_signal  = (lh - lf) - np.nanmedian(lh - lf)
    right_signal = (rh - rf) - np.nanmedian(rh - rf)

    Wn = NORMALIZED_CUTOFF if NORMALIZED_CUTOFF < 1.0 else 0.45
    try:
        left_filt  = _butter_lowpass(left_signal,  BUTTER_ORDER, Wn)
        right_filt = _butter_lowpass(right_signal, BUTTER_ORDER, Wn)
    except Exception as e:
        print(f"  ⚠ Filter failed ({e}), using 5-frame moving average fallback")
        from numpy.lib.stride_tricks import sliding_window_view
        left_filt  = np.convolve(left_signal,  np.ones(5)/5, mode="same")
        right_filt = np.convolve(right_signal, np.ones(5)/5, mode="same")

    peaks_l,  troughs_l  = _detect_events(left_filt,  fps)
    peaks_r,  troughs_r  = _detect_events(right_filt, fps)

    rows: list[dict] = []
    for f in peaks_l:
        rows.append({"side": "left",  "event": "heel_strike", "frame": int(f), "time_s": f / fps})
    for f in troughs_l:
        rows.append({"side": "left",  "event": "toe_off",     "frame": int(f), "time_s": f / fps})
    for f in peaks_r:
        rows.append({"side": "right", "event": "heel_strike", "frame": int(f), "time_s": f / fps})
    for f in troughs_r:
        rows.append({"side": "right", "event": "toe_off",     "frame": int(f), "time_s": f / fps})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_xml_session(
    xml_path: Path | str,
    session_id: str,
    fps: float = 25.0,
    save: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Parse a ProGait XML file and produce the same outputs as
    limb_detection.process_video():

        <SESSION_DIR>/<session_id>/keypoints.npy
        <SESSION_DIR>/<session_id>/detected_events.csv

    Parameters
    ----------
    xml_path   : path to the CVAT XML annotation file
    session_id : string key for the session folder (e.g. "1_1_1_f")
    fps        : frame rate of the original video (needed for event timing)
    save       : if True, write outputs to SESSION_DIR/<session_id>/

    Returns
    -------
    keypoints : np.ndarray [N_frames, 66]
    events    : pd.DataFrame with columns side, event, frame, time_s
    """
    xml_path = Path(xml_path)
    print(f"▶ Loading XML: {xml_path.name}  (fps={fps})")

    keypoints, n_frames = parse_xml(xml_path)
    print(f"  Parsed {n_frames} frames, keypoints shape: {keypoints.shape}")

    events = derive_events(keypoints, fps)
    print(f"  Detected {len(events)} gait events")

    if save:
        out_dir = SESSION_DIR / session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        kp_path  = out_dir / "keypoints.npy"
        ev_path  = out_dir / "detected_events.csv"

        np.save(str(kp_path), keypoints)
        events.to_csv(ev_path, index=False)

        print(f"  💾 keypoints.npy     → {kp_path}")
        print(f"  💾 detected_events.csv → {ev_path}")

    return keypoints, events


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load a ProGait XML annotation into the ReLimb session format")
    parser.add_argument("--xml",        required=True,       help="Path to CVAT XML annotation file")
    parser.add_argument("--session_id", required=True,       help="Session ID string (e.g. 1_1_1_f)")
    parser.add_argument("--fps",        type=float, default=25.0, help="Video frame rate (default: 25.0)")
    parser.add_argument("--no-save",    action="store_true", help="Parse only; do not write output files")
    args = parser.parse_args()

    kp, ev = load_xml_session(
        xml_path   = args.xml,
        session_id = args.session_id,
        fps        = args.fps,
        save       = not args.no_save,
    )

    print(f"\nDone. keypoints shape: {kp.shape}, events: {len(ev)}")
    if not ev.empty:
        print(ev.sort_values("frame").head(10).to_string(index=False))


if __name__ == "__main__":
    main()