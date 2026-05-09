"""
xml_loader.py
=============
Parses a ProGait-format CVAT XML annotation file and produces:

  1. keypoints.npy    — shape [N_frames, 66] float32
                        Mapped to MediaPipe slots. Smoothed using ProGait's 
                        original spline algorithm.
  2. detected_events.csv — heel-strike / toe-off events derived from the
                           smoothed ankle keypoint trajectories.
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import make_smoothing_spline  # <--- Added from original repo

# ── Project layout ──────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SESSION_DIR  = PROJECT_ROOT / "data" / "sessions"

# ── ProGait label (1-indexed str) → MediaPipe landmark index ─────────────────
PROGAIT_TO_MEDIAPIPE: dict[str, int | None] = {
    "6":  12,   # left shoulder
    "7":  11,   # right shoulder
    "8":  14,   # left elbow
    "9":  13,   # right elbow
    "12": 24,   # left hip
    "13": 23,   # right knee
    "14": 26,   # left knee
    "15": 25, # shin mid-point
    "16": 28,   # left ankle
    "17": 27,   # right ankle
    "18": 30,   # left heel
    "19": 32,   # left foot index
    "21": 29,   # right heel
    "22": 31,   # right foot index (FIXED: Was cross-wired to 31!)

}

LM_LEFT_HIP    = 23
LM_LEFT_FOOT   = 31
LM_RIGHT_HIP   = 24
LM_RIGHT_FOOT  = 32

BUTTER_ORDER          = 10
NORMALIZED_CUTOFF     = 0.1752
MIN_STEP_SEPARATION_S = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing Logic (Direct from ProGait Authors)
# ─────────────────────────────────────────────────────────────────────────────

def apply_progait_smoothing(keypoints_66: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """
    Applies the original make_smoothing_spline fix to remove human annotation jitter.
    Leaves completely untracked points (faces/hands) as NaN so they don't break normalization.
    """
    # 1. Fill missing frames using pandas, BUT leave entirely empty columns as NaN
    df = pd.DataFrame(keypoints_66)
    df = df.interpolate(method='linear', limit_direction='both')
    
    # REMOVED .fillna(0.0) so unused points stay as NaN!
    df = df.bfill().ffill() 
    clean_kp = df.to_numpy(dtype=np.float32)

    # 2. Apply Spline Smoothing
    t = np.arange(clean_kp.shape[0])
    
    # Start with an array entirely filled with NaNs
    smoothed = np.full_like(clean_kp, np.nan, dtype=np.float32)

    for i in range(66):
        # Only smooth if the column actually has real data (is not completely NaN)
        if not np.all(np.isnan(clean_kp[:, i])):
            spline = make_smoothing_spline(t, clean_kp[:, i], lam=lam)
            smoothed[:, i] = spline(t)

    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Core parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_xml(xml_path: Path) -> tuple[np.ndarray, int]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    track = None
    for t in root.findall("track"):
        if t.get("label") == "body-foot":
            track = t
            break
    if track is None:
        raise ValueError(f"No 'body-foot' track found in {xml_path}")

    frames_in_xml = [int(sk.get("frame")) for sk in track.findall("skeleton")]
    if not frames_in_xml:
        raise ValueError(f"Track has no skeleton frames in {xml_path}")
    n_frames = max(frames_in_xml) + 1

    keypoints = np.full((n_frames, 66), np.nan, dtype=np.float32)
    accum = np.zeros((n_frames, 33, 2, 2), dtype=np.float64)

    for skeleton in track.findall("skeleton"):
        frame_idx = int(skeleton.get("frame"))
        for pt in skeleton.findall("points"):
            label    = pt.get("label")
            outside  = pt.get("outside", "0")
            if outside == "1":
                continue

            mp_idx = PROGAIT_TO_MEDIAPIPE.get(label)
            if mp_idx is None:
                continue

            coords_str = pt.get("points", "")
            try:
                x_str, y_str = coords_str.split(",")
                x, y = float(x_str), float(y_str)
            except ValueError:
                continue

            accum[frame_idx, mp_idx, 0, 0] += x
            accum[frame_idx, mp_idx, 0, 1] += y
            accum[frame_idx, mp_idx, 1, 0] += 1
            accum[frame_idx, mp_idx, 1, 1] += 1

    for frame_idx in range(n_frames):
        for mp_idx in range(33):
            cnt = accum[frame_idx, mp_idx, 1, 0]
            if cnt > 0:
                keypoints[frame_idx, mp_idx * 2]     = accum[frame_idx, mp_idx, 0, 0] / cnt
                keypoints[frame_idx, mp_idx * 2 + 1] = accum[frame_idx, mp_idx, 0, 1] / cnt

    # ----> APPLY THE SMOOTHING FIX HERE <----
    keypoints = apply_progait_smoothing(keypoints)

    return keypoints, n_frames


# ─────────────────────────────────────────────────────────────────────────────
# Event detection
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
    lh = _interp_nan(keypoints[:, LM_LEFT_HIP  * 2])
    lf = _interp_nan(keypoints[:, LM_LEFT_FOOT * 2])
    rh = _interp_nan(keypoints[:, LM_RIGHT_HIP  * 2])
    rf = _interp_nan(keypoints[:, LM_RIGHT_FOOT * 2])

    left_signal  = (lh - lf) - np.nanmedian(lh - lf)
    right_signal = (rh - rf) - np.nanmedian(rh - rf)

    Wn = NORMALIZED_CUTOFF if NORMALIZED_CUTOFF < 1.0 else 0.45
    try:
        left_filt  = _butter_lowpass(left_signal,  BUTTER_ORDER, Wn)
        right_filt = _butter_lowpass(right_signal, BUTTER_ORDER, Wn)
    except Exception as e:
        left_filt  = np.convolve(left_signal,  np.ones(5)/5, mode="same")
        right_filt = np.convolve(right_signal, np.ones(5)/5, mode="same")

    peaks_l,  troughs_l  = _detect_events(left_filt,  fps)
    peaks_r,  troughs_r  = _detect_events(right_filt, fps)

    rows = []
    for f in peaks_l:
        rows.append({"side": "left",  "event": "heel_strike", "frame": int(f), "time_s": f / fps})
    for f in troughs_l:
        rows.append({"side": "left",  "event": "toe_off",     "frame": int(f), "time_s": f / fps})
    for f in peaks_r:
        rows.append({"side": "right", "event": "heel_strike", "frame": int(f), "time_s": f / fps})
    for f in troughs_r:
        rows.append({"side": "right", "event": "toe_off",     "frame": int(f), "time_s": f / fps})

    return pd.DataFrame(rows)


def load_xml_session(xml_path, session_id, fps=25.0, save=True):
    xml_path = Path(xml_path)
    print(f"▶ Loading XML: {xml_path.name}  (fps={fps})")

    keypoints, n_frames = parse_xml(xml_path)
    print(f"  Parsed {n_frames} frames, smoothed shape: {keypoints.shape}")

    events = derive_events(keypoints, fps)
    print(f"  Detected {len(events)} gait events")

    if save:
        out_dir = SESSION_DIR / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        kp_path  = out_dir / "keypoints.npy"
        ev_path  = out_dir / "detected_events.csv"
        np.save(str(kp_path), keypoints)
        events.to_csv(ev_path, index=False)

    return keypoints, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",        required=True)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--fps",        type=float, default=25.0)
    parser.add_argument("--no-save",    action="store_true")
    args = parser.parse_args()

    kp, ev = load_xml_session(args.xml, args.session_id, args.fps, not args.no_save)
    print(f"\nDone. keypoints shape: {kp.shape}, events: {len(ev)}")

if __name__ == "__main__":
    main()