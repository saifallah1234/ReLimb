import cv2 as cv
import numpy as np
import pandas as pd
from cvzone.PoseModule import PoseDetector
from scipy.signal import butter, filtfilt, find_peaks
import os
from pathlib import Path

# ── Project layout ─────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR    = PROJECT_ROOT / 'data' / 'raw_videos' / 'hf'
SESSION_DIR = PROJECT_ROOT / 'data' / 'sessions'

# ── Parameters ─────────────────────────────────────────────────────────────
T_TOLERANCE           = 0.2
MIN_STEP_SEPARATION_S = 0.8
BUTTER_ORDER          = 10
NORMALIZED_CUTOFF     = 0.1752

# ── Signal processing ──────────────────────────────────────────────────────

def butter_lowpass_filter(x, order, Wn):
    b, a = butter(order, Wn, btype='low', analog=False)
    return filtfilt(b, a, x)

def interp_nan(x):
    n    = len(x)
    idx  = np.arange(n)
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return x
    return np.interp(idx, idx[mask], x[mask])

def detect_events_from_signal(signal, fps, min_separation_s=MIN_STEP_SEPARATION_S, height_threshold=None):
    min_dist = int(min_separation_s * fps)
    if height_threshold is None:
        height_threshold = (0.2 * (np.nanmax(signal) - np.nanmin(signal)) + np.nanmin(signal))
    peaks,   _ = find_peaks( signal, distance=min_dist, height= height_threshold)
    troughs, _ = find_peaks(-signal, distance=min_dist, height=-height_threshold)
    return peaks, troughs

# ── XML keypoint overlay ──────────────────────────────────────────────────

def _apply_xml_overlay(keypoints_mp: np.ndarray, xml_path: Path) -> np.ndarray:
    try:
        from xml_loader import parse_xml
    except ImportError:
        print("  ⚠ xml_loader not found — skipping XML overlay")
        return keypoints_mp

    kp_xml, n_xml = parse_xml(xml_path)
    n_mp  = keypoints_mp.shape[0]

    if n_xml != n_mp:
        print(f"  ⚠ Frame count mismatch: video={n_mp}, XML={n_xml}. Truncating.")
        min_n = min(n_mp, n_xml)
        keypoints_mp = keypoints_mp[:min_n]
        kp_xml       = kp_xml[:min_n]

    valid_mask = ~np.isnan(kp_xml)
    keypoints_mp[valid_mask] = kp_xml[valid_mask]
    print(f"  ✓ XML overlay applied: {valid_mask.sum()} values updated")
    return keypoints_mp

# ── Core processing ────────────────────────────────────────────────────────

def process_video(video_path: Path, session_id: str, xml_path: Path | None = None, xml_only: bool = False):
    if xml_only and xml_path is not None and xml_path.exists():
        print(f"  ℹ xml_only=True — skipping MediaPipe")
        from xml_loader import load_xml_session
        fps = 25.0
        cap = cv.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv.CAP_PROP_FPS) or 25.0
            cap.release()
        load_xml_session(xml_path, session_id, fps=fps, save=True)
        return

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open: {video_path}")
        return

    fps         = cap.get(cv.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    print(f"  FPS={fps:.1f}  frames={frame_count}")

    detector = PoseDetector(staticMode=False, modelComplexity=1, smoothLandmarks=True, 
                            enableSegmentation=False, detectionCon=0.5, trackCon=0.5)

    left_hip_x, left_foot_x, right_hip_x, right_foot_x = [], [], [], []
    keypoint_rows = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 1. PRE-INITIALIZE (Fixes the "Undefined" error)
        lh_x = lf_x = rh_x = rf_x = np.nan
        kp_row = np.full(66, np.nan, dtype=np.float32)

        frame = cv.resize(frame, (640, 480))
        frame = detector.findPose(frame)
        lmList, _ = detector.findPosition(frame, draw=False, bboxWithHands=False)

        # 2. EXTRACT
        if lmList:
            # Fill the 66-column row
            for lm_i in range(min(len(lmList), 33)):
                kp_row[lm_i * 2]     = lmList[lm_i][0]
                kp_row[lm_i * 2 + 1] = lmList[lm_i][1]
            
            try:
                lh_x = lmList[23][0]
                rh_x = lmList[24][0]
                lf_x = lmList[31][0] if len(lmList) > 31 else lmList[27][0]
                rf_x = lmList[32][0] if len(lmList) > 32 else lmList[28][0]
            except Exception: pass

        left_hip_x.append(lh_x)
        left_foot_x.append(lf_x)
        right_hip_x.append(rh_x)
        right_foot_x.append(rf_x)
        keypoint_rows.append(kp_row)

    cap.release()

    # 3. SIGNAL PROCESSING
    lh = interp_nan(np.array(left_hip_x,   dtype=np.float64))
    lf = interp_nan(np.array(left_foot_x,  dtype=np.float64))
    rh = interp_nan(np.array(right_hip_x,  dtype=np.float64))
    rf = interp_nan(np.array(right_foot_x, dtype=np.float64))

    left_signal  = (lh - lf) - np.nanmedian(lh - lf)
    right_signal = (rh - rf) - np.nanmedian(rh - rf)

    Wn = NORMALIZED_CUTOFF if NORMALIZED_CUTOFF < 1.0 else 0.45
    try:
        left_filt  = butter_lowpass_filter(left_signal,  BUTTER_ORDER, Wn)
        right_filt = butter_lowpass_filter(right_signal, BUTTER_ORDER, Wn)
    except Exception:
        left_filt, right_filt = left_signal, right_signal

    peaks_left,   troughs_left  = detect_events_from_signal(left_filt,  fps)
    peaks_right,  troughs_right = detect_events_from_signal(right_filt, fps)

    keypoints_array = np.stack(keypoint_rows, axis=0).astype(np.float32)
    if xml_path and xml_path.exists():
        keypoints_array = _apply_xml_overlay(keypoints_array, xml_path)

    # 4. SAVE
    session_folder = SESSION_DIR / session_id
    session_folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in peaks_left:   rows.append({'side': 'left',  'event': 'heel_strike', 'frame': int(f), 'time_s': f/fps})
    for f in troughs_left:  rows.append({'side': 'left',  'event': 'toe_off',     'frame': int(f), 'time_s': f/fps})
    for f in peaks_right:  rows.append({'side': 'right', 'event': 'heel_strike', 'frame': int(f), 'time_s': f/fps})
    for f in troughs_right: rows.append({'side': 'right', 'event': 'toe_off',     'frame': int(f), 'time_s': f/fps})

    pd.DataFrame(rows).to_csv(session_folder / 'detected_events.csv', index=False)
    np.save(str(session_folder / 'keypoints.npy'), keypoints_array)

def main():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_paths = sorted([p for p in DATA_DIR.rglob('*') if p.suffix.lower() in video_extensions])

    if not video_paths: return

    for video_path in video_paths:
        relative_path = video_path.relative_to(DATA_DIR)
        session_id = "_".join(relative_path.with_suffix('').parts)
        
        print(f"▶ {relative_path}  →  session: {session_id}")

        # --- UPDATED LOGIC HERE ---
        # Instead of just .xml, look for [video_name]_annotations.xml
        xml_name = video_path.stem + "_annotations.xml"
        xml_candidate = video_path.parent / xml_name
        
        xml_path = xml_candidate if xml_candidate.exists() else None
        
        if xml_path:
            print(f"   ✓ Found matching annotation: {xml_name}")
        # --------------------------
        
        try:
            process_video(video_path, session_id, xml_path=xml_path)
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == '__main__':
    main()