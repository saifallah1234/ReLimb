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

# 🎯 STRICT PATHS: Pointing exactly to root/data/..., completely avoiding src/
DATA_DIR     = PROJECT_ROOT / 'data' / 'raw_videos' / 'hf'
SESSION_DIR  = PROJECT_ROOT / 'data' / 'sessions'
INCOMING_DIR = PROJECT_ROOT / 'data' / 'incoming'
RESULTS_DIR  = PROJECT_ROOT / 'data' / 'results'

# ── Parameters ─────────────────────────────────────────────────────────────
TEST_LIMIT            = None   # Set to None to run on all videos
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


# ── Core processing ────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    session_id: str,
    xml_path: Path | None = None,
    output_dir: Path = SESSION_DIR,
):
    # 1. 🌟 IF XML EXISTS: Use the high-quality smoothed CVAT annotations!
    if xml_path and xml_path.exists():
        print(f"  ⭐ XML found! Using high-quality smoothed annotations.")
        from src.pose_extraction.xml_loader import load_xml_session
        if output_dir != SESSION_DIR:
            print("  ⚠️ XML loader writes to data/sessions; output_dir will be ignored.")
        fps = 25.0
        cap = cv.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv.CAP_PROP_FPS) or 25.0
            cap.release()
            
        # This calls your newly updated xml_loader.py (which handles the overwrite natively)
        load_xml_session(xml_path, session_id, fps=fps, save=True)
        return

    # 2. 🤖 IF NO XML: Fallback to high-accuracy MediaPipe
    print(f"  🤖 No XML found. Falling back to AI extraction (MediaPipe).")
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open: {video_path}")
        return

    fps         = cap.get(cv.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    print(f"  FPS={fps:.1f}  frames={frame_count}")

    # Upgraded complexity to 2 for much better clinical tracking
    detector = PoseDetector(staticMode=False, modelComplexity=2, smoothLandmarks=True, 
                            enableSegmentation=False, detectionCon=0.6, trackCon=0.6)

    left_hip_x, left_foot_x, right_hip_x, right_foot_x = [], [], [], []
    keypoint_rows = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        lh_x = lf_x = rh_x = rf_x = np.nan
        kp_row = np.full(66, np.nan, dtype=np.float32)

        frame = cv.resize(frame, (640, 480))
        # draw=False makes extraction massively faster
        frame = detector.findPose(frame, draw=False)
        lmList, _ = detector.findPosition(frame, draw=False, bboxWithHands=False)

        if lmList:
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

    # 3. SIGNAL PROCESSING (For AI keypoints)
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

    # Make MediaPipe output closer to XML smoothing
    try:
        from src.pose_extraction.xml_loader import apply_progait_smoothing
        keypoints_array = apply_progait_smoothing(keypoints_array)
    except Exception as e:
        print(f"  ⚠️ Smoothing skipped: {e}")

    # 4. SAVE (Force Overwrite)
    session_folder = output_dir / session_id
    session_folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in peaks_left:   rows.append({'side': 'left',  'event': 'heel_strike', 'frame': int(f), 'time_s': f/fps})
    for f in troughs_left:  rows.append({'side': 'left',  'event': 'toe_off',     'frame': int(f), 'time_s': f/fps})
    for f in peaks_right:  rows.append({'side': 'right', 'event': 'heel_strike', 'frame': int(f), 'time_s': f/fps})
    for f in troughs_right: rows.append({'side': 'right', 'event': 'toe_off',     'frame': int(f), 'time_s': f/fps})

    pd.DataFrame(rows).to_csv(session_folder / 'detected_events.csv', index=False)
    np.save(str(session_folder / 'keypoints.npy'), keypoints_array)
    print(f"  💾 Overwritten MediaPipe Keypoints & Events to {session_folder}")


def derive_video_id(video_path: Path) -> str:
    return video_path.stem


def process_single_video(
    video_path: Path,
    video_id: str | None = None,
    output_dir: Path | None = None,
    xml_path: Path | None = None,
):
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = RESULTS_DIR
    if video_id is None:
        video_id = derive_video_id(video_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    process_video(video_path, video_id, xml_path=xml_path, output_dir=output_dir)
    return output_dir / video_id


def main():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # 1. Get all clips
    all_video_paths = sorted([p for p in DATA_DIR.rglob('*') if p.suffix.lower() in video_extensions])
    video_paths = [p for p in all_video_paths if "clip" in p.name.lower()]
    
    print(f"📂 Total clips found on disk: {len(video_paths)}")

    # APPLY TEST LIMIT
    if TEST_LIMIT is not None:
        video_paths = video_paths[:TEST_LIMIT]
        print(f"⚠️ TESTING MODE: Limiting execution to first {TEST_LIMIT} videos.")

    processed_count = 0

    for video_path in video_paths:
        relative_path = video_path.relative_to(DATA_DIR)
        session_id = "_".join(relative_path.with_suffix('').parts)
        
        # Note: Resume logic removed. It will overwrite every time now.

        print(f"\n▶ Processing ({processed_count + 1}/{len(video_paths)}): {video_path.name}")

        xml_name = video_path.stem + "_annotations.xml"
        xml_candidate = video_path.parent / xml_name
        xml_path = xml_candidate if xml_candidate.exists() else None
        
        try:
            process_video(video_path, session_id, xml_path=xml_path)
            processed_count += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n✅ Done! Successfully processed and overwritten: {processed_count} clips.")

if __name__ == '__main__':
    main()