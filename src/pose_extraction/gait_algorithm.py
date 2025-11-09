import cv2 as cv
import numpy as np
import pandas as pd
from cvzone.PoseModule import PoseDetector
from scipy.signal import butter, filtfilt, find_peaks
import math
import os
from datetime import datetime

# ---------- PARAMETERS ----------
DATA_DIR = 'data/raw_videos'
SESSION_DIR = 'data/sessions'
#GROUND_TRUTH_CSV = 'data/ground_truth_events.csv'  # optional: CSV with column 'time_s' or 'frame'
T_TOLERANCE = 0.2  # seconds tolerance for matching events
MIN_STEP_SEPARATION_S = 0.8  # minimum seconds between successive steps (paper uses 0.8 s)
BUTTER_ORDER = 10
NORMALIZED_CUTOFF = 0.1752  # from paper (Wn for scipy): you may adjust depending on fps
# --------------------------------

#The Butterworth filter is an analogue filter design which produces the best output response with no ripple in the pass band or the stop band resulting in a maximally flat filter response but at the expense of a relatively wide transition band.
def butter_lowpass_filter(x, order, Wn):
    b, a = butter(order, Wn, btype='low', analog=False)
    y = filtfilt(b, a, x)
    return y

def match_events_to_ground_truth(detected_times, gt_times, tol=T_TOLERANCE):
    detected_times = np.array(detected_times)
    gt_times = np.array(gt_times)
    matched = []
    used_gt = set()
    for d in detected_times:
        # find nearest GT event
        diffs = np.abs(gt_times - d)
        idx = np.argmin(diffs) if len(diffs)>0 else None
        if idx is not None and diffs[idx] <= tol and idx not in used_gt:
            matched.append((d, gt_times[idx], diffs[idx]))
            used_gt.add(idx)
    unmatched_gt = [gt_times[i] for i in range(len(gt_times)) if i not in used_gt]
    unmatched_det = [d for d in detected_times if not any(abs(d - m[0]) <= tol for m in matched)]
    return matched, unmatched_det, unmatched_gt

def detect_events_from_signal(signal, fps, min_separation_s=MIN_STEP_SEPARATION_S, height_threshold=None):
    # Peaks for heel-strike, minima for toe-off
    min_dist_frames = int(min_separation_s * fps)
    # Peaks
    if height_threshold is None:
        height_threshold = 0.2 * (np.nanmax(signal) - np.nanmin(signal)) + np.nanmin(signal)
    peaks, _ = find_peaks(signal, distance=min_dist_frames, height=height_threshold)
    # Minima: find peaks on inverted signal
    inv_signal = -signal
    troughs, _ = find_peaks(inv_signal, distance=min_dist_frames, height=-height_threshold)
    return peaks, troughs

def process_video(VIDEO_PATH,session_id):
    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video:", VIDEO_PATH)
        return

    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / fps if fps>0 else 0
    print(f"Video FPS={fps:.2f}, frames={frame_count}, duration={duration_s:.2f}s")

    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            detectionCon=0.5,
                            trackCon=0.5)

    # buffers for x positions (NaN for missing)
    left_hip_x = []
    left_foot_x = []
    right_hip_x = []
    right_foot_x = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = cv.resize(frame, (640, 480))
        frame = detector.findPose(frame)
        lmList, bboxInfo = detector.findPosition(frame, draw=False, bboxWithHands=False)
        # default NaNs
        lh_x = np.nan; lf_x = np.nan; rh_x = np.nan; rf_x = np.nan
        if lmList and len(lmList) >= 33:
            # MediaPipe indices mapping in cvzone: 
            # left hip = 23, left ankle = 27 (or foot index 31), right hip=24, right ankle=28 (or foot index 32)
            try:
                lh_x = lmList[23][0]
                lf_x = lmList[31][0]  # foot index (if available)
                rh_x = lmList[24][0]
                rf_x = lmList[32][0]
            except Exception:
                # fallback to ankle if foot index not available
                lf_x = lmList[27][0] if len(lmList) > 27 else np.nan
                rf_x = lmList[28][0] if len(lmList) > 28 else np.nan

        left_hip_x.append(lh_x)
        left_foot_x.append(lf_x)
        right_hip_x.append(rh_x)
        right_foot_x.append(rf_x)

    cap.release()
    print("Extraction finished. Frames processed:", frame_idx)

    # convert to numpy arrays
    lh = np.array(left_hip_x, dtype=np.float64)
    lf = np.array(left_foot_x, dtype=np.float64)
    rh = np.array(right_hip_x, dtype=np.float64)
    rf = np.array(right_foot_x, dtype=np.float64)

    # Build horizontal relative distance signals (hip_x - foot_x)
    left_signal = lh - lf
    right_signal = rh - rf

    # Interpolate small NaN gaps
    def interp_nan(x):
        n = len(x)
        idx = np.arange(n)
        mask = ~np.isnan(x)
        if mask.sum() < 2:
            return x  # not enough to interpolate
        return np.interp(idx, idx[mask], x[mask])
    left_signal = interp_nan(left_signal)
    right_signal = interp_nan(right_signal)

    # Detrend (optional) - remove linear trend
    left_signal = left_signal - np.nanmedian(left_signal)
    right_signal = right_signal - np.nanmedian(right_signal)

    # Butterworth filter
    # If you want to recompute normalized Wn from desired cutoff Hz:
    # desired_cutoff_hz = 2.2  # example based on paper
    # Wn = desired_cutoff_hz / (fps/2)
    Wn = NORMALIZED_CUTOFF
    if Wn >= 1.0:
        Wn = 0.45  # fallback safe value
    try:
        left_filt = butter_lowpass_filter(left_signal, BUTTER_ORDER, Wn)
        right_filt = butter_lowpass_filter(right_signal, BUTTER_ORDER, Wn)
    except Exception as e:
        print("Filter failed (maybe not enough samples or bad Wn). Falling back to small smoothing.")
        left_filt = cv.blur(left_signal.reshape(-1,1), (5,1)).reshape(-1)
        right_filt = cv.blur(right_signal.reshape(-1,1), (5,1)).reshape(-1)

    # Detect peaks/minima
    peaks_left, troughs_left = detect_events_from_signal(left_filt, fps)
    peaks_right, troughs_right = detect_events_from_signal(right_filt, fps)

    # Convert frame indices to times
    """times_peaks_left = peaks_left / fps
    times_troughs_left = troughs_left / fps
    times_peaks_right = peaks_right / fps
    times_troughs_right = troughs_right / fps"""

    # Save detected events
    rows = []
    for f in peaks_left:
        rows.append({'side':'left','event':'heel_strike','frame':int(f),'time_s':float(f/fps)})
    for f in troughs_left:
        rows.append({'side':'left','event':'toe_off','frame':int(f),'time_s':float(f/fps)})
    for f in peaks_right:
        rows.append({'side':'right','event':'heel_strike','frame':int(f),'time_s':float(f/fps)})
    for f in troughs_right:
        rows.append({'side':'right','event':'toe_off','frame':int(f),'time_s':float(f/fps)})
    df_events = pd.DataFrame(rows)
    # Save session folder
    session_folder = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_folder, exist_ok=True)
    output_csv = os.path.join(session_folder, "detected_events.csv")
    df_events.to_csv(output_csv, index=False)
    print(f"💾 Saved detected events to {output_csv}")

    # ---------- Optional evaluation ----------
    """if os.path.exists(GROUND_TRUTH_CSV):
        gt = pd.read_csv(GROUND_TRUTH_CSV)
        # Accept either 'frame' or 'time_s' column
        if 'time_s' in gt.columns:
            gt_times = gt['time_s'].values
        elif 'frame' in gt.columns:
            gt_times = gt['frame'].values / fps
        else:
            print("Ground truth CSV must have 'time_s' or 'frame' column.")
            gt_times = []

        # combine all detected heel strikes times (both sides) for evaluation as example
        det_times = np.array(sorted(list(times_peaks_left) + list(times_peaks_right)))
        matched, unmatched_det, unmatched_gt = match_events_to_ground_truth(det_times, gt_times, tol=T_TOLERANCE)
        print("Evaluation:")
        print("GT events:", len(gt_times))
        print("Detected events:", len(det_times))
        print("Matched:", len(matched))
        if len(matched) > 0:
            errors = np.array([m[2] for m in matched])
            print("Mean absolute error (s):", np.mean(errors))
        print("Unmatched GT:", len(unmatched_gt), "Unmatched detections:", len(unmatched_det))
    else:
        print("No ground truth file found — skipped evaluation. Place a CSV at", GROUND_TRUTH_CSV)"""
def main():
    os.makedirs(SESSION_DIR, exist_ok=True)
    video_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".mp4", ".avi", ".mov"))]

    if not video_files:
        print(f"No videos found in {DATA_DIR}")
        return

    print(f"Found {len(video_files)} video(s) to process.")
    for idx, video_file in enumerate(video_files, 1):
        video_path = os.path.join(DATA_DIR, video_file)
        # create session id like 2025-11-01_01_filename
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:02d}_{os.path.splitext(video_file)[0]}"
        process_video(video_path, session_id)

if __name__ == "__main__":
    main()
