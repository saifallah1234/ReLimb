"""
visualize_dataset_pipeline.py
===============================
Loads the actual ProGaitDataset and visualizes the keypoints directly from __getitem__.
Shows RAW dataset output on the left, and NORMALIZED (/25.0) on the right.
Scaled up for better visibility!
"""

import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import random

# ── Project layout ─────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your dataset directly
from src.models.dataset import ProGaitDataset

def visualize_dataset_clip(dataset, idx):
    # 1. Pull directly from YOUR dataset pipeline 
    kp_tensor, metrics, ccc, issue = dataset[idx]
    
    # 2. Convert to numpy for OpenCV (Shape: Frames, 26)
    raw_keypoints = kp_tensor.numpy()
    
    # 💡 THIS IS THE MATH WE ARE TESTING 💡
    scaled_keypoints = raw_keypoints / 25.0
    
    frame_idx = 0
    paused = False
    num_frames = len(raw_keypoints)
    
    num_joints = raw_keypoints.shape[1] // 2 

    print(f"\n▶ Playing Dataset Index: {idx} | Frames: {num_frames}")
    print("  Controls: [Q] Skip clip  |  [ESC] Quit completely  |  [SPACE] Pause")

    # ⬆️ INCREASED CANVAS SIZE to fit the larger skeletons
    canvas_w, canvas_h = 1800, 960
    
    # ⬆️ MASSIVELY INCREASED SCALES
    # Left Side: RAW Data
    offset_x_raw = canvas_w // 4
    offset_y_raw = canvas_h // 2
    visual_scale_raw = 35  # <-- Was 15, now 35!

    # Right Side: SCALED Data
    offset_x_scaled = (canvas_w // 4) * 3
    offset_y_scaled = canvas_h // 2
    visual_scale_scaled = 35 * 25  # <-- Matches the 35 scale proportionally!

    window_name = "Dataset Pipeline Check: Unscaled (Left) vs /25.0 (Right)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 850)

    raw_min, raw_max = raw_keypoints.min(), raw_keypoints.max()
    scl_min, scl_max = scaled_keypoints.min(), scaled_keypoints.max()

    while frame_idx < num_frames:
        if not paused:
            frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            kp_raw_row = raw_keypoints[frame_idx]
            kp_scl_row = scaled_keypoints[frame_idx]
            
            # --- DRAW RAW (LEFT) ---
            for i in range(num_joints):
                x = (kp_raw_row[i * 2] * visual_scale_raw) + offset_x_raw
                y = (kp_raw_row[i * 2 + 1] * visual_scale_raw) + offset_y_raw
                # ⬆️ INCREASED DOT SIZE from 6 to 10
                cv2.circle(frame, (int(x), int(y)), 10, (255, 100, 100), -1)

            # --- DRAW SCALED (RIGHT) ---
            for i in range(num_joints):
                x = (kp_scl_row[i * 2] * visual_scale_scaled) + offset_x_scaled
                y = (kp_scl_row[i * 2 + 1] * visual_scale_scaled) + offset_y_scaled
                # ⬆️ INCREASED DOT SIZE from 6 to 10
                cv2.circle(frame, (int(x), int(y)), 10, (100, 255, 100), -1)

            # --- TEXT OVERLAYS ---
            cv2.line(frame, (canvas_w//2, 0), (canvas_w//2, canvas_h), (50, 50, 50), 2)

            cv2.putText(frame, "DATASET (Filtered, Unscaled)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Values Range: {raw_min:.2f} to {raw_max:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.putText(frame, "NORMALIZED (/ 25.0)", (offset_x_raw * 2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Values Range: {scl_min:.2f} to {scl_max:.2f}", (offset_x_raw * 2 + 50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.putText(frame, f"Frame: {frame_idx} / {num_frames}", (10, canvas_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return False
        elif key == ord('q'): # Skip
            break
        elif key == 32: # Space
            paused = not paused

        if not paused:
            frame_idx += 1

    return True

def main():
    print("Loading ProGaitDataset...")
    try:
        dataset = ProGaitDataset()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    print(f"Dataset loaded with {len(dataset)} clips. Starting visualizer...")

    while True:
        random_idx = random.randint(0, len(dataset) - 1)
        if not visualize_dataset_clip(dataset, random_idx):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()