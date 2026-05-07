"""
visualize_after_modification.py
===============================
Plays back random processed ReLimb sessions on a BLACK background. 
Overlays the smoothed keypoints (skeleton) and flashes gait events on screen.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random  # <--- Added to pick random clips

# ── Project layout ─────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  
SESSION_DIR  = PROJECT_ROOT / 'data' / 'sessions'

POSE_CONNECTIONS = [
 (11, 12), (11, 13), (12, 14),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def visualize_session(session_id: str):
    session_folder = SESSION_DIR / session_id
    
    if not session_folder.exists():
        print(f"⏭️  Skipping {session_id} (Folder does not exist: {session_folder})")
        return True

    # 💡 SMART FILE DISCOVERY: Grab whatever .npy and .csv files exist!
    npy_files = list(session_folder.glob("*.npy"))
    csv_files = list(session_folder.glob("*.csv"))

    if not npy_files or not csv_files:
        print(f"⏭️  Skipping {session_id} (Missing files in {session_folder})")
        return True

    kp_path = npy_files[0]
    ev_path = csv_files[0]

    # Load data
    keypoints = np.load(str(kp_path))
    events_df = pd.read_csv(ev_path)
    
    frame_idx = 0
    paused = False
    num_frames = len(keypoints)

    print(f"\n▶ Playing: {session_id}")
    print(f"  Loaded Keypoints: {kp_path.name} ({num_frames} frames)")
    print(f"  Loaded Events: {ev_path.name}")
    print("  Controls: [Q] Skip clip  |  [ESC] Quit completely  |  [SPACE] Pause")

    # Dynamically size the internal canvas based on coordinates so nothing gets cut off
    try:
        max_x = np.nanmax(keypoints[:, 0::2])
        max_y = np.nanmax(keypoints[:, 1::2])
        canvas_w = max(int(max_x) + 100, 1280)
        canvas_h = max(int(max_y) + 100, 720)
    except:
        canvas_w, canvas_h = 1920, 1080 # Fallback

    window_name = "ReLimb - Gait Event Viewer (Black Canvas)"
    
    # 💡 Setup a resizable window and make it smaller on the screen (800x600)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while frame_idx < num_frames:
        if not paused:
            # Create a pure black frame
            frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Check for events in this frame
            current_events = events_df[events_df['frame'] == frame_idx]
            
            # Draw Skeleton
            kp_row = keypoints[frame_idx]
            
            # Draw Bones
            for connection in POSE_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                x1, y1 = kp_row[pt1_idx * 2], kp_row[pt1_idx * 2 + 1]
                x2, y2 = kp_row[pt2_idx * 2], kp_row[pt2_idx * 2 + 1]
                
                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            
            # Draw Joints
            for i in range(33):
                x, y = kp_row[i * 2], kp_row[i * 2 + 1]
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

            # Overlay Event Text
            y_offset = 80
            for _, event_row in current_events.iterrows():
                side = event_row['side'].upper()
                evt_type = event_row['event'].replace('_', ' ').upper()
                text = f"{side} {evt_type}!"
                
                color = (0, 255, 0) if "HEEL" in evt_type else (0, 0, 255)
                cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, color, 4, cv2.LINE_AA)
                y_offset += 60

            # Display Frame info
            cv2.putText(frame, f"Frame: {frame_idx} / {num_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return False  # Return False to break the infinite loop
        elif key == ord('q'): # Skip
            break # Breaks the inner loop, moves to the next random video
        elif key == 32: # Space
            paused = not paused

        if not paused:
            frame_idx += 1

    return True  # Return True to keep the infinite loop going

def main():
    if not SESSION_DIR.exists():
        print(f"No sessions directory found at: {SESSION_DIR}")
        return

    # 1. Grab all folders inside the session directory
    available_sessions = [d.name for d in SESSION_DIR.iterdir() if d.is_dir()]
    
    if not available_sessions:
        print(f"No session folders found in {SESSION_DIR}.")
        return

    print(f"Found {len(available_sessions)} total sessions. Starting random playback...")

    # 2. Infinite loop picking random clips
    while True:
        sid = random.choice(available_sessions)
        continue_playing = visualize_session(sid)
        
        # If user pressed ESC, continue_playing is False
        if not continue_playing:
            print("\nPlayback terminated by user.")
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()