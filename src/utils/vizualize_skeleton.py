import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random

# Point to your sessions folder
SESSION_DIR = Path('data/sessions')

def clean_keypoints_for_vis(kp_array):
    """Same pandas cleaning logic used in your dataset.py"""
    df = pd.DataFrame(kp_array)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.bfill().ffill().fillna(0.0)
    return df.to_numpy()

def visualize_random_clips(num_clips=3):
    # Get all valid clip folders
    sessions = [d for d in SESSION_DIR.iterdir() if d.is_dir() and "clip" in d.name]
    
    if not sessions:
        print("❌ No sessions found.")
        return

    # Pick a few random clips
    chosen = random.sample(sessions, min(num_clips, len(sessions)))

    for folder in chosen:
        kp_path = folder / "keypoints.npy"
        if not kp_path.exists():
            continue
            
        print(f"\n▶ Playing: {folder.name}")
        
        # Load and clean the keypoints
        raw_kp = np.load(kp_path).astype(np.float32)
        clean_kp = clean_keypoints_for_vis(raw_kp)
        
        # Play the frames
        for frame_idx in range(len(clean_kp)):
            # Create a blank black canvas
            # Adjust size if your original videos were larger (e.g., 1920x1080)
            img = np.zeros((800, 800, 3), dtype=np.uint8) 
            
            # Extract the 66 coordinates for this frame
            row = clean_kp[frame_idx]
            
            # Draw the 33 joints as green dots
            for j in range(33):
                x = int(row[j * 2])
                y = int(row[j * 2 + 1])
                
                # If coordinates are valid, draw them
                if x > 0 and y > 0:
                    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
                    
                    # Optional: Highlight feet in RED (MediaPipe indices 27-32)
                    if 27 <= j <= 32:
                        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

            # Add frame counter text
            cv2.putText(img, f"Frame: {frame_idx}/{len(clean_kp)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Skeleton Viewer (Press 'q' to skip to next clip)", img)
            
            # Wait 40ms (~25 FPS). Press 'q' to break out early.
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
                
    cv2.destroyAllWindows()
    print("\n✅ Finished viewing.")

if __name__ == "__main__":
    visualize_random_clips(5) # Watch 5 random clips