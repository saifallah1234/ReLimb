import cv2
import random
from pathlib import Path
from cvzone.PoseModule import PoseDetector

# Point to your raw videos
DATA_DIR = Path('data/raw_videos/hf')

def visualize_raw_overlay(num_clips=5):
    # Get all video files that are "clips"
    valid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    all_videos = [p for p in DATA_DIR.rglob('*') if p.suffix.lower() in valid_exts and "clip" in p.name.lower()]
    
    if not all_videos:
        print(f"❌ No clip videos found in {DATA_DIR}")
        return

    # Pick a few random clips to test
    chosen_videos = random.sample(all_videos, min(num_clips, len(all_videos)))
    
    # Initialize the same detector used in limb_detection.py
    # Change modelComplexity from 1 to 2
    detector = PoseDetector(staticMode=False, modelComplexity=2, smoothLandmarks=True, 
                            enableSegmentation=False, detectionCon=0.6, trackCon=0.6)

    for video_path in chosen_videos:
        print(f"\n▶ Playing Original Video Overlay: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize exactly how your extraction script does it
            frame = cv2.resize(frame, (640, 480))
            
            # Find the pose and draw it on the frame
            frame = detector.findPose(frame, draw=True)
            
            # Add text overlay
            cv2.putText(frame, "Press 'q' to skip to next video", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the video
            cv2.imshow("Raw Video Skeleton Overlay", frame)
            
            # Playback speed (approx 25 fps). Press 'q' to skip to the next video.
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
                
        cap.release()
        
    cv2.destroyAllWindows()
    print("\n✅ Finished viewing.")

if __name__ == "__main__":
    visualize_raw_overlay(5)  # Watch 5 random videos