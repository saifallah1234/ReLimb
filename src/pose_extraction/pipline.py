import sys
from pathlib import Path

# Add the current directory to sys.path so it can find the other scripts
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

try:
    import limb_detection
    import gait_algorithm
except ImportError as e:
    print(f"❌ Error: Could not find project scripts. Ensure xml_loader.py, "
          f"limb_detection.py, and gait_algorithm.py are in: {SCRIPT_DIR}")
    sys.exit(1)

def main():
    print("="*50)
    print("🚀 STARTING PROGAIT RE-LIMB PIPELINE")
    print("="*50)

    # 1. Feature Extraction (MediaPipe + XML Overlay)
    print("\n[STEP 1/2] Extracting Keypoints and Events...")
    try:
        # This will walk through 'inside' and 'outside' automatically
        limb_detection.main()
    except Exception as e:
        print(f"❌ Fatal error in Step 1: {e}")
        return

    # 2. Metric Calculation
    print("\n[STEP 2/2] Calculating Gait Metrics...")
    try:
        # This analyzes the folders created in Step 1
        gait_algorithm.main()
    except Exception as e:
        print(f"❌ Fatal error in Step 2: {e}")
        return

    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETE")
    print(f"Check your results in: {limb_detection.SESSION_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()