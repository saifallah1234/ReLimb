import numpy as np
from pathlib import Path

# ── Project layout ──
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SESSION_DIR  = PROJECT_ROOT / 'data' / 'sessions'
RESULTS_DIR  = PROJECT_ROOT / 'data' / 'results'


def _normalize_keypoints_array(keypoints: np.ndarray) -> np.ndarray | None:
    normalized_keypoints = np.copy(keypoints)
    num_frames = keypoints.shape[0]

    # Collect per-frame scale estimates first so we can use a robust clip-level scale.
    # Primary: pelvis->neck (shoulders). Fallback: hip-to-hip distance.
    scales = []
    for i in range(num_frames):
        kp_row = keypoints[i]

        # Hips: Left (23), Right (24)
        left_hip_x,  left_hip_y  = kp_row[23 * 2], kp_row[23 * 2 + 1]
        right_hip_x, right_hip_y = kp_row[24 * 2], kp_row[24 * 2 + 1]

        # Shoulders: Left (11), Right (12)
        left_sho_x,  left_sho_y  = kp_row[11 * 2], kp_row[11 * 2 + 1]
        right_sho_x, right_sho_y = kp_row[12 * 2], kp_row[12 * 2 + 1]

        if np.isnan(left_hip_x) or np.isnan(right_hip_x):
            continue

        pelvis_x = (left_hip_x + right_hip_x) / 2.0
        pelvis_y = (left_hip_y + right_hip_y) / 2.0

        # 1) pelvis->neck (from shoulders)
        if not (np.isnan(left_sho_x) or np.isnan(right_sho_x) or np.isnan(left_sho_y) or np.isnan(right_sho_y)):
            neck_x = ((left_sho_x - pelvis_x) + (right_sho_x - pelvis_x)) / 2.0
            neck_y = ((left_sho_y - pelvis_y) + (right_sho_y - pelvis_y)) / 2.0
            s1 = float(np.sqrt(neck_x**2 + neck_y**2))
        else:
            s1 = np.nan

        # 2) hip-to-hip distance fallback
        s2 = float(np.sqrt((left_hip_x - right_hip_x) ** 2 + (left_hip_y - right_hip_y) ** 2))

        s = s1 if (np.isfinite(s1) and s1 > 1e-6) else s2
        if np.isfinite(s) and s > 1e-6:
            scales.append(s)

    # Use a robust scale for the full clip
    if len(scales) == 0:
        # Can't normalize (hips missing everywhere)
        return None

    clip_scale = float(np.median(scales))

    for i in range(num_frames):
        kp_row = keypoints[i]

        left_hip_x,  left_hip_y  = kp_row[23 * 2], kp_row[23 * 2 + 1]
        right_hip_x, right_hip_y = kp_row[24 * 2], kp_row[24 * 2 + 1]

        if np.isnan(left_hip_x) or np.isnan(right_hip_x):
            continue

        pelvis_x = (left_hip_x + right_hip_x) / 2.0
        pelvis_y = (left_hip_y + right_hip_y) / 2.0

        # center around pelvis
        normalized_keypoints[i, 0::2] -= pelvis_x
        normalized_keypoints[i, 1::2] -= pelvis_y

        # scale with robust clip median
        s = max(clip_scale, 1e-6)
        normalized_keypoints[i, 0::2] /= s
        normalized_keypoints[i, 1::2] /= s

        # Hard clamp as a last-resort safety (prevents extreme outliers from poisoning training)
        normalized_keypoints[i, :] = np.clip(normalized_keypoints[i, :], -10.0, 10.0)

    return normalized_keypoints


def normalize_keypoints_file(kp_path: Path, save_path: Path | None = None) -> Path | None:
    if not kp_path.exists():
        return None

    keypoints = np.load(str(kp_path))
    normalized_keypoints = _normalize_keypoints_array(keypoints)
    if normalized_keypoints is None:
        return None

    if save_path is None:
        save_path = kp_path.parent / "keypoints_normalized.npy"

    np.save(str(save_path), normalized_keypoints)
    return save_path

def normalize_session(session_id: str):
    session_folder = SESSION_DIR / session_id
    kp_path = session_folder / "keypoints.npy"

    save_path = normalize_keypoints_file(kp_path)
    if save_path is None:
        return

    print(f"✅ Centered & Scaled: {session_id}")


def normalize_result_video(video_id: str, results_dir: Path | None = None) -> Path | None:
    if results_dir is None:
        results_dir = RESULTS_DIR
    kp_path = results_dir / video_id / "keypoints.npy"
    return normalize_keypoints_file(kp_path)

def main():
    available_sessions = [d.name for d in SESSION_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(available_sessions)} sessions to normalize...")
    
    for sid in available_sessions:
        normalize_session(sid)
        
    print("\n🎉 All sessions centered and scaled successfully!")

if __name__ == "__main__":
    main()