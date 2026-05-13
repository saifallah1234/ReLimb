from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pose_extraction.limb_detection import process_single_video
from src.pose_extraction.normalize_poses import normalize_keypoints_file
from src.inference.stgcn_inference import predict_label
from src.utils.llm_client import generate_llm_summary, load_notes_from_metadata


def run_pipeline(
    video_path: Path,
    video_id: str | None = None,
    output_dir: Path | None = None,
    fixed_label: str | None = None,
    user_input: str = "",
) -> Path:
    session_folder = process_single_video(
        video_path=video_path,
        video_id=video_id,
        output_dir=output_dir,
    )

    kp_path = session_folder / "keypoints.npy"
    normalized_path = normalize_keypoints_file(kp_path)
    if normalized_path is None:
        raise RuntimeError("Normalization failed: missing hips or keypoints file.")
    if fixed_label:
        label = fixed_label
        score = None
    else:
        label, score = predict_label(np.load(str(normalized_path)))

    notes = load_notes_from_metadata(video_path)
    summary = generate_llm_summary(label, notes, user_input=user_input)

    result = {
        "video_id": session_folder.name,
        "label": label,
        "summary": summary,
        "score": score,
        "notes": notes,
        "keypoints": str(kp_path),
        "keypoints_normalized": str(normalized_path),
    }
    result_path = session_folder / "prediction.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return normalized_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-video pose extraction + normalization.")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--video-id", type=str, default=None, help="Optional output ID (defaults to filename stem)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (defaults to data/results)")
    parser.add_argument("--label", type=str, default=None, help="Optional fixed label for stub prediction")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    normalized_path = run_pipeline(
        video_path=video_path,
        video_id=args.video_id,
        output_dir=output_dir,
        fixed_label=args.label,
    )

    print(f"✅ Normalized keypoints saved to: {normalized_path}")


if __name__ == "__main__":
    main()
