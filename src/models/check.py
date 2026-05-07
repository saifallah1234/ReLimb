import json
import re
from collections import Counter
from pathlib import Path


def extract_clean_id(folder_name: str) -> str:
    """
    inside_1_1_f_1_clip_000 → 1_1_f_1
    """
    folder_name = folder_name.replace("inside_", "").replace("outside_", "")
    return folder_name.split("_clip_")[0]


MACRO_MAPPING = {
    "normal gait": "Normal Gait",
    "acceptable gait": "Normal Gait",
    "regular gait (by this patient's standards)": "Normal Gait",
    "asymmetric toe-out": "Toe-Out / Rotation Issue",
    "asymmetric toe-out 2 rotate foot outward underneath knee": "Toe-Out / Rotation Issue",
    "asymmetrical toe-out": "Toe-Out / Rotation Issue",
    "toe-out asymmetry": "Toe-Out / Rotation Issue",
    "internal rotation of foot": "Toe-Out / Rotation Issue",
    "internally rotated foot": "Toe-Out / Rotation Issue",
    "asymmetric knee flexion angles": "Knee Flexion/Extension Issue",
    "hyperextended knee": "Knee Flexion/Extension Issue",
    "insufficient knee flexion": "Knee Flexion/Extension Issue",
    "insufficient swing flexion": "Knee Flexion/Extension Issue",
    "insufficient swing phase flexion": "Knee Flexion/Extension Issue",
    "base too narrow": "Step Dimensions Issue",
    "too narrow step width": "Step Dimensions Issue",
    "uneven step length": "Step Dimensions Issue",
    "prosthetic step too long": "Step Dimensions Issue",
    "prosthesis too long": "Prosthesis Length Issue",
    "prosthesis too short": "Prosthesis Length Issue",
    "knee varus": "Alignment/Varus/Valgus Issue",
    "varus deformity": "Alignment/Varus/Valgus Issue",
    "excessive valgus, may cause whip at higher speeds": "Alignment/Varus/Valgus Issue",
    "leg axis misalignment (leaning pylon)": "Alignment/Varus/Valgus Issue",
    "leaning pylon (from 1_6_3) persists": "Alignment/Varus/Valgus Issue",
    "alignment is okay but socket is too wide in m/l": "Alignment/Varus/Valgus Issue",
    "incongruity of knee and ankle axes": "Alignment/Varus/Valgus Issue",
    "lateral knee instability": "Alignment/Varus/Valgus Issue",
    "circumduction": "Whip / Circumduction",
    "medial whip (ignoring that the prosthesis is still too long)": "Whip / Circumduction",
    "abducted gait": "Whip / Circumduction",
    "hard terminal impact after swing extension": "Terminal Impact / Foot Issue",
    "terminal impact": "Terminal Impact / Foot Issue",
    "early foot flat": "Terminal Impact / Foot Issue",
    "excessive plantarflexion": "Terminal Impact / Foot Issue",
    "incomplete roll-over": "Terminal Impact / Foot Issue",
    "not enough toe clearance": "Terminal Impact / Foot Issue",
    "earlier problems (insufficient swing flexion) persist": "Knee Flexion/Extension Issue",
    "problems (uneven step length, disruped swing initiation) persist": "Step Dimensions Issue",
}

PATTERN_MAPPING: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(normal gait|acceptable gait|regular gait|natural|no action needed|everything looks optimized)\b", re.I), "Normal Gait"),
    (re.compile(r"\b(toe[- ]?out|toe[- ]?in|foot rotation|internal rotation of foot|internally rotated foot|toe-out asymmetry|asymmetrical toe-out|asymmetric toe-out)\b", re.I), "Toe-Out / Rotation Issue"),
    (re.compile(r"\b(knee flexion|swing flexion|insufficient knee flexion|insufficient swing|hyperextended knee|extension resistance|flexion resistance)\b", re.I), "Knee Flexion/Extension Issue"),
    (re.compile(r"\b(step width|step length|base of support|base too narrow|narrow step|prosthetic step too long|uneven step length)\b", re.I), "Step Dimensions Issue"),
    (re.compile(r"\b(prosthesis too (long|short)|too long|too short|lengthen shank|shorten shank)\b", re.I), "Prosthesis Length Issue"),
    (re.compile(r"\b(varus|valgus|alignment|leaning pylon|misalignment|knee instability|socket too wide|knee and ankle axes)\b", re.I), "Alignment/Varus/Valgus Issue"),
    (re.compile(r"\b(whip|circumduction|abducted gait)\b", re.I), "Whip / Circumduction"),
    (re.compile(r"\b(terminal impact|terminal knee impact|early foot flat|plantarflexion|roll-over|toe clearance|pronated|everted)\b", re.I), "Terminal Impact / Foot Issue"),
]


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _extract_primary_issue(entry: dict) -> str | None:
    primary = entry.get("primary_issue")
    if primary:
        return str(primary).strip()

    full_text = entry.get("full_text")
    if not full_text:
        return None

    for line in str(full_text).splitlines():
        line = line.strip()
        if not line or line.lower().startswith("ccc"):
            continue
        m = re.match(r"^\s*1\s*[\)\.\-:]?\s*(.+)$", line)
        if m and m.group(1).strip():
            return m.group(1).strip()

    return None


def _map_to_macro(raw_issue: str | None) -> str:
    normalized = _normalize_text(raw_issue)
    if not normalized:
        return "Unknown / Other"

    if normalized in MACRO_MAPPING:
        return MACRO_MAPPING[normalized]

    for pattern, label in PATTERN_MAPPING:
        if pattern.search(normalized):
            return label

    return "Unknown / Other"


def check_class_distribution():
    print("\n--- 📊 CLIP-LEVEL CLASS DISTRIBUTION ---")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    INDEX_FILE = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
    MAPPING_FILE = PROJECT_ROOT / "data" / "class_mapping.json"
    sessions_dir = PROJECT_ROOT / "data" / "sessions"
    if not sessions_dir.exists():
        sessions_dir = PROJECT_ROOT / "data" / "session"
    if not sessions_dir.exists():
        print("❌ Cannot find data/sessions or data/session")
        return
    

    # Load mapping
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    # Load metadata
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_lookup = {}
    for item in metadata:
        raw_id = item["ID"]
        clean_id = raw_id.replace(".mp4", "").replace(".avi", "")
        meta_lookup[clean_id] = item
    matched = 0
    total = 0

    for folder in sessions_dir.iterdir():
        if "_clip_" not in folder.name:
            continue

        total += 1

        clean_id = folder.name.replace("inside_", "").replace("outside_", "").split("_clip_")[0]

        if clean_id in meta_lookup:
            matched += 1

    print("Matched:", matched)
    print("Total:", total)
    print("Coverage:", matched / total)

    # ─────────────────────────────────────────────
    # CLIP-LEVEL COUNT (IMPORTANT FIX)
    # ─────────────────────────────────────────────
    counts = Counter()
    total_clips = 0

    for folder in sessions_dir.iterdir():
        if not folder.is_dir():
            continue
        if "_clip_" not in folder.name:
            continue

        clean_id = extract_clean_id(folder.name)
        meta = meta_lookup.get(clean_id, None)

        if meta is None:
            label = "Unknown / Other"
        else:
            label = meta.get("clean_primary_issue")
            if not label:
                label = _map_to_macro(_extract_primary_issue(meta))

        counts[label] += 1
        total_clips += 1

    # ─────────────────────────────────────────────
    # PRINT REPORT
    # ─────────────────────────────────────────────
    for label in class_mapping.keys():
        counts.setdefault(label, 0)

    print(f"\nTotal Clips: {total_clips}")
    print("-" * 60)
    print(f"{'Class Name':<35} | {'Count':<6} | {'Percentage'}")
    print("-" * 60)

    ordered_labels = [label for label, _ in sorted(class_mapping.items(), key=lambda kv: kv[1])]
    if "Unknown / Other" in counts and "Unknown / Other" not in ordered_labels:
        ordered_labels.append("Unknown / Other")

    for label in ordered_labels:
        count = counts.get(label, 0)
        pct = (count / total_clips) * 100 if total_clips else 0
        print(f"{label:<35} | {count:<6} | {pct:.1f}%")

    print("-" * 60)

    # ─────────────────────────────────────────────
    # DIAGNOSIS
    # ─────────────────────────────────────────────
    if counts:
        top_label, top_count = counts.most_common(1)[0]
        if (top_count / total_clips) > 0.5:
            print("\n⚠️ WARNING: Severe imbalance at clip level")
            print(f"'{top_label}' dominates dataset.")
        else:
            print("\n✅ Clip-level distribution is acceptable")


if __name__ == "__main__":
    check_class_distribution()
    