import json
import re
from pathlib import Path

# Paths based on your perfect directory structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_FILE = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
OUTPUT_MAPPING = PROJECT_ROOT / "data" / "class_mapping.json"

# Keep the same macro classes used in normalize_labels.py
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
    cleaned = re.sub(r"\s+", " ", str(value)).strip().lower()
    return cleaned


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

def build_encoder():
    if not INDEX_FILE.exists():
        print(f"❌ Cannot find {INDEX_FILE}")
        return

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_issues = set()

    for entry in data:
        clean_issue = entry.get("clean_primary_issue")
        if not clean_issue:
            extracted = _extract_primary_issue(entry)
            clean_issue = _map_to_macro(extracted)
        if clean_issue:
            unique_issues.add(str(clean_issue))

    # Convert the set to a sorted list, then to a dictionary mapping text -> integer
    sorted_issues = sorted(list(unique_issues))
    class_mapping = {issue: idx for idx, issue in enumerate(sorted_issues)}
    
    # We should also map the CCC score just to see the distribution
    ccc_scores = [entry.get("ccc_score") for entry in data if entry.get("ccc_score") is not None]

    # Save the mapping so PyTorch can use it later
    with open(OUTPUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)

    print("✅ Label Encoder Built Successfully!")
    print(f"📁 Saved to: {OUTPUT_MAPPING}")
    print("\n--- Unique Primary Issues Found ---")
    print(f"Total Unique Classes: {len(class_mapping)}")
    for issue, idx in class_mapping.items():
        print(f" [{idx}] -> {issue}")

if __name__ == "__main__":
    build_encoder()