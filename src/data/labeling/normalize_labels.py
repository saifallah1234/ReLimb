import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_FILE = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
CLEAN_MAPPING_FILE = PROJECT_ROOT / "data" / "class_mapping.json"

# ── 5-class clinical grouping rationale ────────────────────────────────────
#
#  0  Normal Gait
#     → acceptable / no intervention needed
#
#  1  Foot & Ankle Issues
#     → Terminal Impact + Toe-Out/Rotation + Prosthesis Length
#     These all manifest at the foot/ankle level and share the same
#     clinical fix space (foot alignment, ankle stiffness, shank length)
#
#  2  Knee Issues
#     → Knee Flexion/Extension + Whip/Circumduction
#     Whip is usually a rotational compensation for insufficient knee flexion;
#     clinically treated at the same joint level
#
#  3  Alignment Issues
#     → Alignment/Varus/Valgus + Step Dimensions
#     Step width/length problems are almost always caused by coronal
#     alignment issues (varus/valgus shifts the base of support)
#
#  4  Unknown / Other
#     → kept as a catch-all so unlabeled data still contributes
#
# Result: from 8 clinical classes → 4 real classes + 1 unknown = 5 total
# Smallest class jumps from ~10 samples to ~28+ samples
# ──────────────────────────────────────────────────────────────────────────

MACRO_MAPPING = {
    # ── Normal ─────────────────────────────────────────────────────────────
    "normal gait":                                          "Normal Gait",
    "acceptable gait":                                      "Normal Gait",
    "regular gait (by this patient's standards)":           "Normal Gait",

    # ── Foot & Ankle ────────────────────────────────────────────────────────
    "asymmetric toe-out":                                   "Foot & Ankle Issue",
    "asymmetric toe-out 2 rotate foot outward underneath knee": "Foot & Ankle Issue",
    "asymmetrical toe-out":                                 "Foot & Ankle Issue",
    "toe-out asymmetry":                                    "Foot & Ankle Issue",
    "internal rotation of foot":                            "Foot & Ankle Issue",
    "internally rotated foot":                              "Foot & Ankle Issue",
    "hard terminal impact after swing extension":           "Foot & Ankle Issue",
    "terminal impact":                                      "Foot & Ankle Issue",
    "early foot flat":                                      "Foot & Ankle Issue",
    "excessive plantarflexion":                             "Foot & Ankle Issue",
    "incomplete roll-over":                                 "Foot & Ankle Issue",
    "not enough toe clearance":                             "Foot & Ankle Issue",
    "prosthesis too long":                                  "Foot & Ankle Issue",
    "prosthesis too short":                                 "Foot & Ankle Issue",

    # ── Knee ────────────────────────────────────────────────────────────────
    "asymmetric knee flexion angles":                       "Knee Issue",
    "hyperextended knee":                                   "Knee Issue",
    "insufficient knee flexion":                            "Knee Issue",
    "insufficient swing flexion":                           "Knee Issue",
    "insufficient swing phase flexion":                     "Knee Issue",
    "earlier problems (insufficient swing flexion) persist":"Knee Issue",
    "circumduction":                                        "Knee Issue",
    "medial whip (ignoring that the prosthesis is still too long)": "Knee Issue",
    "abducted gait":                                        "Knee Issue",

    # ── Alignment ───────────────────────────────────────────────────────────
    "knee varus":                                           "Alignment Issue",
    "varus deformity":                                      "Alignment Issue",
    "excessive valgus, may cause whip at higher speeds":    "Alignment Issue",
    "leg axis misalignment (leaning pylon)":                "Alignment Issue",
    "leaning pylon (from 1_6_3) persists":                  "Alignment Issue",
    "alignment is okay but socket is too wide in m/l":      "Alignment Issue",
    "incongruity of knee and ankle axes":                   "Alignment Issue",
    "lateral knee instability":                             "Alignment Issue",
    "base too narrow":                                      "Alignment Issue",
    "too narrow step width":                                "Alignment Issue",
    "uneven step length":                                   "Alignment Issue",
    "prosthetic step too long":                             "Alignment Issue",
    "problems (uneven step length, disruped swing initiation) persist": "Alignment Issue",
}

PATTERN_MAPPING: list[tuple[re.Pattern, str]] = [
    # Normal
    (re.compile(r"\b(normal gait|acceptable gait|regular gait|no action needed|everything looks optimized)\b", re.I),
     "Normal Gait"),

    # Foot & Ankle — toe/rotation/terminal/length
    (re.compile(r"\b(toe[- ]?out|toe[- ]?in|foot rotation|internal rotation|internally rotated|terminal impact|early foot flat|plantarflexion|roll-over|toe clearance|pronated|everted|prosthesis too (long|short)|lengthen shank|shorten shank)\b", re.I),
     "Foot & Ankle Issue"),

    # Knee — flexion/extension/whip/circumduction
    (re.compile(r"\b(knee flexion|swing flexion|insufficient knee|insufficient swing|hyperextended|extension resistance|flexion resistance|whip|circumduction|abducted gait)\b", re.I),
     "Knee Issue"),

    # Alignment — varus/valgus/step width/pylon
    (re.compile(r"\b(varus|valgus|alignment|leaning pylon|misalignment|knee instability|socket too wide|knee and ankle axes|step width|step length|base of support|base too narrow|narrow step|uneven step)\b", re.I),
     "Alignment Issue"),
]


def _normalize_text(value) -> str:
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


def normalize():
    if not INDEX_FILE.exists():
        print(f"❌ Cannot find {INDEX_FILE}")
        return

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts: dict[str, int] = {}
    for entry in data:
        extracted = _extract_primary_issue(entry)
        if entry.get("primary_issue") is None and extracted:
            entry["primary_issue"] = extracted
        clean = _map_to_macro(extracted)
        entry["clean_primary_issue"] = clean
        counts[clean] = counts.get(clean, 0) + 1

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # Stable ordering: Normal first, Unknown last, rest alphabetical
    ORDER = ["Normal Gait", "Foot & Ankle Issue", "Knee Issue", "Alignment Issue", "Unknown / Other"]
    unique = sorted(counts.keys(), key=lambda x: ORDER.index(x) if x in ORDER else 99)
    final_mapping = {label: idx for idx, label in enumerate(unique)}

    with open(CLEAN_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, indent=4)

    total = sum(counts.values())
    print("✅ Labels normalized to 5 classes\n")
    print(f"{'Class':<25} {'ID':>3}  {'Count':>6}  {'%':>5}")
    print("─" * 45)
    for label, idx in final_mapping.items():
        n = counts.get(label, 0)
        print(f"{label:<25} [{idx}]  {n:>6}  {n/total*100:>4.1f}%")
    print("─" * 45)
    print(f"{'TOTAL':<25}      {total:>6}")


if __name__ == "__main__":
    normalize()