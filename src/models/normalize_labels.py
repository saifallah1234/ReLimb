import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_FILE = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
CLEAN_MAPPING_FILE = PROJECT_ROOT / "data" / "class_mapping.json"

# We map the noisy raw text to clean, standardized categories
MACRO_MAPPING = {
    # 1. Normal / Acceptable
    "normal gait": "Normal Gait",
    "acceptable gait": "Normal Gait",
    "regular gait (by this patient's standards)": "Normal Gait",
    
    # 2. Foot Rotation / Toe-Out Issues
    "asymmetric toe-out": "Toe-Out / Rotation Issue",
    "asymmetric toe-out 2 rotate foot outward underneath knee": "Toe-Out / Rotation Issue",
    "asymmetrical toe-out": "Toe-Out / Rotation Issue",
    "toe-out asymmetry": "Toe-Out / Rotation Issue",
    "internal rotation of foot": "Toe-Out / Rotation Issue",
    "internally rotated foot": "Toe-Out / Rotation Issue",
    
    # 3. Knee Flexion / Extension Issues
    "asymmetric knee flexion angles": "Knee Flexion/Extension Issue",
    "hyperextended knee": "Knee Flexion/Extension Issue",
    "insufficient knee flexion": "Knee Flexion/Extension Issue",
    "insufficient swing flexion": "Knee Flexion/Extension Issue",
    "insufficient swing phase flexion": "Knee Flexion/Extension Issue",
    
    # 4. Step Width / Length Issues
    "base too narrow": "Step Dimensions Issue",
    "too narrow step width": "Step Dimensions Issue",
    "uneven step length": "Step Dimensions Issue",
    "prosthetic step too long": "Step Dimensions Issue",
    
    # 5. Prosthesis Length
    "prosthesis too long": "Prosthesis Length Issue",
    "prosthesis too short": "Prosthesis Length Issue",
    
    # 6. Alignment / Varus / Valgus
    "knee varus": "Alignment/Varus/Valgus Issue",
    "varus deformity": "Alignment/Varus/Valgus Issue",
    "excessive valgus, may cause whip at higher speeds": "Alignment/Varus/Valgus Issue",
    "leg axis misalignment (leaning pylon)": "Alignment/Varus/Valgus Issue",
    "leaning pylon (from 1_6_3) persists": "Alignment/Varus/Valgus Issue",
    "alignment is okay but socket is too wide in m/l": "Alignment/Varus/Valgus Issue",
    "incongruity of knee and ankle axes": "Alignment/Varus/Valgus Issue",
    "lateral knee instability": "Alignment/Varus/Valgus Issue",
    
    # 7. Whip / Circumduction
    "circumduction": "Whip / Circumduction",
    "medial whip (ignoring that the prosthesis is still too long)": "Whip / Circumduction",
    "abducted gait": "Whip / Circumduction",
    
    # 8. Ankle / Terminal Impact
    "hard terminal impact after swing extension": "Terminal Impact / Foot Issue",
    "terminal impact": "Terminal Impact / Foot Issue",
    "early foot flat": "Terminal Impact / Foot Issue",
    "excessive plantarflexion": "Terminal Impact / Foot Issue",
    "incomplete roll-over": "Terminal Impact / Foot Issue",
    "not enough toe clearance": "Terminal Impact / Foot Issue",
    
    # 9. Generic Persistence (Can be grouped as 'Other' or 'Normal')
    "earlier problems (insufficient swing flexion) persist": "Knee Flexion/Extension Issue",
    "problems (uneven step length, disruped swing initiation) persist": "Step Dimensions Issue"
}

def normalize():
    if not INDEX_FILE.exists():
        print(f"❌ Cannot find {INDEX_FILE}")
        return

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update the master index with the CLEAN labels
    for entry in data:
        raw_issue = str(entry.get("primary_issue")).strip().lower()
        if raw_issue in MACRO_MAPPING:
            entry["clean_primary_issue"] = MACRO_MAPPING[raw_issue]
        else:
            entry["clean_primary_issue"] = "Unknown / Other" # Fallback

    # 2. Overwrite the dataset_index.json so it has the clean labels
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # 3. Create the final integer mapping for PyTorch
    unique_clean_issues = sorted(list(set([entry["clean_primary_issue"] for entry in data if entry.get("clean_primary_issue")])))
    final_mapping = {issue: idx for idx, issue in enumerate(unique_clean_issues)}

    with open(CLEAN_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, indent=4)

    print("✅ Labels Normalized Successfully!")
    print(f"Total Macro-Classes: {len(final_mapping)}")
    for issue, idx in final_mapping.items():
        print(f" [{idx}] -> {issue}")

if __name__ == "__main__":
    normalize()