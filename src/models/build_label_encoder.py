import json
from pathlib import Path

# Paths based on your perfect directory structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_FILE = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
OUTPUT_MAPPING = PROJECT_ROOT / "data" / "class_mapping.json"

def build_encoder():
    if not INDEX_FILE.exists():
        print(f"❌ Cannot find {INDEX_FILE}")
        return

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_issues = set()
    
    for entry in data:
        issue = entry.get("primary_issue")
        if issue:
            # Lowercase and strip whitespace to prevent duplicates like "Drop foot" vs "drop foot "
            clean_issue = str(issue).strip().lower()
            unique_issues.add(clean_issue)

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