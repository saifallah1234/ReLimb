import json
from collections import Counter
from pathlib import Path

def check_class_distribution():
    print("\n--- 📊 CHECKING DATASET CLASS DISTRIBUTION ---")
    
    # 1. Define Paths (Same as your training script)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    INDEX_FILE = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
    MAPPING_FILE = PROJECT_ROOT / "data" / "class_mapping.json"
    
    # 2. Load the Class Mapping
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)

    # 3. Load the Metadata
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    # 4. Tally the classes
    class_counts = Counter()
    total_valid = 0
    
    for item in metadata:
        # Get the text just like your training script does
        issue_text = item.get("clean_primary_issue")
        if not issue_text:
            issue_text = "Unknown / Other"
        
        # Find the official class ID
        class_id = class_mapping.get(issue_text, class_mapping.get("Unknown / Other", 0))
        
        # Get the string name back for readability in the terminal
        class_name = [name for name, cid in class_mapping.items() if cid == class_id]
        class_name = class_name[0] if class_name else f"Class ID {class_id}"
        
        class_counts[class_name] += 1
        total_valid += 1
        
    # 5. Print out a beautiful terminal report
    print(f"\nTotal Videos Analyzed: {total_valid}")
    print("-" * 60)
    print(f"{'Class Name':<35} | {'Count':<6} | {'Percentage'}")
    print("-" * 60)
    
    # Sort by the most common class down to the least common
    for class_name, count in class_counts.most_common():
        percentage = (count / total_valid) * 100
        print(f"{class_name:<35} | {count:<6} | {percentage:.1f}%")
        
    print("-" * 60)
    
    # 6. Diagnosis Warning
    if class_counts:
        most_common_name, most_common_count = class_counts.most_common(1)[0]
        max_percentage = (most_common_count / total_valid) * 100
        
        if max_percentage > 50:
            print(f"\n⚠️ DIAGNOSIS: SEVERE IMBALANCE DETECTED")
            print(f"'{most_common_name}' makes up {max_percentage:.1f}% of your entire dataset.")
            print("Your AI is likely cheating by guessing this class for every video.")
        else:
            print("\n✅ DIAGNOSIS: Dataset looks reasonably balanced!")

if __name__ == "__main__":
    check_class_distribution()