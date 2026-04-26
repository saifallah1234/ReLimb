import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ProGaitDataset(Dataset):
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.sessions_dir = self.project_root / "data" / "sessions"
        self.index_file = self.project_root / "data" / "raw_videos" / "hf" / "dataset_index.json"
        self.mapping_file = self.project_root / "data" / "class_mapping.json"
        
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
            
        with open(self.index_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        self.valid_sessions = []
        
        for item in metadata:
            raw_id = item["ID"]
            clean_id = raw_id.replace(".mp4", "").replace(".avi", "")
            
            # Construct both possible folder names
            folder_inside = f"inside_{clean_id}"
            folder_outside = f"outside_{clean_id}"
            
            kp_path_inside = self.sessions_dir / folder_inside / "keypoints.npy"
            kp_path_outside = self.sessions_dir / folder_outside / "keypoints.npy"
            
            # Check which one actually exists and save that specific folder name
            if kp_path_inside.exists():
                item["actual_folder_name"] = folder_inside
                self.valid_sessions.append(item)
            elif kp_path_outside.exists():
                item["actual_folder_name"] = folder_outside
                self.valid_sessions.append(item)

        print(f"✅ PyTorch Dataset Initialized: {len(self.valid_sessions)} valid videos ready for training.")

    def __len__(self):
        return len(self.valid_sessions)

    def __getitem__(self, idx):
        item = self.valid_sessions[idx]
        
        # Use the exact folder name we found during __init__
        session_folder_name = item["actual_folder_name"]
        session_folder = self.sessions_dir / session_folder_name
        
        # ---------------------------------------------------------
        # A. LOAD INPUTS
        # ---------------------------------------------------------
        kp_path = session_folder / "keypoints.npy"
        keypoints = np.load(kp_path)
        keypoints = np.nan_to_num(keypoints, nan=0.0) 
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        
        # Updated placeholder: 5 zeros for metrics, 5 zeros for flags
        metrics_tensor = torch.zeros(10, dtype=torch.float32)
        
        # ---------------------------------------------------------
        # B. LOAD TARGETS 
        # ---------------------------------------------------------
        ccc_score = float(item.get("ccc_score", 0.0))
        ccc_tensor = torch.tensor([ccc_score], dtype=torch.float32)
        
        issue_text = item.get("clean_primary_issue", "Unknown / Other")
        issue_idx = self.class_mapping.get(issue_text, self.class_mapping.get("Unknown / Other", 0))
        issue_tensor = torch.tensor(issue_idx, dtype=torch.long)
        
        return keypoints_tensor, metrics_tensor, ccc_tensor, issue_tensor

def pad_collate_fn(batch):
    """
    Combines a list of individual samples into a padded batch.
    Required because Video A might be 150 frames and Video B might be 200 frames.
    """
    keypoints_list, metrics_list, ccc_list, issue_list = [], [], [], []
    lengths_list = []

    for kp, met, ccc, iss in batch:
        keypoints_list.append(kp)
        metrics_list.append(met)
        ccc_list.append(ccc)
        issue_list.append(iss)
        lengths_list.append(kp.shape[0])

    # Pad keypoints with 0.0 so all videos in the batch match the longest one
    padded_keypoints = pad_sequence(keypoints_list, batch_first=True, padding_value=0.0)
    
    metrics_batch = torch.stack(metrics_list)
    ccc_batch = torch.stack(ccc_list)
    issue_batch = torch.stack(issue_list)
    lengths_batch = torch.tensor(lengths_list, dtype=torch.long)

    return padded_keypoints, metrics_batch, ccc_batch, issue_batch, lengths_batch

# ... (Keep all the previous code above) ...

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("\n--- Testing ProGaitDataset ---")
    
    # 1. Instantiate the dataset
    dataset = ProGaitDataset()
    
    if len(dataset) == 0:
        print("❌ Dataset is empty. Check your folder paths in __init__.")
    else:
        # 2. Create a DataLoader to test batching and padding
        # batch_size=4 means we grab 4 videos at a time.
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=pad_collate_fn
        )
        
        # 3. Grab exactly one batch
        keypoints, metrics, ccc, issues, lengths = next(iter(dataloader))
        
        print("\n✅ Successfully loaded a batch! Here are the tensor shapes:")
        print(f"Keypoints Batch : {keypoints.shape}  -> [Batch_Size, Max_Frames, 66 (Features)] (Type: {keypoints.dtype})")
        print(f"Metrics Batch   : {metrics.shape}       -> [Batch_Size, 5 (Mocked Gait Metrics)] (Type: {metrics.dtype})")
        print(f"CCC Score Batch : {ccc.shape}       -> [Batch_Size, 1] (Type: {ccc.dtype})")
        print(f"Issues Batch    : {issues.shape}          -> [Batch_Size] (Type: {issues.dtype})")
        print(f"True Lengths    : {lengths.shape}          -> [Batch_Size] (Values: {lengths.tolist()})")
        
        print("\n--- Sample Values from Batch ---")
        print(f"CCC Scores      : {ccc.squeeze().tolist()}")
        print(f"Issue Class IDs : {issues.tolist()}")