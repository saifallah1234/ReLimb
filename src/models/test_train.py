import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

# Import your models (assuming you saved them in model_mlp.py and model_lstm.py)
from model_mlp import GaitMetricsMLP
from model_lstm import GaitSequenceLSTM

# ==========================================
# 1. DATASET & COLLATOR (Handling the files)
# ==========================================
class ProGaitDataset(Dataset):
    def __init__(self, session_folders, labels_dict):
        """
        session_folders: list of paths to your 'inside/1/1_1_1/' folders
        labels_dict: dictionary mapping session ID to its (class_id, ccc_score)
        """
        self.session_folders = session_folders
        self.labels_dict = labels_dict
        
        # We will extract metrics into a standard array to fit the Scaler later
        self.raw_metrics = []
        self.masks = []
        self.keypoints_list = []
        self.targets_class = []
        self.targets_ccc = []
        
        self._load_all_data()

    def _load_all_data(self):
        for folder in self.session_folders:
            session_id = os.path.basename(folder) # e.g., '1_1_1_f'
            
            # 1. Load Keypoints
            kp_path = os.path.join(folder, 'keypoints.npy')
            kp_data = np.load(kp_path) # Shape: (Frames, 66)
            
            # --- THE FIX: Convert array to tensor, then scrub NaNs to 0.0 ---
            kp_tensor = torch.tensor(kp_data, dtype=torch.float32)
            kp_tensor = torch.nan_to_num(kp_tensor, nan=0.0) 
            self.keypoints_list.append(kp_tensor)
            
            # 2. Load Metrics
            metrics_path = os.path.join(folder, 'gait_metrics.json')
            with open(metrics_path, 'r') as f:
                m_data = json.load(f)
                
            # Extract the 5 core features
            feats = [
                m_data['gait_cycle'].get('cadence_bpm', np.nan),
                m_data['gait_cycle'].get('stride_time_avg_l', np.nan),
                m_data['gait_cycle'].get('stride_time_avg_r', np.nan),
                m_data['symmetry'].get('stride_time_asymmetry', np.nan),
                m_data['symmetry'].get('step_length_pixel_avg', np.nan)
            ]
            
            # Create NaN mask
            mask = [1.0 if not np.isnan(v) else 0.0 for v in feats]
            feats = [v if not np.isnan(v) else 0.0 for v in feats]
            
            self.raw_metrics.append(feats)
            self.masks.append(mask)
            
            # 3. Load Targets (From your master JSON)
            label_info = self.labels_dict[session_id]
            self.targets_class.append(label_info['class_id'])
            self.targets_ccc.append(label_info['ccc_score'])

    def __len__(self):
        return len(self.session_folders)

    def __getitem__(self, idx):
        # Metrics will be scaled externally during LOOCV, so we just return indices or raw
        return {
            'metrics': torch.tensor(self.raw_metrics[idx], dtype=torch.float32),
            'mask': torch.tensor(self.masks[idx], dtype=torch.float32),
            'keypoints': self.keypoints_list[idx],
            'length': len(self.keypoints_list[idx]),
            'class_id': torch.tensor(self.targets_class[idx], dtype=torch.long),
            'ccc': torch.tensor(self.targets_ccc[idx], dtype=torch.float32)
        }

def collate_fn(batch):
    """Pads variable length keypoint sequences with zeros."""
    metrics = torch.stack([item['metrics'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    class_ids = torch.stack([item['class_id'] for item in batch])
    cccs = torch.stack([item['ccc'] for item in batch])
    
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Pad sequences
    keypoints = [item['keypoints'] for item in batch]
    padded_keypoints = torch.nn.utils.rnn.pad_sequence(keypoints, batch_first=True, padding_value=0.0)
    
    return metrics, masks, padded_keypoints, lengths, class_ids, cccs

# ==========================================
# 2. THE FUSION MODEL
# ==========================================
class ProGaitFusion(nn.Module):
    def __init__(self, num_classes, mode="fusion"):
        """mode can be: 'fusion', 'mlp_only', 'lstm_only'"""
        super().__init__()
        self.mode = mode
        
        self.mlp = GaitMetricsMLP(n_metrics=5, hidden_dim=32, output_dim=64)
        self.lstm = GaitSequenceLSTM(input_dim=66, hidden_dim=64, num_layers=2)
        
        # Decide fusion dimension based on mode
        if mode == "fusion":
            fusion_dim = 64 + 128 # 192
        elif mode == "mlp_only":
            fusion_dim = 64
        elif mode == "lstm_only":
            fusion_dim = 128
            
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.ccc_scorer = nn.Linear(fusion_dim, 1)

    def forward(self, metrics, masks, keypoints, lengths):
        # Concatenate scaled metrics and masks (5 + 5 = 10)
        mlp_input = torch.cat([metrics, masks], dim=1)
        
        out_mlp = self.mlp(mlp_input)                 # (Batch, 64)
        out_lstm = self.lstm(keypoints, lengths)      # (Batch, 128)
        
        if self.mode == "fusion":
            fused = torch.cat([out_mlp, out_lstm], dim=1)
        elif self.mode == "mlp_only":
            fused = out_mlp
        elif self.mode == "lstm_only":
            fused = out_lstm
            
        class_logits = self.classifier(fused)
        ccc_pred = self.ccc_scorer(fused).squeeze(1)
        
        return class_logits, ccc_pred

# ==========================================
# 3. LOOCV TRAINING LOOP
# ==========================================
def train_and_evaluate_loocv(dataset, num_classes, mode="fusion", epochs=15):
    print(f"\n--- Starting LOOCV for mode: {mode.upper()} ---")
    loo = LeaveOneOut()
    
    all_true_classes = []
    all_pred_classes = []
    all_true_cccs = []
    all_pred_cccs = []
    
    # loo.split creates 30 folds. 29 for train, 1 for val.
    for fold, (train_idx, val_idx) in enumerate(loo.split(dataset)):
        
        # 1. Fit Standard Scaler ONLY on training fold metrics to prevent data leakage
        train_metrics = np.array([dataset.raw_metrics[i] for i in train_idx])
        scaler = StandardScaler()
        scaler.fit(train_metrics)
        
        # 2. Extract and scale data for this fold
        for i in range(len(dataset)):
            dataset.raw_metrics[i] = scaler.transform([dataset.raw_metrics[i]])[0]

        # 3. Create DataLoaders
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # 4. Initialize Model, Loss, Optimizer
        model = ProGaitFusion(num_classes=num_classes, mode=mode)
        criterion_class = nn.CrossEntropyLoss()
        criterion_ccc = nn.MSELoss() # Mean Squared Error for regression
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        
        # 5. Train for X epochs
        model.train()
        for epoch in range(epochs):
            for metrics, masks, keypoints, lengths, class_ids, cccs in train_loader:
                optimizer.zero_grad()
                
                logits, ccc_preds = model(metrics, masks, keypoints, lengths)
                
                loss_cls = criterion_class(logits, class_ids)
                loss_ccc = criterion_ccc(ccc_preds, cccs)
                loss = loss_cls + (loss_ccc * 0.5) # Balance the two tasks
                
                loss.backward()
                optimizer.step()
                
        # 6. Evaluate on the 1 left-out video
        model.eval()
        with torch.no_grad():
            for metrics, masks, keypoints, lengths, class_ids, cccs in val_loader:
                logits, ccc_preds = model(metrics, masks, keypoints, lengths)
                
                pred_class = torch.argmax(logits, dim=1).item()
                pred_ccc = ccc_preds.item()
                
                all_true_classes.append(class_ids.item())
                all_pred_classes.append(pred_class)
                all_true_cccs.append(cccs.item())
                all_pred_cccs.append(pred_ccc)

    # 7. Calculate Final LOOCV Metrics
    acc = accuracy_score(all_true_classes, all_pred_classes)
    mae = mean_absolute_error(all_true_cccs, all_pred_cccs)
    
    print(f"Results for {mode.upper()}:")
    print(f"-> Classification Accuracy: {acc * 100:.2f}%")
    print(f"-> CCC Score MAE (Error margin): {mae:.2f}")
    return acc, mae

# ==========================================
# RUNNING THE EXPERIMENTS
# ==========================================

from sklearn.model_selection import KFold

def train_and_evaluate_quick_test(dataset, num_classes, mode="fusion", epochs=5):
    print(f"\n--- Starting QUICK TEST for mode: {mode.upper()} ---")
    
    # Using 5-Fold instead of Leave-One-Out for speed
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_true_classes, all_pred_classes = [], []
    all_true_cccs, all_pred_cccs = [], []
    
    # We only take the first 50 samples for this quick test
    subset_indices = np.arange(min(len(dataset), 50))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(subset_indices)):
        print(f"Processing Fold {fold+1}/5...")
        
        # 1. Scaling
        train_metrics = np.array([dataset.raw_metrics[i] for i in train_idx])
        scaler = StandardScaler().fit(train_metrics)
        for i in range(len(dataset)):
            dataset.raw_metrics[i] = scaler.transform([dataset.raw_metrics[i]])[0]

        # 2. DataLoaders
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_sub, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # 3. Model & Optimizer
        model = ProGaitFusion(num_classes=num_classes, mode=mode)
        # MOVE TO GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion_class = nn.CrossEntropyLoss()
        criterion_ccc = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        # 4. Short Training
        model.train()
        for epoch in range(epochs):
            for metrics, masks, keypoints, lengths, class_ids, cccs in train_loader:
                metrics, masks = metrics.to(device), masks.to(device)
                keypoints, class_ids, cccs = keypoints.to(device), class_ids.to(device), cccs.to(device)
                
                optimizer.zero_grad()
                logits, ccc_preds = model(metrics, masks, keypoints, lengths)
                loss = criterion_class(logits, class_ids) + (criterion_ccc(ccc_preds, cccs) * 0.5)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 5. Quick Eval
        model.eval()
        with torch.no_grad():
            for metrics, masks, keypoints, lengths, class_ids, cccs in val_loader:
                metrics, masks, keypoints = metrics.to(device), masks.to(device), keypoints.to(device)
                logits, ccc_preds = model(metrics, masks, keypoints, lengths)
                all_true_classes.append(class_ids.item())
                all_pred_classes.append(torch.argmax(logits, dim=1).item())
                all_true_cccs.append(cccs.item())
                all_pred_cccs.append(ccc_preds.item())

    print(f"\n✅ Quick Test Results: Acc {accuracy_score(all_true_classes, all_pred_classes)*100:.1f}% | MAE {mean_absolute_error(all_true_cccs, all_pred_cccs):.2f}")
if __name__ == "__main__":
    from pathlib import Path

    print("\n--- INITIALIZING RELIMB AI TRAINING ---")
    
    # 1. Define your real paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
    INDEX_FILE = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
    MAPPING_FILE = PROJECT_ROOT / "data" / "class_mapping.json"

    # 2. Load the Class Mapping to know the total number of classes
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)

    # 3. Load the Metadata and build the inputs for the Dataset
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    session_folders = []
    labels_dict = {}

    for item in metadata:
        raw_id = item["ID"]
        clean_id = raw_id.replace(".mp4", "").replace(".avi", "")
        
        folder_inside = f"inside_{clean_id}"
        folder_outside = f"outside_{clean_id}"
        
        kp_path_inside = SESSIONS_DIR / folder_inside / "keypoints.npy"
        kp_path_outside = SESSIONS_DIR / folder_outside / "keypoints.npy"
        
        actual_folder = None
        if kp_path_inside.exists():
            actual_folder = folder_inside
        elif kp_path_outside.exists():
            actual_folder = folder_outside
            
        if actual_folder:
            # SAFETY CHECK: Get the score, but handle NoneType
            raw_ccc = item.get("ccc_score")
            if raw_ccc is None or str(raw_ccc).strip().lower() == 'nan':
                ccc_score = 0.0  
            else:
                ccc_score = float(raw_ccc)
                import math
                if math.isnan(ccc_score): ccc_score = 0.0

            # SAFETY CHECK: Handle primary issue similarly
            issue_text = item.get("clean_primary_issue")
            if not issue_text:
                issue_text = "Unknown / Other"
            
            class_id = class_mapping.get(issue_text, class_mapping.get("Unknown / Other", 0))
            
            full_folder_path = str(SESSIONS_DIR / actual_folder)
            session_folders.append(full_folder_path)
            
            labels_dict[actual_folder] = {
                "class_id": class_id,
                "ccc_score": ccc_score
            }

    print(f"✅ Found {len(session_folders)} valid training samples.")
    print(f"🧠 Total Output Classes: {num_classes}")

    # 4. Instantiate the Dataset
    dataset = ProGaitDataset(session_folders, labels_dict)

    # 5. Run the 5-Minute Test
    train_and_evaluate_quick_test(dataset, num_classes, mode="fusion", epochs=5)