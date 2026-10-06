import os
import json
import time
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================================================================
# 1. Global Configurations and File Paths
# =========================================================================
BASE_DIR = r"c:\Users\vince\Desktop\NN"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets") if os.path.exists(os.path.join(BASE_DIR, "datasets")) else (
    os.path.join(BASE_DIR, "Datasets") if os.path.exists(os.path.join(BASE_DIR, "Datasets")) else "datasets"
)
PLOTS_DIR = os.path.join(BASE_DIR, "triage_pareto_plots")
WEIGHTS_DIR = os.path.join(BASE_DIR, "models_weights")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 4 Clinical feature groups available in the dataset repository
FEATURE_GROUPS = {
    "AV": "26_av",
    "Anemia": "anemia",
    "Metabolismo": "metabolismo",
    "Nutrizione": "nutrizione"
}

WINDOWS = [30, 60]
TARGET_CONFIGS = [
    {"target_str": "capped360", "is_capped": True},
    {"target_str": "uncapped",  "is_capped": False}
]

SPLIT_MODES = ["split_patient", "split_temporal"]
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 2e-2
EARLY_STOPPING_PATIENCE = 40

TTE_TRAIN_CAP_DAYS = 400.0

T_RED = 180.0
T_GREEN = 360.0
KM_EVAL_RED = 180.0
KM_EVAL_GREEN = 360.0
CENSORING_HORIZON_DAYS = 360.0
KM_MIN_DISPLAY_DAYS = 1000.0
MIN_TRIAGE_SAMPLES = 1000


# =========================================================================
# 2. Scalable Feed-Forward Neural Network (N -> H -> 1)
# =========================================================================
class SystematicFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.20, noise_std=0.03):
        super().__init__()
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim
        
        if hidden_dim == 1:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.net(x)


# =========================================================================
# 3. Patient Age Metadata & Raw Ingestion
# =========================================================================
def load_patient_age_map(datasets_dir):
    candidates = [
        os.path.join(datasets_dir, "patient_age.json"),
        os.path.join(BASE_DIR, "Datasets", "patient_age.json"),
        os.path.join(BASE_DIR, "datasets", "patient_age.json"),
        os.path.join(BASE_DIR, "patient_age.json"),
        os.path.join("Datasets", "patient_age.json"),
        os.path.join("datasets", "patient_age.json"),
        "patient_age.json",
    ]
    age_file = next((c for c in candidates if os.path.exists(c)), None)
    if age_file is None:
        return {}

    try:
        with open(age_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        age_map = {}
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    val = float(v) if not isinstance(v, dict) else float(v.get('age', v.get('eta', np.nan)))
                    age_map[str(k)] = val
                    if str(k).isdigit():
                        age_map[int(k)] = val
                except (ValueError, TypeError):
                    continue
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    pid = item.get('patient_id', item.get('partient_id', item.get('id', None)))
                    age = item.get('age', item.get('eta', item.get('età', None)))
                    if pid is not None and age is not None:
                        try:
                            val = float(age)
                            age_map[str(pid)] = val
                            if str(pid).isdigit():
                                age_map[int(pid)] = val
                        except (ValueError, TypeError):
                            continue
        return age_map
    except Exception as e:
        print(f"[DATA LOADER] Error reading {age_file}: {e}")
        return {}


def find_dataset_file(datasets_dir, base_name):
    candidates = [
        os.path.join(datasets_dir, base_name),
        os.path.join(datasets_dir, base_name + ".csv"),
        os.path.join(datasets_dir, base_name + ".CSV"),
        os.path.join(BASE_DIR, "Datasets", base_name + ".csv"),
        os.path.join(BASE_DIR, "datasets", base_name + ".csv"),
        os.path.join(BASE_DIR, base_name + ".csv"),
        os.path.join(".", base_name),
        os.path.join(".", base_name + ".csv"),
    ]
    return next((c for c in candidates if os.path.exists(c)), os.path.join(datasets_dir, base_name + ".csv"))


def load_dataset(file_path):
    t0 = time.time()
    df = pd.read_csv(file_path)
    pid_col = 'partient_id' if 'partient_id' in df.columns else 'patient_id'
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if pid_col in df.columns:
            df = df.sort_values(by=[pid_col, 'timestamp']).reset_index(drop=True)
    return df, pid_col


def prepare_raw_data(df, pid_col, is_capped, age_map):
    pid_all = df[pid_col].values
    time_all = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
    y_raw_days = df['tte'].values.astype(np.float32)
    
    age_col = next((c for c in ['age', 'eta', 'età', 'patient_age', 'eta_paziente', 'age_years', 'anni'] if c in df.columns), None)
    if age_col is not None:
        age_all = df[age_col].values.astype(np.float32)
    elif age_map:
        age_all = np.array([age_map.get(p, age_map.get(str(p), np.nan)) for p in pid_all], dtype=np.float32)
    else:
        age_all = np.full(len(df), np.nan, dtype=np.float32)

    if is_capped:
        y_train_days = np.clip(y_raw_days, 1.0, TTE_TRAIN_CAP_DAYS)
    else:
        y_train_days = np.maximum(y_raw_days, 1.0)
    y_log = np.log(y_train_days)
    
    event_col = next((c for c in ['event', 'status', 'evento', 'failure', 'observed', 'censor', 'censored'] if c in df.columns), None)
    if event_col is not None:
        events_all = df[event_col].values.astype(np.int32)
        if 'censor' in event_col.lower():
            events_all = 1 - events_all
    else:
        events_all = (y_raw_days < CENSORING_HORIZON_DAYS).astype(np.int32)

    # Parse feature array
    all_str = ",".join(df['misure'].str.strip('{}'))
    parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
    X_raw = parsed.reshape(len(df), -1)
    
    # Extract feature column names if available, otherwise None
    feature_names = None
    for cand in ['feature_names', 'nomi_misure', 'nomi_features', 'columns_misure']:
        if cand in df.columns:
            try:
                first_val = df[cand].iloc[0]
                if isinstance(first_val, str):
                    feature_names = [s.strip(" '\"{}[]") for s in first_val.split(",")]
                elif isinstance(first_val, (list, tuple)):
                    feature_names = list(first_val)
            except Exception:
                feature_names = None
            break

    return X_raw, feature_names, y_log, y_raw_days, events_all, age_all, pid_all, time_all


def create_split_masks(pid_all, time_all, split_mode="split_temporal", train_ratio=0.5, val_ratio=0.1):
    n_samples = len(pid_all)
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    unique_pids = np.unique(pid_all)
    
    if split_mode in ["split_patient", "patient_wise"]:
        n_pids = len(unique_pids)
        rng = np.random.RandomState(42)
        shuffled = rng.permutation(unique_pids)
        n_tr = int(n_pids * train_ratio)
        n_va = int(n_pids * val_ratio)
        
        train_pids = set(shuffled[:n_tr])
        val_pids = set(shuffled[n_tr : n_tr + n_va])
        test_pids = set(shuffled[n_tr + n_va :])
        
        for idx, pid in enumerate(pid_all):
            if pid in train_pids:
                train_mask[idx] = True
            elif pid in val_pids:
                val_mask[idx] = True
            elif pid in test_pids:
                test_mask[idx] = True
                
    elif split_mode in ["split_temporal", "temporal"]:
        for pid in unique_pids:
            p_indices = np.where(pid_all == pid)[0]
            n_p = len(p_indices)
            n_tr = int(n_p * train_ratio)
            n_va = int(n_p * val_ratio)
            train_mask[p_indices[:n_tr]] = True
            val_mask[p_indices[n_tr : n_tr + n_va]] = True
            test_mask[p_indices[n_tr + n_va :]] = True
            
    return train_mask, val_mask, test_mask


# =========================================================================
# 4. Feature Combination Deduplication & (a, b, c, z) Extraction
# =========================================================================
def merge_and_deduplicate_raw_features(group_list, group_cache):
    """
    Concatenates raw measurements across selected groups ensuring that if any
    parameter is present in multiple groups, it is included ONLY ONCE.
    """
    if len(group_list) == 1:
        g = group_list[0]
        return group_cache[g]["X"], group_cache[g].get("names", None)
        
    collected_matrices = []
    seen_names = set()
    collected_names = []
    has_all_names = all(group_cache[g].get("names") is not None for g in group_list)

    if has_all_names:
        # Deduplication based on explicit column/measurement names
        for g in group_list:
            X_g = group_cache[g]["X"]
            names_g = group_cache[g]["names"]
            keep_indices = []
            for col_idx, name in enumerate(names_g):
                clean_name = str(name).strip().lower()
                if clean_name not in seen_names:
                    seen_names.add(clean_name)
                    keep_indices.append(col_idx)
                    collected_names.append(name)
            if keep_indices:
                collected_matrices.append(X_g[:, keep_indices])
        X_merged = np.concatenate(collected_matrices, axis=1) if collected_matrices else group_cache[group_list[0]]["X"]
        return X_merged, collected_names
    else:
        # Deduplication based on exact numerical column equality
        merged_cols = []
        for g in group_list:
            X_g = group_cache[g]["X"]
            n_cols = X_g.shape[1]
            for j in range(n_cols):
                candidate_col = X_g[:, j]
                is_duplicate = False
                for existing_col in merged_cols:
                    if np.allclose(candidate_col, existing_col, rtol=1e-5, atol=1e-5, equal_nan=True):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    merged_cols.append(candidate_col)
        X_merged = np.column_stack(merged_cols)
        return X_merged, None


def extract_features(X_all, y_log, y_days, events_all, age_all, pid_all, target_mask, W):
    """
    Computes linear regression features over temporal sliding window W for each of the M measures:
      - a: slope (trend over W sessions)
      - b: intercept (at window center)
      - c_rmse: residual root mean squared error (fluctuation/instability around trend)
      - z_curr: current instantaneous measurement at session t
    Output feature dimension N = 4 * M_raw.
    """
    M_raw = X_all.shape[1]
    
    # If file already contains the precomputed (a, b, c, z) representation
    if M_raw in [104, 176, 208]:
        valid_indices = np.where(target_mask)[0]
        return (X_all[valid_indices], 
                y_log[valid_indices].reshape(-1, 1), 
                y_days[valid_indices].reshape(-1, 1),
                events_all[valid_indices].reshape(-1, 1),
                age_all[valid_indices].reshape(-1, 1))
        
    N_total = len(pid_all)
    pids_arr = np.array(pid_all)
    same_patient = (pids_arr[W - 1:] == pids_arr[: N_total - W + 1])
    target_in_set = target_mask[W - 1:]
    valid_indices = np.where(same_patient & target_in_set)[0] + (W - 1)
    
    if len(valid_indices) == 0:
        return (np.empty((0, 4 * M_raw), dtype=np.float32), 
                np.empty((0, 1), dtype=np.float32), 
                np.empty((0, 1), dtype=np.float32), 
                np.empty((0, 1), dtype=np.int32),
                np.empty((0, 1), dtype=np.float32))

    t = np.arange(W, dtype=np.float32)
    t_mean = (W - 1) / 2.0
    t_dev = t - t_mean
    sum_t_dev_sq = np.sum(t_dev**2)
    
    windows = np.zeros((len(valid_indices), W, M_raw), dtype=np.float32)
    for idx_out, target_idx in enumerate(valid_indices):
        windows[idx_out] = X_all[target_idx - W + 1 : target_idx + 1]
        
    z_mean = np.mean(windows, axis=1)
    a = np.sum(t_dev[:, None] * windows, axis=1) / sum_t_dev_sq
    b = z_mean - a * t_mean
    y_line = a[:, None, :] * t[None, :, None] + b[:, None, :]
    residuals = y_line - windows
    c_rmse = np.sqrt(np.mean(residuals**2, axis=1))
    z_curr = windows[:, -1, :]
    
    # Concatenate (a, b, c, z) along feature axis -> shape: (num_samples, 4 * M_raw)
    X_features = np.concatenate([a, b, c_rmse, z_curr], axis=1)
    
    return (X_features, 
            y_log[valid_indices].reshape(-1, 1), 
            y_days[valid_indices].reshape(-1, 1),
            events_all[valid_indices].reshape(-1, 1),
            age_all[valid_indices].reshape(-1, 1))


# =========================================================================
# 5. PyTorch Dataset and Training Pipeline
# =========================================================================
class TabularLogDataset(Dataset):
    def __init__(self, X_tab, y_log, y_days, events, ages):
        self.X = torch.tensor(X_tab, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32)
        self.y_days = torch.tensor(y_days, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.int32)
        self.ages = torch.tensor(ages, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_log[idx], self.y_days[idx], self.events[idx], self.ages[idx]


def evaluate_model(model, data_loader, device, is_capped):
    model.eval()
    all_preds_days, all_true_days, all_events, all_ages = [], [], [], []
    
    with torch.no_grad():
        for X_b, _, y_days_b, events_b, ages_b in data_loader:
            X_b = X_b.to(device)
            preds_days = torch.exp(model(X_b))
            if is_capped:
                preds_days = torch.clamp(preds_days, max=TTE_TRAIN_CAP_DAYS)
                
            all_preds_days.append(preds_days.cpu().numpy().flatten())
            all_true_days.append(y_days_b.numpy().flatten())
            all_events.append(events_b.numpy().flatten())
            all_ages.append(ages_b.numpy().flatten())
            
    all_preds = np.concatenate(all_preds_days)
    all_true = np.concatenate(all_true_days)
    all_events = np.concatenate(all_events)
    all_ages = np.concatenate(all_ages)
    
    mae = float(np.mean(np.abs(all_preds - all_true)))
    denom = np.clip(all_true, 10.0, None)
    mape = float(np.mean(np.abs(all_preds - all_true) / denom) * 100.0)
    
    return mae, mape, all_preds, all_true, all_events, all_ages


def train_model(model, train_loader, val_loader, test_loader, device, is_capped, epochs=EPOCHS):
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_mae = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_log_b, _, _, _ in train_loader:
            X_b, y_log_b = X_b.to(device), y_log_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_log_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        val_mae, _, _, _, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
                
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        
    tr_mae, tr_mape, _, _, _, _ = evaluate_model(model, train_loader, device, is_capped=is_capped)
    va_mae, va_mape, _, _, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
    te_mae, te_mape, te_preds, te_true, te_events, te_ages = evaluate_model(model, test_loader, device, is_capped=is_capped)
    
    return tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, te_preds, te_true, te_events, te_ages


def check_architectural_pruning(history, max_patience=2, gap_threshold=35.0, gap_growth_ratio=1.35):
    if len(history) < 2:
        return False, ""
    curr = history[-1]
    prev = history[-2]
    gap_curr = abs(curr["val_mae"] - curr["train_mae"])
    gap_prev = abs(prev["val_mae"] - prev["train_mae"])
    
    if gap_curr > gap_threshold and (gap_prev == 0 or gap_curr > gap_growth_ratio * gap_prev):
        return True, f"Overfitting detected: |Val-Train| gap widened to {gap_curr:.1f}d (> {gap_threshold}d)."
        
    best_val = min(h["val_mae"] for h in history[:-1])
    if len(history) >= max_patience + 1:
        recent_vals = [h["val_mae"] for h in history[-max_patience:]]
        if all(v >= best_val + 2.0 for v in recent_vals):
            return True, f"Validation stagnation relative to minimum ({best_val:.2f}d)."
            
    return False, ""


def compute_kaplan_meier_curve(durations, events):
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.int32)
    order = np.lexsort((-events, durations))
    durations, events = durations[order], events[order]
    
    unique_times = np.unique(durations)
    n_at_risk = len(durations)
    survival_prob = 1.0
    km_t, km_s = [0.0], [1.0]
    
    for t in unique_times:
        n_events_t = np.sum((durations == t) & (events == 1))
        n_censored_t = np.sum((durations == t) & (events == 0))
        if n_at_risk > 0 and n_events_t > 0:
            survival_prob *= (1.0 - n_events_t / n_at_risk)
        km_t.append(float(t))
        km_s.append(float(survival_prob))
        n_at_risk -= (n_events_t + n_censored_t)
        
    return np.array(km_t), np.array(km_s)


def evaluate_triage_and_kaplan_meier(y_test_true, y_test_pred, y_test_events, test_ages):
    y_test_true = np.asarray(y_test_true, dtype=np.float32)
    y_test_pred = np.asarray(y_test_pred, dtype=np.float32)
    y_test_events = np.asarray(y_test_events, dtype=np.int32)
    test_ages = np.asarray(test_ages, dtype=np.float32)
    
    mask_red = y_test_pred < T_RED
    mask_yellow = (y_test_pred >= T_RED) & (y_test_pred < T_GREEN)
    mask_green = y_test_pred >= T_GREEN
    
    if np.sum(mask_red) > 0:
        t_r, s_r = compute_kaplan_meier_curve(y_test_true[mask_red], y_test_events[mask_red])
        idx = np.searchsorted(t_r, KM_EVAL_RED, side='right') - 1
        p_r = float(s_r[np.clip(idx, 0, len(s_r) - 1)])
    else:
        t_r, s_r = np.array([0.0, KM_EVAL_RED]), np.array([1.0, 1.0])
        p_r = 1.0
        
    if np.sum(mask_green) > 0:
        t_g, s_g = compute_kaplan_meier_curve(y_test_true[mask_green], y_test_events[mask_green])
        idx = np.searchsorted(t_g, KM_EVAL_GREEN, side='right') - 1
        p_g = float(s_g[np.clip(idx, 0, len(s_g) - 1)])
    else:
        t_g, s_g = np.array([0.0, KM_EVAL_GREEN]), np.array([0.0, 0.0])
        p_g = 0.0
        
    t_y, s_y = compute_kaplan_meier_curve(y_test_true[mask_yellow], y_test_events[mask_yellow]) if np.sum(mask_yellow) > 0 else (np.array([0.0]), np.array([1.0]))
    p_bar_g = 1.0 - p_g
    
    has_valid_age = not np.all(np.isnan(test_ages))
    if has_valid_age:
        age_red = round(float(np.nanmean(test_ages[mask_red])), 1) if np.sum(mask_red) > 0 else np.nan
        age_yellow = round(float(np.nanmean(test_ages[mask_yellow])), 1) if np.sum(mask_yellow) > 0 else np.nan
        age_green = round(float(np.nanmean(test_ages[mask_green])), 1) if np.sum(mask_green) > 0 else np.nan
        age_total = round(float(np.nanmean(test_ages)), 1) if len(test_ages) > 0 else np.nan
    else:
        age_red, age_yellow, age_green, age_total = np.nan, np.nan, np.nan, np.nan
        
    return {
        "P_R": round(p_r, 4),
        "P_G": round(p_g, 4),
        "P_bar_G": round(p_bar_g, 4),
        "Sessions_Red": int(np.sum(mask_red)),
        "Sessions_Yellow": int(np.sum(mask_yellow)),
        "Sessions_Green": int(np.sum(mask_green)),
        "Total_Sessions": len(y_test_true),
        "Age_Red": age_red,
        "Age_Yellow": age_yellow,
        "Age_Green": age_green,
        "Age_Total": age_total,
        "km_curves": {"Red": (t_r, s_r), "Yellow": (t_y, s_y), "Green": (t_g, s_g)}
    }


def compute_pareto_mask(p_bar_g_arr, p_r_arr):
    pts = np.column_stack([p_bar_g_arr, p_r_arr])
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1]) and \
                   (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1]):
                    is_pareto[i] = False
                    break
    return is_pareto


# =========================================================================
# 6. Plotting Routines
# =========================================================================
def plot_kaplan_meier_curves(km_data, title, save_path, min_x_extent=KM_MIN_DISPLAY_DAYS):
    plt.figure(figsize=(9.5, 5.5))
    colors = {'Green': '#2ca02c', 'Yellow': '#ff7f0e', 'Red': '#d62728'}
    
    def format_legend(name, threshold_str, count_key, age_key):
        count = km_data[count_key]
        age_val = km_data[age_key]
        age_str = f", Mean Age={age_val:.1f}y" if not np.isnan(age_val) else ""
        return f"{name} ({threshold_str}, Sessions={count:,}{age_str})"

    labels = {
        'Green': format_legend('Green', f"Pred >= {int(T_GREEN)}d", 'Sessions_Green', 'Age_Green'),
        'Yellow': format_legend('Yellow', f"{int(T_RED)}-{int(T_GREEN)}d", 'Sessions_Yellow', 'Age_Yellow'),
        'Red': format_legend('Red', f"Pred < {int(T_RED)}d", 'Sessions_Red', 'Age_Red')
    }
    line_widths = {'Green': 2.5, 'Yellow': 2.0, 'Red': 2.5}
    
    observed_max_times = [np.max(km_data["km_curves"][c][0]) for c in ['Green', 'Yellow', 'Red']]
    max_observed = float(np.max(observed_max_times))
    x_axis_limit = max(min_x_extent, max_observed)
    
    for c in ['Green', 'Yellow', 'Red']:
        t_c, s_c = km_data["km_curves"][c]
        t_plot = list(t_c)
        s_plot = list(s_c)
        if t_plot[-1] < x_axis_limit:
            t_plot.append(x_axis_limit)
            s_plot.append(s_plot[-1])
            
        plt.step(t_plot, s_plot, label=labels[c], color=colors[c], linewidth=line_widths[c], where='post')
    
    p_r = km_data["P_R"]
    p_g = km_data["P_G"]
    
    plt.axvline(x=KM_EVAL_RED, color="gray", linestyle="--", alpha=0.7)
    plt.scatter([KM_EVAL_RED], [p_r], color="#d62728", s=85, zorder=5)
    plt.annotate(f"$P_R = {p_r:.2f}$ $\\downarrow$", xy=(KM_EVAL_RED, p_r), xytext=(KM_EVAL_RED, p_r - 0.12),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8),
                 ha="center", fontsize=11, fontweight="bold", color="#d62728")
                 
    plt.axvline(x=KM_EVAL_GREEN, color="gray", linestyle="--", alpha=0.7)
    plt.scatter([KM_EVAL_GREEN], [p_g], color="#2ca02c", s=85, zorder=5)
    plt.annotate(f"$P_G = {p_g:.2f}$ $\\uparrow$", xy=(KM_EVAL_GREEN, p_g), xytext=(KM_EVAL_GREEN, min(1.0, p_g + 0.10)),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.8),
                 ha="center", fontsize=11, fontweight="bold", color="#2ca02c")
                 
    plt.xlabel("Observed Survival Time (Days)", fontsize=11)
    plt.ylabel("Survival Probability S(t)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, x_axis_limit)
    
    x_ticks = [0, 180, 360, 500, 750, 1000]
    if x_axis_limit > 1050:
        x_ticks.append(int(x_axis_limit))
    plt.xticks(x_ticks, labels=[str(xt) for xt in x_ticks])
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_global_pareto_clean(df_all, save_path):
    valid_df = df_all[df_all["Clinically_Valid"]].copy()
    pareto_pts = valid_df[valid_df["is_pareto"]].sort_values(by="P_bar_G").reset_index(drop=True)
    dominated_pts = valid_df[~valid_df["is_pareto"]].copy()

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)

    feature_colors = {1: '#4A90E2', 2: '#F5A623', 3: '#7ED321', 4: '#9013FE'}
    feature_markers = {1: 'o', 2: 's', 3: '^', 4: 'D'}

    for size in [1, 2, 3, 4]:
        sub = dominated_pts[dominated_pts['Combo_Size'] == size]
        if not sub.empty:
            ax.scatter(sub['P_bar_G'], sub['P_R'], 
                       color=feature_colors[size], 
                       marker=feature_markers[size],
                       alpha=0.45, s=36, edgecolors='none', 
                       label=f'{size} Feature Group{"s" if size > 1 else ""} ({len(sub)} models)')

    # Red Pareto connection line
    if not pareto_pts.empty:
        ax.plot(pareto_pts['P_bar_G'], pareto_pts['P_R'], 
                color='#D0021B', linestyle='-', linewidth=2.4, zorder=5, 
                label='Global Pareto Frontier (Non-Dominated)')

        ax.scatter(pareto_pts['P_bar_G'], pareto_pts['P_R'], 
                   color='#D0021B', s=115, edgecolors='black', linewidth=1.2, zorder=6)

        for idx, r in pareto_pts.iterrows():
            tag = f"#{int(r['Model_Num'])}"
            ox = 14 if idx % 2 == 0 else -28
            oy = 14 if idx % 2 == 0 else -16
            ha = 'left' if ox > 0 else 'right'
            ax.annotate(
                tag,
                (r["P_bar_G"], r["P_R"]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=9.5,
                fontweight="bold",
                color='#900000',
                ha=ha,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#D0021B", lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#D0021B", lw=1.0)
            )

    # Utopia reference point
    ax.scatter([0], [0], color='#0052CC', marker='*', s=260, zorder=7, label='Ideal Utopia Point (0, 0)')

    # Axis limits locked at 0.6
    ax.set_xlim(-0.015, 0.6)
    ax.set_ylim(-0.015, 0.6)

    ax.set_xlabel(r"$\bar{P}_G = 1 - P_G$ (1-Year Failure Rate at 360d $\to 0$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$P_R$ (6-Month False Survival Rate at 180d $\to 0$)", fontsize=11, fontweight="bold")
    ax.set_title("Global Pareto Frontier (Clinically Valid Models Only)", fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", frameon=True, fontsize=9.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pareto_winners_comparison(winners_df, title, save_path):
    plt.figure(figsize=(11, 7.5))
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D'}
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}
    
    for size, group in winners_df.groupby("Combo_Size"):
        plt.scatter(
            group["P_bar_G"], group["P_R"], 
            s=140, marker=markers.get(size, 'o'), 
            color=colors.get(size, '#333333'),
            label=f"{size} Feature Group(s)", edgecolors='black', linewidth=1.2, zorder=5
        )
        for _, r in group.iterrows():
            plt.annotate(
                f"{r['Feature_Combo']}\n(MAE: {r['Test_MAE_days']}d)",
                (r["P_bar_G"], r["P_R"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8, fontweight="semibold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", fc="#fdfdfd", ec="#888888", lw=0.8, alpha=0.85)
            )

    plt.scatter([0], [0], color="gold", edgecolors="black", marker="*", s=260, zorder=6, label="Ideal Utopia (0, 0)")
    plt.xlabel(r"$\bar{P}_G = 1 - P_G$ (Green Class Error $\to 0$)", fontsize=12, fontweight="bold")
    plt.ylabel(r"$P_R$ (Red Class False Survival $\to 0$)", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlim(-0.04, 0.6)
    plt.ylim(-0.04, 0.6)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================================================================
# 7. Main Pipeline Execution
# =========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print(" FFNN PIPELINE: TEMPORAL FEATURE EXTRACTION (a, b, c, z) + DEDUPLICATION")
    print(f" DEVICE: {device} | CLINICAL SAMPLE THRESHOLD PER CLASS >= {MIN_TRIAGE_SAMPLES}")
    print("=" * 95 + "\n")
    
    age_map = load_patient_age_map(DATASETS_DIR)
    
    group_names = list(FEATURE_GROUPS.keys())
    all_combinations = []
    for r in [1, 2, 3, 4]:
        for combo in itertools.combinations(group_names, r):
            all_combinations.append(combo)

    all_results = []
    all_km_cache = {}
    model_counter = 0

    for W in WINDOWS:
        for tgt_cfg in TARGET_CONFIGS:
            target_str = tgt_cfg["target_str"]
            is_capped = tgt_cfg["is_capped"]
            
            print("\n" + "#" * 95)
            print(f" INGESTING GROUP BASELINE ARRAYS: Window W={W}d | Target={target_str.upper()}")
            print("#" * 95)
            
            group_cache = {}
            for g_name, file_key in FEATURE_GROUPS.items():
                base_name = f"NN_training_dataset_W{W}_{file_key}_{target_str}"
                file_path = find_dataset_file(DATASETS_DIR, base_name)
                if not os.path.exists(file_path):
                    continue
                df_raw, pid_col = load_dataset(file_path)
                X_raw, f_names, y_log, y_raw_days, events_all, age_all, pid_all, time_all = prepare_raw_data(
                    df_raw, pid_col, is_capped=is_capped, age_map=age_map
                )
                group_cache[g_name] = {
                    "X": X_raw,
                    "names": f_names,
                    "y_log": y_log,
                    "y_days": y_raw_days,
                    "events": events_all,
                    "age": age_all,
                    "pid": pid_all,
                    "time": time_all
                }

            for combo in all_combinations:
                combo_str = "+".join(combo)
                combo_size = len(combo)
                
                if not all(g in group_cache for g in combo):
                    continue
                
                ref_g = combo[0]
                pid_all = group_cache[ref_g]["pid"]
                time_all = group_cache[ref_g]["time"]
                y_log = group_cache[ref_g]["y_log"]
                y_days = group_cache[ref_g]["y_days"]
                events_all = group_cache[ref_g]["events"]
                age_all = group_cache[ref_g]["age"]
                
                # Merge and deduplicate raw measurements across selected clinical groups
                X_combo_dedup, _ = merge_and_deduplicate_raw_features(combo, group_cache)
                M_unique = X_combo_dedup.shape[1]
                
                for split_mode in SPLIT_MODES:
                    train_mask, val_mask, test_mask = create_split_masks(pid_all, time_all, split_mode=split_mode)
                    
                    # EXTRACT (a, b, c_rmse, z_curr) OVER TEMPORAL SLIDING WINDOW W -> SHAPE: (samples, 4 * M_unique)
                    X_tr, y_tr_log, y_tr_d, ev_tr, age_tr = extract_features(
                        X_combo_dedup, y_log, y_days, events_all, age_all, pid_all, train_mask, W
                    )
                    X_va, y_va_log, y_va_d, ev_va, age_va = extract_features(
                        X_combo_dedup, y_log, y_days, events_all, age_all, pid_all, val_mask, W
                    )
                    X_te, y_te_log, y_te_d, ev_te, age_te = extract_features(
                        X_combo_dedup, y_log, y_days, events_all, age_all, pid_all, test_mask, W
                    )
                    
                    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                        continue
                        
                    # Standardize strictly using training set statistics
                    mean = X_tr.mean(axis=0, keepdims=True)
                    std = X_tr.std(axis=0, keepdims=True)
                    std[std == 0] = 1.0
                    
                    X_tr_s = (X_tr - mean) / std
                    X_va_s = (X_va - mean) / std
                    X_te_s = (X_te - mean) / std
                    
                    train_loader = DataLoader(TabularLogDataset(X_tr_s, y_tr_log, y_tr_d, ev_tr, age_tr), batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(TabularLogDataset(X_va_s, y_va_log, y_va_d, ev_va, age_va), batch_size=BATCH_SIZE, shuffle=False)
                    test_loader = DataLoader(TabularLogDataset(X_te_s, y_te_log, y_te_d, ev_te, age_te), batch_size=BATCH_SIZE, shuffle=False)
                    
                    # N = 4 * M_unique (4 inputs per distinct parameter: a, b, c, z)
                    N = X_tr_s.shape[1]
                    base_steps = [1, 2, 3, 5, 8, 16, 32, 64, 128]
                    candidate_H = sorted(list(set([h for h in base_steps if h < N] + [N])))
                    
                    print(f"\n[{combo_str}] Unique M={M_unique} -> Inputs N={N} (4 x M). Window W={W}, Split={split_mode}")
                    
                    arch_history = []
                    
                    for h in candidate_H:
                        model_counter += 1
                        model_num = model_counter
                        model_id = f"{combo_str}_W{W}_H{h}_{target_str}_{split_mode}"
                        
                        model = SystematicFFNN(input_dim=N, hidden_dim=h).to(device)
                        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        
                        tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, te_preds, te_true, te_events, te_ages = train_model(
                            model, train_loader, val_loader, test_loader, device, is_capped=is_capped
                        )
                        
                        weights_path = os.path.join(WEIGHTS_DIR, f"Model_{model_num}_{model_id}.pt")
                        torch.save(model.state_dict(), weights_path)
                        
                        triage_metrics = evaluate_triage_and_kaplan_meier(te_true, te_preds, te_events, te_ages)
                        all_km_cache[model_num] = (model_id, triage_metrics)
                        
                        diff_mae = abs(te_mae - tr_mae)
                        diff_mape = abs(te_mape - tr_mape)
                        overfitting_ratio = te_mae / (tr_mae + 1e-6)
                        
                        is_clinically_valid = (
                            triage_metrics["Sessions_Red"] >= MIN_TRIAGE_SAMPLES and
                            triage_metrics["Sessions_Yellow"] >= MIN_TRIAGE_SAMPLES and
                            triage_metrics["Sessions_Green"] >= MIN_TRIAGE_SAMPLES
                        )
                        
                        dist_ideal = np.sqrt(triage_metrics["P_bar_G"]**2 + triage_metrics["P_R"]**2)
                        
                        entry = {
                            "Model_Num": model_num,
                            "Model_ID": model_id,
                            "Feature_Combo": combo_str,
                            "Combo_Size": combo_size,
                            "Measures_M": M_unique,
                            "N_inputs": N,  # 4 * M_unique
                            "Window_W": W,
                            "Target_Type": target_str,
                            "Split_Mode": split_mode,
                            "Hidden_H": h,
                            "Parameters": num_params,
                            "Train_MAE_days": round(tr_mae, 2),
                            "Test_MAE_days": round(te_mae, 2),
                            "Diff_MAE_days": round(diff_mae, 2),
                            "Train_MAPE": round(tr_mape, 2),
                            "Test_MAPE": round(te_mape, 2),
                            "Diff_MAPE": round(diff_mape, 2),
                            "Val_MAE_days": round(va_mae, 2),
                            "Val_MAPE": round(va_mape, 2),
                            "Overfitting_Ratio": round(overfitting_ratio, 3),
                            "Clinically_Valid": is_clinically_valid,
                            "Sessions_Red": triage_metrics["Sessions_Red"],
                            "Sessions_Yellow": triage_metrics["Sessions_Yellow"],
                            "Sessions_Green": triage_metrics["Sessions_Green"],
                            "Total_Sessions": triage_metrics["Total_Sessions"],
                            "P_R": triage_metrics["P_R"],
                            "P_G": triage_metrics["P_G"],
                            "P_bar_G": triage_metrics["P_bar_G"],
                            "Dist_Ideal": round(float(dist_ideal), 4),
                            "Age_Red": triage_metrics["Age_Red"],
                            "Age_Yellow": triage_metrics["Age_Yellow"],
                            "Age_Green": triage_metrics["Age_Green"],
                            "Age_Total": triage_metrics["Age_Total"]
                        }
                        all_results.append(entry)
                        arch_history.append({"H": h, "train_mae": tr_mae, "val_mae": va_mae, "test_mae": te_mae})
                        
                        status = "VALID" if is_clinically_valid else f"FILTERED (< {MIN_TRIAGE_SAMPLES})"
                        print(f"  [#{model_num:3d}: H={h:3d}] | Tr/Te MAE: {tr_mae:4.1f}/{te_mae:4.1f}d | P_R: {triage_metrics['P_R']:.2f}, P_G: {triage_metrics['P_G']:.2f} [{status}]")
                        
                        should_prune, reason = check_architectural_pruning(arch_history)
                        if should_prune:
                            print(f"       ==> Pruning H for '{combo_str}': {reason}")
                            break

    df_all = pd.DataFrame(all_results)
    if df_all.empty:
        return

    df_all["is_pareto"] = False
    valid_mask = df_all["Clinically_Valid"]
    valid_indices = df_all[valid_mask].index
    
    if len(valid_indices) > 0:
        valid_p_bar_g = df_all.loc[valid_indices, "P_bar_G"].values
        valid_p_r = df_all.loc[valid_indices, "P_R"].values
        pareto_sub_mask = compute_pareto_mask(valid_p_bar_g, valid_p_r)
        actual_pareto_indices = valid_indices[pareto_sub_mask]
        df_all.loc[actual_pareto_indices, "is_pareto"] = True
        
    # Winner selection per feature combination
    winners = []
    for combo_str, group in df_all.groupby("Feature_Combo"):
        valid_group = group[group["Clinically_Valid"]].copy()
        if valid_group.empty:
            continue
        local_mask = compute_pareto_mask(valid_group["P_bar_G"].values, valid_group["P_R"].values)
        local_pareto = valid_group[local_mask].sort_values(by=["Dist_Ideal", "Test_MAE_days"])
        winners.append(local_pareto.iloc[0].to_dict())
        
    winners_df = pd.DataFrame(winners).sort_values(by="Dist_Ideal").reset_index(drop=True)

    # Save full CSV with symmetrical train/test metrics
    df_all.to_csv(os.path.join(BASE_DIR, "final_combinatorial_ffnn_pareto_results.csv"), index=False)
    winners_df.to_csv(os.path.join(BASE_DIR, "pareto_winners_comparison.csv"), index=False)
    print("\n[EXPORT] Master results and winners comparison CSVs saved successfully.")

    # =========================================================================
    # EXPORT KAPLAN-MEIER CURVES (.PNG) FOR EVALUATED MODELS
    # =========================================================================
    print(f"\n[EXPORT] Rendering Kaplan-Meier curves into: {PLOTS_DIR}")
    
    # 1. Export KM plots for all non-dominated Pareto models and winning models
    # (Oppure per tutti i modelli valutati rimuovendo il filtro 'if is_pareto or is_winner')
    winner_nums = set(winners_df["Model_Num"].values) if not winners_df.empty else set()
    
    for model_num, (mid, km_data) in all_km_cache.items():
        row = df_all[df_all["Model_Num"] == model_num].iloc[0]
        is_opt = row["is_pareto"]
        is_winner = model_num in winner_nums
        is_valid = row["Clinically_Valid"]
        
        # Tag per identificare lo stato del modello nel titolo
        status_tag = "[PARETO]" if is_opt else ("[WINNER]" if is_winner else ("[VALID]" if is_valid else "[PRUNED]"))
        
        # Salva la curva KM (se desideri salvarli TUTTI gli 800+ modelli lascia la chiamata diretta):
        km_title = f"{status_tag} Model #{model_num} - {mid}"
        km_filename = os.path.join(PLOTS_DIR, f"KM_Model_{model_num}_{mid}.png")
        plot_kaplan_meier_curves(km_data, km_title, km_filename, min_x_extent=KM_MIN_DISPLAY_DAYS)
        
    print(f"[EXPORT] Successfully saved {len(all_km_cache)} Kaplan-Meier plots in: {PLOTS_DIR}")

    # =========================================================================
    # EXPORT SUMMARY PARETO CHARTS
    # =========================================================================
    clean_pareto_plot_file = os.path.join(PLOTS_DIR, "global_pareto_frontier_0_6_perfect_english.png")
    plot_global_pareto_clean(df_all, clean_pareto_plot_file)
    print(f"[EXPORT] Saved Global Pareto Frontier Plot: {clean_pareto_plot_file}")

    winners_plot_file = os.path.join(PLOTS_DIR, "pareto_winners_comparison.png")
    plot_pareto_winners_comparison(winners_df, "Winning Pareto Models Across Feature Combinations", winners_plot_file)
    print(f"[EXPORT] Saved Winners Comparison Plot: {winners_plot_file}")

    print("\n[EXPORT COMPLETED] All models, Pareto frontiers, and KM curves have been generated.")

if __name__ == "__main__":
    main()