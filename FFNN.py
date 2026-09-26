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

# Windows and Target configurations
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

# Network training saturation threshold (clamping in log-space)
TTE_TRAIN_CAP_DAYS = 400.0

# Clinical Triage cut-offs
T_RED = 180.0
T_GREEN = 360.0
KM_EVAL_RED = 180.0
KM_EVAL_GREEN = 360.0

# Scenario B: Clinical study follow-up threshold (360.0 days)
CENSORING_HORIZON_DAYS = 360.0

# Extended KM horizontal axis display range
KM_MIN_DISPLAY_DAYS = 1000.0

# Clinical cohort size threshold: models with < 1000 sessions in ANY class are pruned upstream
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
# 3. Patient Age Metadata & Raw Data Ingestion
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
    age_file = None
    for c in candidates:
        if os.path.exists(c):
            age_file = c
            break

    if age_file is None:
        print("[DATA LOADER] Warning: 'patient_age.json' not found. Age fields will default to NaN.")
        return {}

    try:
        with open(age_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        age_map = {}
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    age_val = float(v) if not isinstance(v, dict) else float(v.get('age', v.get('eta', np.nan)))
                    age_map[str(k)] = age_val
                    if str(k).isdigit():
                        age_map[int(k)] = age_val
                except (ValueError, TypeError):
                    continue
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    pid = item.get('patient_id', item.get('partient_id', item.get('id', None)))
                    age = item.get('age', item.get('eta', item.get('età', None)))
                    if pid is not None and age is not None:
                        try:
                            age_val = float(age)
                            age_map[str(pid)] = age_val
                            if str(pid).isdigit():
                                age_map[int(pid)] = age_val
                        except (ValueError, TypeError):
                            continue
        print(f"[DATA LOADER] Successfully loaded age metadata for {len(age_map) // 2} patients.")
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
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(datasets_dir, base_name + ".csv")


def load_dataset(file_path):
    t0 = time.time()
    df = pd.read_csv(file_path)
    print(f"  [IO] Loaded '{os.path.basename(file_path)}' ({len(df):,} sessions) in {time.time() - t0:.2f}s.")
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
    
    age_col = None
    for cand in ['age', 'eta', 'età', 'patient_age', 'eta_paziente', 'age_years', 'anni']:
        if cand in df.columns:
            age_col = cand
            break
            
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
    
    event_col = None
    for cand in ['event', 'status', 'evento', 'failure', 'observed', 'censor', 'censored', 'complicanza', 'occlusione']:
        if cand in df.columns:
            event_col = cand
            break
            
    if event_col is not None:
        events_all = df[event_col].values.astype(np.int32)
        if 'censor' in event_col.lower():
            events_all = 1 - events_all
    else:
        events_all = (y_raw_days < CENSORING_HORIZON_DAYS).astype(np.int32)

    all_str = ",".join(df['misure'].str.strip('{}'))
    parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
    X_raw = parsed.reshape(len(df), -1)
    
    return X_raw, y_log, y_raw_days, events_all, age_all, pid_all, time_all


def create_split_masks(pid_all, time_all, split_mode="split_temporal", train_ratio=0.5, val_ratio=0.1):
    n_samples = len(pid_all)
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    unique_pids = np.unique(pid_all)
    
    if split_mode in ["split_patient", "patient_wise"]:
        n_pids = len(unique_pids)
        rng = np.random.RandomState(42)
        shuffled_pids = rng.permutation(unique_pids)
        
        n_tr_pids = int(n_pids * train_ratio)
        n_va_pids = int(n_pids * val_ratio)
        
        train_pids = set(shuffled_pids[:n_tr_pids])
        val_pids = set(shuffled_pids[n_tr_pids : n_tr_pids + n_va_pids])
        test_pids = set(shuffled_pids[n_tr_pids + n_va_pids :])
        
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


def extract_features(X_all, y_log, y_days, events_all, age_all, target_mask):
    valid_indices = np.where(target_mask)[0]
    return (X_all[valid_indices], 
            y_log[valid_indices].reshape(-1, 1), 
            y_days[valid_indices].reshape(-1, 1),
            events_all[valid_indices].reshape(-1, 1),
            age_all[valid_indices].reshape(-1, 1))


# =========================================================================
# 4. PyTorch Dataset with Censoring & Age Vectors
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


# =========================================================================
# 5. Model Evaluation and Symmetric Metric Training Pipeline
# =========================================================================
def evaluate_model(model, data_loader, device, is_capped):
    model.eval()
    all_preds_days = []
    all_true_days = []
    all_events = []
    all_ages = []
    
    with torch.no_grad():
        for X_b, _, y_days_b, events_b, ages_b in data_loader:
            X_b = X_b.to(device)
            preds_log = model(X_b)
            preds_days = torch.exp(preds_log)
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
            out_log = model(X_b)
            loss = criterion(out_log, y_log_b)
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
        
    # Re-run inference across Train, Validation, and Test splits symmetrically
    tr_mae, tr_mape, _, _, _, _ = evaluate_model(model, train_loader, device, is_capped=is_capped)
    va_mae, va_mape, _, _, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
    te_mae, te_mape, test_preds, test_true, test_events, test_ages = evaluate_model(model, test_loader, device, is_capped=is_capped)
    
    return tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, test_preds, test_true, test_events, test_ages


# =========================================================================
# 6. Architectural Pruning Controller
# =========================================================================
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


# =========================================================================
# 7. Kaplan-Meier Survival Analysis with Scenario B Censoring
# =========================================================================
def compute_kaplan_meier_curve(durations, events):
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.int32)
        
    order = np.lexsort((-events, durations))
    durations = durations[order]
    events = events[order]
    
    unique_times = np.unique(durations)
    n_at_risk = len(durations)
    survival_prob = 1.0
    
    km_t = [0.0]
    km_s = [1.0]
    
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
        
    if np.sum(mask_yellow) > 0:
        t_y, s_y = compute_kaplan_meier_curve(y_test_true[mask_yellow], y_test_events[mask_yellow])
    else:
        t_y, s_y = np.array([0.0]), np.array([1.0])
        
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
        "km_curves": {
            "Red": (t_r, s_r),
            "Yellow": (t_y, s_y),
            "Green": (t_g, s_g)
        }
    }


# =========================================================================
# 8. Pareto Optimization Algorithms (Bi-Objective Minimization)
# =========================================================================
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
# 9. Plotting Routines
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


def plot_global_pareto_numbered(df_all, title, save_path):
    valid_df = df_all[df_all["Clinically_Valid"]].copy()
    pareto_pts = valid_df[valid_df["is_pareto"]].sort_values(by="P_bar_G").reset_index(drop=True)
    dominated_pts = valid_df[~valid_df["is_pareto"]]
    
    plt.figure(figsize=(11, 7.5))
    
    if not dominated_pts.empty:
        plt.scatter(dominated_pts["P_bar_G"], dominated_pts["P_R"], color="#a5a5a5", alpha=0.55, s=55, label="Dominated Configurations")
        for _, r in dominated_pts.iterrows():
            plt.annotate(str(int(r["Model_Num"])), (r["P_bar_G"], r["P_R"]),
                         textcoords="offset points", xytext=(3, 3), fontsize=7, color="#555555", alpha=0.85)
            
    if not pareto_pts.empty:
        plt.plot(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", linestyle="-", linewidth=2.2, zorder=4)
        plt.scatter(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", s=130, zorder=5, label="Global Pareto Frontier")
        
        for idx, r in pareto_pts.iterrows():
            model_num_tag = f"#{int(r['Model_Num'])} ({r['Feature_Combo']})"
            offset_y = 12 if idx % 2 == 0 else -20
            offset_x = 10 if r["P_bar_G"] < 0.8 else -45
            
            plt.annotate(
                model_num_tag,
                (r["P_bar_G"], r["P_R"]),
                textcoords="offset points",
                xytext=(offset_x, offset_y),
                fontsize=8.5,
                fontweight="bold",
                color="#b30000",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", lw=1.2, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0)
            )
        
    plt.scatter([0], [0], color="#1f77b4", marker="*", s=220, zorder=6, label="Ideal Point (0, 0)")
    plt.xlabel(r"$\bar{P}_G = 1 - P_G$ (Failure Rate at 360d $\to 0$)", fontsize=12, fontweight="bold")
    plt.ylabel(r"$P_R$ (False Survival Rate at 180d $\to 0$)", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlim(-0.04, 1.06)
    plt.ylim(-0.04, 1.06)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
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
    plt.xlim(-0.04, 1.06)
    plt.ylim(-0.04, 1.06)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================================================================
# 10. Main Pipeline Execution with Feature Combinatorics
# =========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print(" FFNN PIPELINE: COMBINATORIAL FEATURE EVALUATION + SYMMETRIC METRICS")
    print(f" DEVICE: {device} | CLINICAL SAMPLE THRESHOLD PER CLASS >= {MIN_TRIAGE_SAMPLES}")
    print("=" * 95 + "\n")
    
    age_map = load_patient_age_map(DATASETS_DIR)
    
    # Generate all 15 non-empty combinations of the 4 clinical groups
    group_names = list(FEATURE_GROUPS.keys())
    all_combinations = []
    for r in [1, 2, 3, 4]:
        for combo in itertools.combinations(group_names, r):
            all_combinations.append(combo)
            
    print(f"[COMBINATORICS] Configured {len(all_combinations)} feature combinations:")
    for idx, c in enumerate(all_combinations, 1):
        print(f"  ({idx:2d}) {' + '.join(c)} ({len(c)} group{'s' if len(c)>1 else ''})")

    all_results = []
    all_km_cache = {}
    model_counter = 0

    # Main search across Sliding Windows and Target Formulations
    for W in WINDOWS:
        for tgt_cfg in TARGET_CONFIGS:
            target_str = tgt_cfg["target_str"]
            is_capped = tgt_cfg["is_capped"]
            
            print("\n" + "#" * 95)
            print(f" INGESTING GROUP BASELINE ARRAYS: Window W={W}d | Target={target_str.upper()}")
            print("#" * 95)
            
            # Cache individual group arrays to prevent redundant file re-reads
            group_cache = {}
            for g_name, file_key in FEATURE_GROUPS.items():
                base_name = f"NN_training_dataset_W{W}_{file_key}_{target_str}"
                file_path = find_dataset_file(DATASETS_DIR, base_name)
                if not os.path.exists(file_path):
                    print(f"[MISSING FILE] Cannot find '{file_path}'. Skipping group '{g_name}'.")
                    continue
                df_raw, pid_col = load_dataset(file_path)
                X_raw, y_log, y_raw_days, events_all, age_all, pid_all, time_all = prepare_raw_data(
                    df_raw, pid_col, is_capped=is_capped, age_map=age_map
                )
                group_cache[g_name] = {
                    "X": X_raw,
                    "y_log": y_log,
                    "y_days": y_raw_days,
                    "events": events_all,
                    "age": age_all,
                    "pid": pid_all,
                    "time": time_all
                }

            # Evaluate each feature combination
            for combo in all_combinations:
                combo_str = "+".join(combo)
                combo_size = len(combo)
                
                # Check if all constituent groups exist
                if not all(g in group_cache for g in combo):
                    continue
                
                ref_g = combo[0]
                pid_all = group_cache[ref_g]["pid"]
                time_all = group_cache[ref_g]["time"]
                y_log = group_cache[ref_g]["y_log"]
                y_days = group_cache[ref_g]["y_days"]
                events_all = group_cache[ref_g]["events"]
                age_all = group_cache[ref_g]["age"]
                
                # Concatenate features horizontally
                X_combo = np.concatenate([group_cache[g]["X"] for g in combo], axis=1)
                
                print(f"\n{'-'*90}")
                print(f" EVALUATING COMBO: [{combo_str}] | W={W} | Target={target_str} | Total Raw Features: {X_combo.shape[1]}")
                print(f"{'-'*90}")
                
                for split_mode in SPLIT_MODES:
                    train_mask, val_mask, test_mask = create_split_masks(pid_all, time_all, split_mode=split_mode)
                    
                    X_tr, y_tr_log, y_tr_d, ev_tr, age_tr = extract_features(X_combo, y_log, y_days, events_all, age_all, train_mask)
                    X_va, y_va_log, y_va_d, ev_va, age_va = extract_features(X_combo, y_log, y_days, events_all, age_all, val_mask)
                    X_te, y_te_log, y_te_d, ev_te, age_te = extract_features(X_combo, y_log, y_days, events_all, age_all, test_mask)
                    
                    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                        continue
                        
                    # Standardize strictly on training statistics
                    mean = X_tr.mean(axis=0, keepdims=True)
                    std = X_tr.std(axis=0, keepdims=True)
                    std[std == 0] = 1.0
                    
                    X_tr_s = (X_tr - mean) / std
                    X_va_s = (X_va - mean) / std
                    X_te_s = (X_te - mean) / std
                    
                    train_loader = DataLoader(TabularLogDataset(X_tr_s, y_tr_log, y_tr_d, ev_tr, age_tr), batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(TabularLogDataset(X_va_s, y_va_log, y_va_d, ev_va, age_va), batch_size=BATCH_SIZE, shuffle=False)
                    test_loader = DataLoader(TabularLogDataset(X_te_s, y_te_log, y_te_d, ev_te, age_te), batch_size=BATCH_SIZE, shuffle=False)
                    
                    N = X_tr_s.shape[1]
                    base_steps = [1, 2, 3, 5, 8, 16, 32, 64, 128]
                    candidate_H = sorted(list(set([h for h in base_steps if h < N] + [N])))
                    
                    arch_history = []
                    
                    for h in candidate_H:
                        model_counter += 1
                        model_num = model_counter
                        model_id = f"{combo_str}_W{W}_H{h}_{target_str}_{split_mode}"
                        
                        model = SystematicFFNN(input_dim=N, hidden_dim=h).to(device)
                        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        
                        tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, test_preds, test_true, test_events, test_ages = train_model(
                            model, train_loader, val_loader, test_loader, device, is_capped=is_capped
                        )
                        
                        weights_path = os.path.join(WEIGHTS_DIR, f"Model_{model_num}_{model_id}.pt")
                        torch.save(model.state_dict(), weights_path)
                        
                        triage_metrics = evaluate_triage_and_kaplan_meier(test_true, test_preds, test_events, test_ages)
                        all_km_cache[model_num] = (model_id, triage_metrics)
                        
                        # Symmetric error gaps and divergence ratios
                        diff_mae = abs(te_mae - tr_mae)
                        diff_mape = abs(te_mape - tr_mape)
                        overfitting_ratio = te_mae / (tr_mae + 1e-6)
                        
                        # Minimum cohort clinical filter (>= 1000 in Red, Yellow, Green)
                        is_clinically_valid = (
                            triage_metrics["Sessions_Red"] >= MIN_TRIAGE_SAMPLES and
                            triage_metrics["Sessions_Yellow"] >= MIN_TRIAGE_SAMPLES and
                            triage_metrics["Sessions_Green"] >= MIN_TRIAGE_SAMPLES
                        )
                        
                        # Distance to Utopia point (0, 0)
                        dist_ideal = np.sqrt(triage_metrics["P_bar_G"]**2 + triage_metrics["P_R"]**2)
                        
                        entry = {
                            "Model_Num": model_num,
                            "Model_ID": model_id,
                            "Feature_Combo": combo_str,
                            "Combo_Size": combo_size,
                            "Window_W": W,
                            "Target_Type": target_str,
                            "Split_Mode": split_mode,
                            "N_inputs": N,
                            "Hidden_H": h,
                            "Parameters": num_params,
                            # Symmetrically matched Train and Test metrics
                            "Train_MAE_days": round(tr_mae, 2),
                            "Test_MAE_days": round(te_mae, 2),
                            "Diff_MAE_days": round(diff_mae, 2),
                            "Train_MAPE": round(tr_mape, 2),
                            "Test_MAPE": round(te_mape, 2),
                            "Diff_MAPE": round(diff_mape, 2),
                            "Val_MAE_days": round(va_mae, 2),
                            "Val_MAPE": round(va_mape, 2),
                            "Overfitting_Ratio": round(overfitting_ratio, 3),
                            # Upstream cohort size gating
                            "Clinically_Valid": is_clinically_valid,
                            "Sessions_Red": triage_metrics["Sessions_Red"],
                            "Sessions_Yellow": triage_metrics["Sessions_Yellow"],
                            "Sessions_Green": triage_metrics["Sessions_Green"],
                            "Total_Sessions": triage_metrics["Total_Sessions"],
                            # Survival probabilities
                            "P_R": triage_metrics["P_R"],
                            "P_G": triage_metrics["P_G"],
                            "P_bar_G": triage_metrics["P_bar_G"],
                            "Dist_Ideal": round(float(dist_ideal), 4),
                            # Demographic stats
                            "Age_Red": triage_metrics["Age_Red"],
                            "Age_Yellow": triage_metrics["Age_Yellow"],
                            "Age_Green": triage_metrics["Age_Green"],
                            "Age_Total": triage_metrics["Age_Total"]
                        }
                        all_results.append(entry)
                        arch_history.append({"H": h, "train_mae": tr_mae, "val_mae": va_mae, "test_mae": te_mae})
                        
                        status_str = "VALID" if is_clinically_valid else f"FILTERED (Class < {MIN_TRIAGE_SAMPLES})"
                        print(f"  [#{model_num:3d}: H={h:3d}] | Tr/Te MAE: {tr_mae:4.1f}/{te_mae:4.1f}d (Diff: {diff_mae:4.1f}d) | "
                              f"Tr/Te MAPE: {tr_mape:4.1f}%/{te_mape:4.1f}% | P_R: {triage_metrics['P_R']:.2f}, P_G: {triage_metrics['P_G']:.2f} | "
                              f"Sessions: {triage_metrics['Sessions_Red']}/{triage_metrics['Sessions_Yellow']}/{triage_metrics['Sessions_Green']} [{status_str}]")
                        
                        should_prune, reason = check_architectural_pruning(arch_history)
                        if should_prune:
                            print(f"       ==> Architectural expansion halted for '{combo_str}'. {reason}")
                            break

    # =========================================================================
    # 11. Upstream Gated Pareto Frontier & Winner Selection
    # =========================================================================
    df_all = pd.DataFrame(all_results)
    if df_all.empty:
        print("[ERROR] No models were trained. Please verify dataset file paths.")
        return

    df_all["is_pareto"] = False
    
    # Upstream cohort gating: strictly retain models satisfying class size thresholds
    valid_mask = df_all["Clinically_Valid"]
    valid_indices = df_all[valid_mask].index
    print(f"\n[CLINICAL GATE] Filtered {len(df_all) - len(valid_indices)} out of {len(df_all)} models due to cohort threshold (< {MIN_TRIAGE_SAMPLES} sessions).")
    
    if len(valid_indices) > 0:
        valid_p_bar_g = df_all.loc[valid_indices, "P_bar_G"].values
        valid_p_r = df_all.loc[valid_indices, "P_R"].values
        pareto_sub_mask = compute_pareto_mask(valid_p_bar_g, valid_p_r)
        actual_pareto_indices = valid_indices[pareto_sub_mask]
        df_all.loc[actual_pareto_indices, "is_pareto"] = True
        
    # Winner Selection per Feature Combination
    winners = []
    for combo_str, group in df_all.groupby("Feature_Combo"):
        valid_group = group[group["Clinically_Valid"]].copy()
        if valid_group.empty:
            continue
            
        # Identify local Pareto frontier for this combination
        local_p_bar = valid_group["P_bar_G"].values
        local_pr = valid_group["P_R"].values
        local_pareto_mask = compute_pareto_mask(local_p_bar, local_pr)
        local_pareto = valid_group[local_pareto_mask].copy()
        
        # Choose winner minimizing distance to ideal (0,0) and secondarily Test MAE
        local_pareto = local_pareto.sort_values(by=["Dist_Ideal", "Test_MAE_days"])
        winner_row = local_pareto.iloc[0].to_dict()
        winners.append(winner_row)
        
    winners_df = pd.DataFrame(winners).sort_values(by="Dist_Ideal").reset_index(drop=True)

    # Save full CSV with symmetrical train/test metrics
    out_csv = os.path.join(BASE_DIR, "final_combinatorial_ffnn_pareto_results.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\n[EXPORT] Master execution dataset saved to: {out_csv}")
    
    # Save winner comparison CSV
    winners_csv = os.path.join(BASE_DIR, "pareto_winners_comparison.csv")
    winners_df.to_csv(winners_csv, index=False)
    print(f"[EXPORT] Pareto winners per combination saved to: {winners_csv}")

    # Generate visual outputs
    print(f"\n[EXPORT] Rendering Kaplan-Meier curves for evaluated models into: {PLOTS_DIR}")
    for model_num, (mid, km_data) in all_km_cache.items():
        row = df_all[df_all["Model_Num"] == model_num].iloc[0]
        status_tag = "[PARETO]" if row["is_pareto"] else ("[VALID]" if row["Clinically_Valid"] else "[PRUNED]")
        km_title = f"{status_tag} Model #{model_num} - {mid}"
        km_filename = os.path.join(PLOTS_DIR, f"KM_Model_{model_num}_{mid}.png")
        plot_kaplan_meier_curves(km_data, km_title, km_filename, min_x_extent=KM_MIN_DISPLAY_DAYS)

    global_pareto_file = os.path.join(PLOTS_DIR, "global_pareto_frontier_numbered.png")
    plot_global_pareto_numbered(df_all, "Global Pareto Frontier (Models Meeting Cohort Constraints)", global_pareto_file)
    print(f"[EXPORT] Saved Global Pareto Frontier Plot: {global_pareto_file}")

    winners_plot_file = os.path.join(PLOTS_DIR, "pareto_winners_comparison.png")
    plot_pareto_winners_comparison(winners_df, "Winning Pareto Models Across Feature Combinations", winners_plot_file)
    print(f"[EXPORT] Saved Winner Comparison Plot: {winners_plot_file}")

    # Display Winner Comparison Table in Console
    disp_cols = ["Feature_Combo", "Combo_Size", "Model_Num", "Window_W", "Target_Type", 
                 "Train_MAE_days", "Test_MAE_days", "Diff_MAE_days", "Train_MAPE", "Test_MAPE",
                 "P_R", "P_G", "Dist_Ideal", "Sessions_Red", "Sessions_Green"]
    print("\n" + "=" * 135)
    print(" SUMMARY OF WINNING MODELS ACROSS ALL 15 FEATURE COMBINATIONS (SORTED BY DISTANCE TO UTOPIA)")
    print("=" * 135)
    print(winners_df[disp_cols].to_string(index=False))
    print("=" * 135)


if __name__ == "__main__":
    main()