import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 1. Global Configurations and File Paths ---
BASE_DIR = r"c:\Users\vince\Desktop\NN"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets") if os.path.exists(os.path.join(BASE_DIR, "datasets")) else "datasets"
PLOTS_DIR = os.path.join(BASE_DIR, "triage_pareto_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# The 6 dataset files ordered with priority to W=60
DATASET_FILES = [
    {"name": "44misure_capped360_W60", "file": "NN_training_dataset_W60_44_capped360", "W": 60, "capped": True},
    {"name": "26misure_capped360_W60", "file": "NN_training_dataset_W60_26_capped360", "W": 60, "capped": True},
    {"name": "26misure_uncapped_W60",  "file": "NN_training_dataset_W60_26_uncapped",  "W": 60, "capped": False},
    {"name": "44misure_capped360_W30", "file": "NN_training_dataset_W30_44_capped360", "W": 30, "capped": True},
    {"name": "26misure_capped360_W30", "file": "NN_training_dataset_W30_26_capped360", "W": 30, "capped": True},
    {"name": "26misure_uncapped_W30",  "file": "NN_training_dataset_W30_26_uncapped",  "W": 30, "capped": False},
]

SPLIT_MODES = ["patient_wise", "temporal"]
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 2e-2
EARLY_STOPPING_PATIENCE = 40
TTE_CAP_DAYS = 360.0

# Clinical Triage & KM Evaluation Cut-offs
T_RED = 180.0
T_GREEN = 365.0
KM_EVAL_RED = 180.0
KM_EVAL_GREEN = 360.0


# --- 2. Scalable Feed-Forward Architecture (N -> H -> 1) ---
class SystematicFFNN(nn.Module):
    """
    Feed-Forward Neural Network:
    - Input dimension: N = 4 * M (slope a, intercept b, RMSE c, current value z)
    - Hidden dimension: H (dynamically explored from 1, 2, 3 up to N)
    - Output dimension: 1 (predicts log(TTE))
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.20, noise_std=0.03):
        super().__init__()
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim
        
        # When hidden_dim == 1, LayerNorm across dimension 1 would zero out outputs.
        # LayerNorm is therefore applied only when hidden_dim > 1.
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


# --- 3. Dataset Loading & Feature Extraction (N = 4 * M) ---
def find_dataset_file(datasets_dir, base_name):
    candidates = [
        os.path.join(datasets_dir, base_name),
        os.path.join(datasets_dir, base_name + ".csv"),
        os.path.join(datasets_dir, base_name + ".CSV"),
        os.path.join(BASE_DIR, base_name),
        os.path.join(BASE_DIR, base_name + ".csv"),
        os.path.join(".", base_name),
        os.path.join(".", base_name + ".csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(datasets_dir, base_name + ".csv")


def load_dataset(file_path):
    print(f"\n[DATA LOADER] Reading file: {file_path}")
    t0 = time.time()
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns in {time.time() - t0:.2f}s.")
    
    # Dynamically resolve patient ID column spelling
    pid_col = 'partient_id' if 'partient_id' in df.columns else 'patient_id'
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if pid_col in df.columns:
            df = df.sort_values(by=[pid_col, 'timestamp']).reset_index(drop=True)
            
    return df, pid_col


def prepare_raw_data(df, pid_col):
    pid_all = df[pid_col].values
    time_all = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
    y_raw_days = df['tte'].values.astype(np.float32)
    
    # Target: use log_tte directly from column if present, otherwise compute log(1 + TTE)
    if 'log_tte' in df.columns:
        y_log = df['log_tte'].values.astype(np.float32)
    else:
        y_log = np.log(np.maximum(y_raw_days, 1.0))
        
    all_str = ",".join(df['misure'].str.strip('{}'))
    parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
    X_raw = parsed.reshape(len(df), -1)
    
    return X_raw, y_log, y_raw_days, pid_all, time_all


def create_split_masks(pid_all, time_all, split_mode="patient_wise", train_ratio=0.5, val_ratio=0.1):
    n_samples = len(pid_all)
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    unique_pids = np.unique(pid_all)
    
    if split_mode == "patient_wise":
        # Disjoint patient split
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
                
    elif split_mode == "temporal":
        # Chronological split per patient
        for pid in unique_pids:
            p_indices = np.where(pid_all == pid)[0]
            n_p = len(p_indices)
            n_tr = int(n_p * train_ratio)
            n_va = int(n_p * val_ratio)
            
            train_mask[p_indices[:n_tr]] = True
            val_mask[p_indices[n_tr : n_tr + n_va]] = True
            test_mask[p_indices[n_tr + n_va :]] = True
            
    return train_mask, val_mask, test_mask


def extract_features(X_all, y_log, y_days, pid_all, target_mask, W):
    """
    Extracts features for the specified partition:
    - If already pre-extracted (104 or 176 features), selects valid rows directly.
    - If raw measures (26 or 44), computes the linear regression trend over window W.
      Features extracted per measure: a (slope), b (intercept), c (RMSE), z (current value).
    """
    M_raw = X_all.shape[1]
    if M_raw in [104, 176]:
        valid_indices = np.where(target_mask)[0]
        return X_all[valid_indices], y_log[valid_indices].reshape(-1, 1), y_days[valid_indices].reshape(-1, 1)
        
    N_total = len(pid_all)
    pids_arr = np.array(pid_all)
    same_patient = (pids_arr[W - 1:] == pids_arr[: N_total - W + 1])
    target_in_set = target_mask[W - 1:]
    valid_indices = np.where(same_patient & target_in_set)[0] + (W - 1)
    
    if len(valid_indices) == 0:
        return np.empty((0, 4 * M_raw), dtype=np.float32), np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

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
    
    X_features = np.concatenate([a, b, c_rmse, z_curr], axis=1)
    return X_features, y_log[valid_indices].reshape(-1, 1), y_days[valid_indices].reshape(-1, 1)


# --- 4. PyTorch Tabular Dataset ---
class TabularLogDataset(Dataset):
    def __init__(self, X_tab, y_log, y_days):
        self.X = torch.tensor(X_tab, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32)
        self.y_days = torch.tensor(y_days, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_log[idx], self.y_days[idx]


# --- 5. Model Evaluation and Training Loop with Early Stopping ---
def evaluate_model(model, data_loader, device, is_capped):
    model.eval()
    all_preds_days = []
    all_true_days = []
    
    with torch.no_grad():
        for X_b, _, y_days_b in data_loader:
            X_b = X_b.to(device)
            preds_log = model(X_b)
            # Inverse log transform: exp(pred_log) guarantees strictly positive TTE
            preds_days = torch.exp(preds_log)
            if is_capped:
                preds_days = torch.clamp(preds_days, max=TTE_CAP_DAYS)
                
            all_preds_days.append(preds_days.cpu().numpy().flatten())
            all_true_days.append(y_days_b.numpy().flatten())
            
    all_preds = np.concatenate(all_preds_days)
    all_true = np.concatenate(all_true_days)
    
    mae = np.mean(np.abs(all_preds - all_true))
    denom = np.clip(all_true, 10.0, None)
    mape = np.mean(np.abs(all_preds - all_true) / denom) * 100.0
    
    return mae, mape, all_preds, all_true


def train_model(model, train_loader, val_loader, test_loader, device, is_capped, epochs=EPOCHS):
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_mae = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_log_b, _ in train_loader:
            X_b, y_log_b = X_b.to(device), y_log_b.to(device)
            optimizer.zero_grad()
            out_log = model(X_b)
            loss = criterion(out_log, y_log_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        val_mae, _, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
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
        
    tr_mae, tr_mape, _, _ = evaluate_model(model, train_loader, device, is_capped=is_capped)
    va_mae, va_mape, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
    te_mae, te_mape, test_preds, test_true = evaluate_model(model, test_loader, device, is_capped=is_capped)
    
    return tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, test_preds, test_true


# --- 6. Architectural Pruning Controller (Overfitting Check) ---
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
            return True, f"Validation stagnation: no improvement over {best_val:.2f}d."
            
    return False, ""


# --- 7. Kaplan-Meier and Clinical Triage Engine (Computed on Test Set) ---
def compute_kaplan_meier_curve(durations, events=None):
    durations = np.asarray(durations, dtype=np.float64)
    if events is None:
        events = np.ones_like(durations, dtype=np.int32)
    else:
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


def get_survival_probability_at_time(km_times, km_probs, target_time):
    idx = np.searchsorted(km_times, target_time, side='right') - 1
    idx = np.clip(idx, 0, len(km_probs) - 1)
    return float(km_probs[idx])


def evaluate_triage_and_kaplan_meier(y_test_true, y_test_pred):
    y_test_true = np.asarray(y_test_true, dtype=np.float32)
    y_test_pred = np.asarray(y_test_pred, dtype=np.float32)
    
    # Clinical Triage Classification based on predicted TTE
    mask_red = y_test_pred < T_RED
    mask_yellow = (y_test_pred >= T_RED) & (y_test_pred <= T_GREEN)
    mask_green = y_test_pred > T_GREEN
    
    # 1. KM Curve for Red Class (High Risk) -> Objective: Minimize P_R at 180d
    if np.sum(mask_red) > 0:
        t_r, s_r = compute_kaplan_meier_curve(y_test_true[mask_red])
        p_r = get_survival_probability_at_time(t_r, s_r, KM_EVAL_RED)
    else:
        t_r, s_r = np.array([0.0, KM_EVAL_RED]), np.array([1.0, 1.0])
        p_r = 1.0
        
    # 2. KM Curve for Green Class (Low Risk) -> Objective: Maximize P_G at 360d
    if np.sum(mask_green) > 0:
        t_g, s_g = compute_kaplan_meier_curve(y_test_true[mask_green])
        p_g = get_survival_probability_at_time(t_g, s_g, KM_EVAL_GREEN)
    else:
        t_g, s_g = np.array([0.0, KM_EVAL_GREEN]), np.array([0.0, 0.0])
        p_g = 0.0
        
    # 3. KM Curve for Yellow Class (Medium Risk)
    if np.sum(mask_yellow) > 0:
        t_y, s_y = compute_kaplan_meier_curve(y_test_true[mask_yellow])
    else:
        t_y, s_y = np.array([0.0]), np.array([1.0])
        
    p_bar_g = 1.0 - p_g  # Objective: Minimize alongside P_R
    
    return {
        "P_R": round(p_r, 4),
        "P_G": round(p_g, 4),
        "P_bar_G": round(p_bar_g, 4),
        "N_Red": int(np.sum(mask_red)),
        "N_Yellow": int(np.sum(mask_yellow)),
        "N_Green": int(np.sum(mask_green)),
        "km_curves": {
            "Red": (t_r, s_r),
            "Yellow": (t_y, s_y),
            "Green": (t_g, s_g)
        }
    }


# --- 8. Pareto Frontier Identification ---
def compute_pareto_mask(p_bar_g_arr, p_r_arr):
    pts = np.column_stack([p_bar_g_arr, p_r_arr])
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                # Point j dominates point i if <= on both objectives and < on at least one
                if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1]) and \
                   (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1]):
                    is_pareto[i] = False
                    break
    return is_pareto


# --- 9. Automated Plotting Functions (KM Curves & Pareto Frontier) ---
def plot_kaplan_meier_curves(km_data, title, save_path):
    plt.figure(figsize=(9, 5.5))
    
    t_g, s_g = km_data["km_curves"]["Green"]
    t_y, s_y = km_data["km_curves"]["Yellow"]
    t_r, s_r = km_data["km_curves"]["Red"]
    
    p_r = km_data["P_R"]
    p_g = km_data["P_G"]
    
    plt.step(t_g, s_g, label=f"Green (Pred > 365d, N={km_data['N_Green']})", color="#2ca02c", linewidth=2.5, where='post')
    plt.step(t_y, s_y, label=f"Yellow (180-365d, N={km_data['N_Yellow']})", color="#ff7f0e", linewidth=2.0, where='post')
    plt.step(t_r, s_r, label=f"Red (Pred < 180d, N={km_data['N_Red']})", color="#d62728", linewidth=2.5, where='post')
    
    # Reference vertical lines at 180 and 360 days with arrows
    plt.axvline(x=KM_EVAL_RED, color="gray", linestyle="--", alpha=0.7)
    plt.scatter([KM_EVAL_RED], [p_r], color="#d62728", s=80, zorder=5)
    plt.annotate(f"$P_R = {p_r:.2f}$ $\\downarrow$", xy=(KM_EVAL_RED, p_r), xytext=(KM_EVAL_RED, p_r - 0.12),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8),
                 ha="center", fontsize=11, fontweight="bold", color="#d62728")
                 
    plt.axvline(x=KM_EVAL_GREEN, color="gray", linestyle="--", alpha=0.7)
    plt.scatter([KM_EVAL_GREEN], [p_g], color="#2ca02c", s=80, zorder=5)
    plt.annotate(f"$P_G = {p_g:.2f}$ $\\uparrow$", xy=(KM_EVAL_GREEN, p_g), xytext=(KM_EVAL_GREEN, min(1.0, p_g + 0.10)),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.8),
                 ha="center", fontsize=11, fontweight="bold", color="#2ca02c")
                 
    plt.xlabel("Actual Survival Time (Days)", fontsize=11)
    plt.ylabel("Survival Probability S(t)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pareto_frontier(df_pareto, title, save_path):
    if df_pareto.empty:
        return
        
    pts = df_pareto[["P_bar_G", "P_R"]].values
    is_pareto = compute_pareto_mask(pts[:, 0], pts[:, 1])
    df_pareto["is_pareto"] = is_pareto
    
    pareto_pts = df_pareto[df_pareto["is_pareto"]].sort_values(by="P_bar_G")
    dominated_pts = df_pareto[~df_pareto["is_pareto"]]
    
    plt.figure(figsize=(8, 6.5))
    
    # Dominated models
    if not dominated_pts.empty:
        plt.scatter(dominated_pts["P_bar_G"], dominated_pts["P_R"], color="#7f7f7f", alpha=0.6, s=60, label="Dominated Models")
        for _, r in dominated_pts.iterrows():
            plt.annotate(f"H={int(r['Hidden_H'])}", (r["P_bar_G"], r["P_R"]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#555555")
            
    # Non-dominated models (Pareto Frontier)
    plt.scatter(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", s=110, zorder=5, label="Pareto Frontier (Non-Dominated)")
    plt.plot(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", linestyle="-", linewidth=2.0, zorder=4)
    for _, r in pareto_pts.iterrows():
        plt.annotate(f"H={int(r['Hidden_H'])}\n({r['Parameters']}p)", (r["P_bar_G"], r["P_R"]), textcoords="offset points", xytext=(6, -6), fontsize=9, fontweight="bold", color="#d62728")
        
    # Ideal utopia point
    plt.scatter([0], [0], color="#1f77b4", marker="*", s=160, zorder=6, label="Ideal Point (0, 0)")
    
    plt.xlabel(r"$\bar{P}_G = 1 - P_G$ (Minimize $\to 0$)", fontsize=11)
    plt.ylabel(r"$P_R$ (Minimize $\to 0$)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlim(-0.02, max(1.0, df_pareto["P_bar_G"].max() * 1.1))
    plt.ylim(-0.02, max(1.0, df_pareto["P_R"].max() * 1.1))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- 10. Main Execution Pipeline ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=========================================================================")
    print(f" COMPLETE PIPELINE: FFNN GRID SEARCH + TRIAGE + KM + PARETO FRONTIER")
    print(f" COMPUTATION DEVICE: {device}")
    print(f"=========================================================================\n")
    
    all_results = []
    best_models_km = {}
    
    for ds_entry in DATASET_FILES:
        ds_name = ds_entry["name"]
        ds_file = ds_entry["file"]
        W = ds_entry["W"]
        is_capped = ds_entry["capped"]
        
        file_path = find_dataset_file(DATASETS_DIR, ds_file)
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}. Skipping dataset.")
            continue
            
        print(f"\n{'='*90}\n PROCESSING DATASET: {ds_name} (Window W={W}, Capped={is_capped})\n{'='*90}")
        df, pid_col = load_dataset(file_path)
        X_raw, y_log, y_days, pid_all, time_all = prepare_raw_data(df, pid_col)
        
        for split_mode in SPLIT_MODES:
            print(f"\n>>> Split Mode: {split_mode.upper()} <<<")
            train_mask, val_mask, test_mask = create_split_masks(pid_all, time_all, split_mode=split_mode)
            
            X_tr, y_tr_log, y_tr_d = extract_features(X_raw, y_log, y_days, pid_all, train_mask, W)
            X_va, y_va_log, y_va_d = extract_features(X_raw, y_log, y_days, pid_all, val_mask, W)
            X_te, y_te_log, y_te_d = extract_features(X_raw, y_log, y_days, pid_all, test_mask, W)
            
            if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                print(f"[SKIP] Insufficient samples for split {split_mode}.")
                continue
                
            mean = X_tr.mean(axis=0, keepdims=True)
            std = X_tr.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            
            X_tr_s = (X_tr - mean) / std
            X_va_s = (X_va - mean) / std
            X_te_s = (X_te - mean) / std
            
            train_loader = DataLoader(TabularLogDataset(X_tr_s, y_tr_log, y_tr_d), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TabularLogDataset(X_va_s, y_va_log, y_va_d), batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(TabularLogDataset(X_te_s, y_te_log, y_te_d), batch_size=BATCH_SIZE, shuffle=False)
            
            N = X_tr_s.shape[1]
            base_steps = [1, 2, 3, 5, 8, 16, 32, 64, 128]
            candidate_H = sorted(list(set([h for h in base_steps if h < N] + [N])))
            
            print(f"Input Features N = {N} (4 x {N//4} measures). Candidate H Grid: {candidate_H}")
            
            arch_history = []
            dataset_split_results = []
            
            for h in candidate_H:
                model = SystematicFFNN(input_dim=N, hidden_dim=h).to(device)
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, test_preds, test_true = train_model(
                    model, train_loader, val_loader, test_loader, device, is_capped=is_capped
                )
                
                # Clinical Triage and Kaplan-Meier evaluation on the real Test Set
                triage_metrics = evaluate_triage_and_kaplan_meier(test_true, test_preds)
                
                diff_tr_te = abs(tr_mae - te_mae)
                
                entry = {
                    "Dataset": ds_name,
                    "Measures": N // 4,
                    "Capped": is_capped,
                    "Split": split_mode,
                    "Window_W": W,
                    "N_inputs": N,
                    "Hidden_H": h,
                    "Parameters": num_params,
                    "Train_MAE_days": round(tr_mae, 2),
                    "Val_MAE_days": round(va_mae, 2),
                    "Test_MAE_days": round(te_mae, 2),
                    "Diff_Train_Test": round(diff_tr_te, 2),
                    "Test_MAPE": round(te_mape, 2),
                    "P_R": triage_metrics["P_R"],
                    "P_G": triage_metrics["P_G"],
                    "P_bar_G": triage_metrics["P_bar_G"],
                    "N_Red": triage_metrics["N_Red"],
                    "N_Yellow": triage_metrics["N_Yellow"],
                    "N_Green": triage_metrics["N_Green"]
                }
                all_results.append(entry)
                dataset_split_results.append(entry)
                arch_history.append({"H": h, "train_mae": tr_mae, "val_mae": va_mae, "test_mae": te_mae})
                
                print(f"  [H = {h:3d}] ({num_params:5d} params) -> Train MAE: {tr_mae:5.2f}d | Val MAE: {va_mae:5.2f}d | Test MAE: {te_mae:5.2f}d | "
                      f"P_R (180d): {triage_metrics['P_R']:.2f} | P_G (360d): {triage_metrics['P_G']:.2f} | P_bar_G: {triage_metrics['P_bar_G']:.2f}")
                
                # Cache KM data for the best configuration (lowest Test MAE) in this scenario
                key_model = f"{ds_name}_{split_mode}"
                if key_model not in best_models_km or te_mae < best_models_km[key_model]["Test_MAE"]:
                    best_models_km[key_model] = {
                        "H": h,
                        "Test_MAE": te_mae,
                        "km_data": triage_metrics,
                        "title": f"Kaplan-Meier Triage Curves - {ds_name} ({split_mode}, H={h})"
                    }
                    
                should_prune, reason = check_architectural_pruning(arch_history)
                if should_prune:
                    print(f"     ==> [PRUNING] Stopped H expansion for {ds_name} ({split_mode}). Reason: {reason}")
                    break
                    
            # Generate Pareto Frontier for the individual scenario (Dataset + Split)
            df_scenario = pd.DataFrame(dataset_split_results)
            pareto_filename = os.path.join(PLOTS_DIR, f"pareto_{ds_name}_{split_mode}.png")
            plot_pareto_frontier(df_scenario, f"Pareto Frontier: {ds_name} ({split_mode})", pareto_filename)

    # --- 11. Generate KM Plots for Best Models ---
    print(f"\n{'='*90}\n GENERATING KAPLAN-MEIER PLOTS AND GLOBAL PARETO FRONTIER\n{'='*90}")
    for key_model, info in best_models_km.items():
        km_filename = os.path.join(PLOTS_DIR, f"km_curve_{key_model}_H{info['H']}.png")
        plot_kaplan_meier_curves(info["km_data"], info["title"], km_filename)
        print(f"Saved KM plot: {km_filename}")

    # --- 12. Global Pareto Frontier & Final CSV Export ---
    df_all = pd.DataFrame(all_results)
    if not df_all.empty:
        df_all["is_pareto"] = compute_pareto_mask(df_all["P_bar_G"].values, df_all["P_R"].values)
        
        global_pareto_file = os.path.join(PLOTS_DIR, "pareto_frontier_global.png")
        plot_pareto_frontier(df_all, "Global Pareto Frontier (All Models)", global_pareto_file)
        print(f"Saved Global Pareto Frontier plot: {global_pareto_file}")
        
        out_csv = os.path.join(BASE_DIR, "final_systematic_ffnn_triage_pareto_results.csv")
        df_all.to_csv(out_csv, index=False)
        print(f"Full results exported to: {out_csv}")
        
        # Display optimal non-dominated models
        pareto_models = df_all[df_all["is_pareto"]].sort_values(by="P_bar_G")
        print("\n" + "="*110)
        print(" OPTIMAL NON-DOMINATED MODELS ON THE PARETO FRONTIER (P_R vs P_bar_G)")
        print("="*110)
        print(pareto_models[["Dataset", "Split", "Window_W", "Hidden_H", "Parameters", "Test_MAE_days", "P_R", "P_G", "P_bar_G"]].to_string(index=False))
        print("="*110)

if __name__ == "__main__":
    main()