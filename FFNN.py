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
WEIGHTS_DIR = os.path.join(BASE_DIR, "models_weights")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 6 dataset configurations with biological feature count M and window size W
DATASET_FILES = [
    {"name": "44misure_capped360_W60", "file": "NN_training_dataset_W60_44_capped360", "W": 60, "M": 44, "capped": True},
    {"name": "26misure_capped360_W60", "file": "NN_training_dataset_W60_26_capped360", "W": 60, "M": 26, "capped": True},
    {"name": "26misure_uncapped_W60",  "file": "NN_training_dataset_W60_26_uncapped",  "W": 60, "M": 26, "capped": False},
    {"name": "44misure_capped360_W30", "file": "NN_training_dataset_W30_44_capped360", "W": 30, "M": 44, "capped": True},
    {"name": "26misure_capped360_W30", "file": "NN_training_dataset_W30_26_capped360", "W": 30, "M": 26, "capped": True},
    {"name": "26misure_uncapped_W30",  "file": "NN_training_dataset_W30_26_uncapped",  "W": 30, "M": 26, "capped": False},
]

# Standardized split naming
SPLIT_MODES = ["split_patient", "split_temporal"]
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 2e-2
EARLY_STOPPING_PATIENCE = 40

# Training cap updated to 400 days to prevent prediction boundary compression
TTE_TRAIN_CAP_DAYS = 400.0

# Clinical Triage cut-offs
T_RED = 180.0
T_GREEN = 360.0
KM_EVAL_RED = 180.0
KM_EVAL_GREEN = 360.0
KM_MAX_DISPLAY_TIME = 500.0  # Extended KM x-axis display range (at least 500 days)


# --- 2. Scalable Feed-Forward Neural Network (N -> H -> 1) ---
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


# --- 3. Data Loading & Feature Extraction with Censoring Vector ---
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
    print(f"Loaded {len(df)} rows in {time.time() - t0:.2f}s.")
    
    pid_col = 'partient_id' if 'partient_id' in df.columns else 'patient_id'
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if pid_col in df.columns:
            df = df.sort_values(by=[pid_col, 'timestamp']).reset_index(drop=True)
            
    return df, pid_col


def prepare_raw_data(df, pid_col, is_capped):
    pid_all = df[pid_col].values
    time_all = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
    y_raw_days = df['tte'].values.astype(np.float32)
    
    # 1. Training target with updated cap of 400 days (raw y_raw_days remain unchanged)
    if is_capped:
        y_train_days = np.clip(y_raw_days, 1.0, TTE_TRAIN_CAP_DAYS)
    else:
        y_train_days = np.maximum(y_raw_days, 1.0)
    y_log = np.log(y_train_days)
    
    # 2. Binary event indicator vector (E=1 event observed, E=0 right-censored)
    event_col = None
    for cand in ['event', 'status', 'evento', 'failure', 'observed', 'censor', 'censored']:
        if cand in df.columns:
            event_col = cand
            break
            
    if event_col is not None:
        events_all = df[event_col].values.astype(np.int32)
        if 'censor' in event_col.lower():
            events_all = 1 - events_all
        print(f"[DATA LOADER] Detected event column '{event_col}': {np.sum(events_all==1)} events, {np.sum(events_all==0)} censored.")
    else:
        if is_capped:
            # Patients reaching 360 days without an event are administratively right-censored
            events_all = (y_raw_days < 360.0).astype(np.int32)
            print(f"[DATA LOADER] Applied administrative censoring at 360d: {np.sum(events_all==1)} events, {np.sum(events_all==0)} censored.")
        else:
            events_all = np.ones_like(y_raw_days, dtype=np.int32)
            print("[DATA LOADER] Uncapped dataset without explicit event column. Defaulted events to 1.")

    # Parse feature array N = 4 * M
    all_str = ",".join(df['misure'].str.strip('{}'))
    parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
    X_raw = parsed.reshape(len(df), -1)
    
    return X_raw, y_log, y_raw_days, events_all, pid_all, time_all


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


def extract_features(X_all, y_log, y_days, events_all, pid_all, target_mask, W):
    M_raw = X_all.shape[1]
    if M_raw in [104, 176]:
        valid_indices = np.where(target_mask)[0]
        return (X_all[valid_indices], 
                y_log[valid_indices].reshape(-1, 1), 
                y_days[valid_indices].reshape(-1, 1),
                events_all[valid_indices].reshape(-1, 1))
        
    N_total = len(pid_all)
    pids_arr = np.array(pid_all)
    same_patient = (pids_arr[W - 1:] == pids_arr[: N_total - W + 1])
    target_in_set = target_mask[W - 1:]
    valid_indices = np.where(same_patient & target_in_set)[0] + (W - 1)
    
    if len(valid_indices) == 0:
        return (np.empty((0, 4 * M_raw), dtype=np.float32), 
                np.empty((0, 1), dtype=np.float32), 
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.int32))

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
    return (X_features, 
            y_log[valid_indices].reshape(-1, 1), 
            y_days[valid_indices].reshape(-1, 1),
            events_all[valid_indices].reshape(-1, 1))


# --- 4. PyTorch Tabular Dataset with Censoring Vector ---
class TabularLogDataset(Dataset):
    def __init__(self, X_tab, y_log, y_days, events):
        self.X = torch.tensor(X_tab, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32)
        self.y_days = torch.tensor(y_days, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.int32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_log[idx], self.y_days[idx], self.events[idx]


# --- 5. Evaluation and Training Loop with 400-Day Cap ---
def evaluate_model(model, data_loader, device, is_capped):
    model.eval()
    all_preds_days = []
    all_true_days = []
    all_events = []
    
    with torch.no_grad():
        for X_b, _, y_days_b, events_b in data_loader:
            X_b = X_b.to(device)
            preds_log = model(X_b)
            preds_days = torch.exp(preds_log)
            if is_capped:
                preds_days = torch.clamp(preds_days, max=TTE_TRAIN_CAP_DAYS)
                
            all_preds_days.append(preds_days.cpu().numpy().flatten())
            all_true_days.append(y_days_b.numpy().flatten())
            all_events.append(events_b.numpy().flatten())
            
    all_preds = np.concatenate(all_preds_days)
    all_true = np.concatenate(all_true_days)
    all_events = np.concatenate(all_events)
    
    mae = np.mean(np.abs(all_preds - all_true))
    denom = np.clip(all_true, 10.0, None)
    mape = np.mean(np.abs(all_preds - all_true) / denom) * 100.0
    
    return mae, mape, all_preds, all_true, all_events


def train_model(model, train_loader, val_loader, test_loader, device, is_capped, epochs=EPOCHS):
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_mae = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_log_b, _, _ in train_loader:
            X_b, y_log_b = X_b.to(device), y_log_b.to(device)
            optimizer.zero_grad()
            out_log = model(X_b)
            loss = criterion(out_log, y_log_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        val_mae, _, _, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
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
        
    tr_mae, tr_mape, _, _, _ = evaluate_model(model, train_loader, device, is_capped=is_capped)
    va_mae, va_mape, _, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
    te_mae, te_mape, test_preds, test_true, test_events = evaluate_model(model, test_loader, device, is_capped=is_capped)
    
    return tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, test_preds, test_true, test_events


# --- 6. Architectural Pruning Controller ---
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


# --- 7. Rigorous Kaplan-Meier Estimator with Binary Event Vector ---
def compute_kaplan_meier_curve(durations, events):
    """
    Computes the non-parametric Kaplan-Meier survival curve S(t) = Prod (1 - d_i / n_i)
    explicitly accounting for event occurrences (E=1) and right-censored cases (E=0).
    """
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


def get_survival_probability_at_time(km_times, km_probs, target_time):
    idx = np.searchsorted(km_times, target_time, side='right') - 1
    idx = np.clip(idx, 0, len(km_probs) - 1)
    return float(km_probs[idx])


def evaluate_triage_and_kaplan_meier(y_test_true, y_test_pred, y_test_events):
    y_test_true = np.asarray(y_test_true, dtype=np.float32)
    y_test_pred = np.asarray(y_test_pred, dtype=np.float32)
    y_test_events = np.asarray(y_test_events, dtype=np.int32)
    
    # Clinical Triage: Green defined as TTE >= 360d for all datasets
    mask_red = y_test_pred < T_RED
    mask_yellow = (y_test_pred >= T_RED) & (y_test_pred < T_GREEN)
    mask_green = y_test_pred >= T_GREEN
    
    # 1. KM Curve for Red Class (High Risk) -> Minimize P_R at 180d
    if np.sum(mask_red) > 0:
        t_r, s_r = compute_kaplan_meier_curve(y_test_true[mask_red], y_test_events[mask_red])
        p_r = get_survival_probability_at_time(t_r, s_r, KM_EVAL_RED)
    else:
        t_r, s_r = np.array([0.0, KM_EVAL_RED]), np.array([1.0, 1.0])
        p_r = 1.0
        
    # 2. KM Curve for Green Class (Low Risk) -> Maximize P_G at 360d
    if np.sum(mask_green) > 0:
        t_g, s_g = compute_kaplan_meier_curve(y_test_true[mask_green], y_test_events[mask_green])
        p_g = get_survival_probability_at_time(t_g, s_g, KM_EVAL_GREEN)
    else:
        t_g, s_g = np.array([0.0, KM_EVAL_GREEN]), np.array([0.0, 0.0])
        p_g = 0.0
        
    # 3. KM Curve for Yellow Class (Moderate Risk)
    if np.sum(mask_yellow) > 0:
        t_y, s_y = compute_kaplan_meier_curve(y_test_true[mask_yellow], y_test_events[mask_yellow])
    else:
        t_y, s_y = np.array([0.0]), np.array([1.0])
        
    p_bar_g = 1.0 - p_g
    
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
                if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1]) and \
                   (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1]):
                    is_pareto[i] = False
                    break
    return is_pareto


# --- 9. Plotting with Extended Time Axis (500+ Days) and Unique Tags ---
def plot_kaplan_meier_curves(km_data, title, save_path, min_x_extent=KM_MAX_DISPLAY_TIME):
    plt.figure(figsize=(9.5, 5.5))
    
    colors = {'Green': '#2ca02c', 'Yellow': '#ff7f0e', 'Red': '#d62728'}
    labels = {
        'Green': f"Green (Pred >= {int(T_GREEN)}d, N={km_data['N_Green']})",
        'Yellow': f"Yellow ({int(T_RED)}-{int(T_GREEN)}d, N={km_data['N_Yellow']})",
        'Red': f"Red (Pred < {int(T_RED)}d, N={km_data['N_Red']})"
    }
    line_widths = {'Green': 2.5, 'Yellow': 2.0, 'Red': 2.5}
    
    # Calculate maximum x-axis extent (at least 500 days)
    observed_max_times = [np.max(km_data["km_curves"][c][0]) for c in ['Green', 'Yellow', 'Red']]
    x_axis_limit = max(min_x_extent, float(np.max(observed_max_times)))
    
    for c in ['Green', 'Yellow', 'Red']:
        t_c, s_c = km_data["km_curves"][c]
        t_plot = list(t_c)
        s_plot = list(s_c)
        # Extend horizontal step function up to x_axis_limit
        if t_plot[-1] < x_axis_limit:
            t_plot.append(x_axis_limit)
            s_plot.append(s_plot[-1])
            
        plt.step(t_plot, s_plot, label=labels[c], color=colors[c], linewidth=line_widths[c], where='post')
    
    # Vertical reference lines at 180 and 360 days with arrows
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
    
    x_ticks = [0, 100, 180, 250, 360, int(x_axis_limit)]
    plt.xticks(x_ticks, labels=[str(xt) for xt in x_ticks])
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scenario_pareto(df_scenario, title, save_path):
    if df_scenario.empty:
        return
        
    pts = df_scenario[["P_bar_G", "P_R"]].values
    is_pareto = compute_pareto_mask(pts[:, 0], pts[:, 1])
    df_scenario["is_pareto"] = is_pareto
    
    pareto_pts = df_scenario[df_scenario["is_pareto"]].sort_values(by="P_bar_G")
    dominated_pts = df_scenario[~df_scenario["is_pareto"]]
    
    plt.figure(figsize=(8.5, 6.5))
    if not dominated_pts.empty:
        plt.scatter(dominated_pts["P_bar_G"], dominated_pts["P_R"], color="#909090", alpha=0.5, s=60, label="Dominated Models")
        for _, r in dominated_pts.iterrows():
            plt.annotate(f"H{int(r['Hidden_H'])}", (r["P_bar_G"], r["P_R"]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#555555")
            
    plt.plot(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", linestyle="-", linewidth=2.0, zorder=4)
    plt.scatter(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", s=110, zorder=5, label="Pareto Frontier (Non-Dominated)")
    for _, r in pareto_pts.iterrows():
        plt.annotate(f"H{int(r['Hidden_H'])}\n({r['Parameters']}p)", (r["P_bar_G"], r["P_R"]), textcoords="offset points", xytext=(6, -6), fontsize=9, fontweight="bold", color="#d62728")
        
    plt.scatter([0], [0], color="#1f77b4", marker="*", s=180, zorder=6, label="Ideal Point (0, 0)")
    plt.xlabel(r"$\bar{P}_G = 1 - P_G$ (Minimize $\to 0$)", fontsize=11)
    plt.ylabel(r"$P_R$ (Minimize $\to 0$)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlim(-0.02, max(1.0, df_scenario["P_bar_G"].max() * 1.08))
    plt.ylim(-0.02, max(1.0, df_scenario["P_R"].max() * 1.08))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_global_pareto_numbered(df_all, title, save_path):
    """
    Plots the Global Pareto Frontier with sequential numbered tags (KM 1, KM 2, ...)
    directly corresponding to the individually exported Kaplan-Meier plots.
    """
    pts = df_all[["P_bar_G", "P_R"]].values
    is_pareto = compute_pareto_mask(pts[:, 0], pts[:, 1])
    df_all["is_pareto"] = is_pareto
    
    pareto_pts = df_all[df_all["is_pareto"]].sort_values(by="P_bar_G").reset_index(drop=True)
    dominated_pts = df_all[~df_all["is_pareto"]]
    
    plt.figure(figsize=(9.5, 7.5))
    if not dominated_pts.empty:
        plt.scatter(dominated_pts["P_bar_G"], dominated_pts["P_R"], color="#a5a5a5", alpha=0.5, s=50, label="Dominated Models")
        
    plt.plot(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", linestyle="-", linewidth=2.2, zorder=4)
    plt.scatter(pareto_pts["P_bar_G"], pareto_pts["P_R"], color="#d62728", s=130, zorder=5, label="Pareto Frontier (Non-Dominated)")
    
    for idx, r in pareto_pts.iterrows():
        km_num = idx + 1
        label_text = f"KM {km_num}"
        offset_y = 12 if idx % 2 == 0 else -20
        offset_x = 10 if r["P_bar_G"] < 0.8 else -45
        
        plt.annotate(
            label_text,
            (r["P_bar_G"], r["P_R"]),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=10,
            fontweight="bold",
            color="#b30000",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", lw=1.2, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0)
        )
        
    plt.scatter([0], [0], color="#1f77b4", marker="*", s=220, zorder=6, label="Ideal Point (0, 0)")
    plt.xlabel(r"$\bar{P}_G = 1 - P_G$ (Minimize $\to 0$)", fontsize=12, fontweight="bold")
    plt.ylabel(r"$P_R$ (Minimize $\to 0$)", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlim(-0.04, 1.06)
    plt.ylim(-0.04, 1.06)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- 10. Main Pipeline Execution ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=========================================================================")
    print(f" PIPELINE: RIGOROUS KM CENSORING, 400D CAP, STANDARDIZED NAMING")
    print(f" COMPUTATION DEVICE: {device}")
    print(f"=========================================================================\n")
    
    all_results = []
    all_km_cache = {}
    
    for ds_entry in DATASET_FILES:
        ds_name = ds_entry["name"]
        ds_file = ds_entry["file"]
        W = ds_entry["W"]
        M = ds_entry["M"]
        is_capped = ds_entry["capped"]
        target_str = "capped" if is_capped else "uncapped"
        
        file_path = find_dataset_file(DATASETS_DIR, ds_file)
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}. Skipping dataset.")
            continue
            
        print(f"\n{'='*90}\n DATASET: {ds_name} (M={M}, W={W}, Target={target_str})\n{'='*90}")
        df_raw, pid_col = load_dataset(file_path)
        X_raw, y_log, y_days, events_all, pid_all, time_all = prepare_raw_data(df_raw, pid_col, is_capped=is_capped)
        
        for split_mode in SPLIT_MODES:
            print(f"\n>>> Split Mode: {split_mode.upper()} <<<")
            train_mask, val_mask, test_mask = create_split_masks(pid_all, time_all, split_mode=split_mode)
            
            X_tr, y_tr_log, y_tr_d, ev_tr = extract_features(X_raw, y_log, y_days, events_all, pid_all, train_mask, W)
            X_va, y_va_log, y_va_d, ev_va = extract_features(X_raw, y_log, y_days, events_all, pid_all, val_mask, W)
            X_te, y_te_log, y_te_d, ev_te = extract_features(X_raw, y_log, y_days, events_all, pid_all, test_mask, W)
            
            if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                print(f"[SKIP] Insufficient samples for split {split_mode}.")
                continue
                
            # Z-score normalization computed exclusively on the Training set
            mean = X_tr.mean(axis=0, keepdims=True)
            std = X_tr.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            
            X_tr_s = (X_tr - mean) / std
            X_va_s = (X_va - mean) / std
            X_te_s = (X_te - mean) / std
            
            train_loader = DataLoader(TabularLogDataset(X_tr_s, y_tr_log, y_tr_d, ev_tr), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TabularLogDataset(X_va_s, y_va_log, y_va_d, ev_va), batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(TabularLogDataset(X_te_s, y_te_log, y_te_d, ev_te), batch_size=BATCH_SIZE, shuffle=False)
            
            N = X_tr_s.shape[1]
            base_steps = [1, 2, 3, 5, 8, 16, 32, 64, 128]
            candidate_H = sorted(list(set([h for h in base_steps if h < N] + [N])))
            
            print(f"Input features N = {N} (4 x {M} measures). Hidden nodes H: {candidate_H}")
            
            arch_history = []
            dataset_split_results = []
            
            for h in candidate_H:
                # Standardized naming: M<features>_W<finestra>_H<nodi>_<target>_<split>
                model_id = f"M{M}_W{W}_H{h}_{target_str}_{split_mode}"
                
                model = SystematicFFNN(input_dim=N, hidden_dim=h).to(device)
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape, test_preds, test_true, test_events = train_model(
                    model, train_loader, val_loader, test_loader, device, is_capped=is_capped
                )
                
                # Save model weights with standardized naming
                weights_path = os.path.join(WEIGHTS_DIR, f"{model_id}.pt")
                torch.save(model.state_dict(), weights_path)
                
                # Evaluate Triage and Kaplan-Meier with explicit event vector
                triage_metrics = evaluate_triage_and_kaplan_meier(test_true, test_preds, test_events)
                
                # Cache survival curve data associated with model_id
                all_km_cache[model_id] = triage_metrics
                
                diff_tr_te = abs(tr_mae - te_mae)
                
                entry = {
                    "Model_ID": model_id,
                    "Pareto_ID": "-",
                    "Dataset": ds_name,
                    "Features": f"M{M}",
                    "Window": f"W{W}",
                    "Hidden": f"H{h}",
                    "Target": target_str,
                    "Split": split_mode,
                    "Measures": M,
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
                
                print(f"  [{model_id}] ({num_params:5d}p) -> Test MAE: {te_mae:5.2f}d | "
                      f"P_R(180d): {triage_metrics['P_R']:.4f} | P_G(360d): {triage_metrics['P_G']:.4f} | "
                      f"Counts (R/Y/G): {triage_metrics['N_Red']}/{triage_metrics['N_Yellow']}/{triage_metrics['N_Green']}")
                
                should_prune, reason = check_architectural_pruning(arch_history)
                if should_prune:
                    print(f"     ==> [PRUNING] Stopped H expansion for {model_id}. Reason: {reason}")
                    break
                    
            # Save individual scenario Pareto plot
            df_scenario = pd.DataFrame(dataset_split_results)
            scenario_pareto_file = os.path.join(PLOTS_DIR, f"pareto_M{M}_W{W}_{target_str}_{split_mode}.png")
            plot_scenario_pareto(df_scenario, f"Pareto: M{M}_W{W}_{target_str}_{split_mode}", scenario_pareto_file)

    # --- 11. Global Pareto Optimization and Numbered KM Plots ---
    df_all = pd.DataFrame(all_results)
    if not df_all.empty:
        df_all["is_pareto"] = compute_pareto_mask(df_all["P_bar_G"].values, df_all["P_R"].values)
        pareto_indices = df_all[df_all["is_pareto"]].sort_values(by="P_bar_G").index
        
        print(f"\n{'='*110}\n EXPORTING NUMBERED KM CURVES FOR PARETO MODELS (KM 1 - KM K)\n{'='*110}")
        
        for idx, orig_idx in enumerate(pareto_indices):
            km_num = idx + 1
            km_tag = f"KM {km_num}"
            df_all.loc[orig_idx, "Pareto_ID"] = km_tag
            
            row = df_all.loc[orig_idx]
            mid = row["Model_ID"]
            
            if mid in all_km_cache:
                km_data = all_km_cache[mid]
                km_title = f"[{km_tag}] Kaplan-Meier Triage - {mid}"
                km_filename = os.path.join(PLOTS_DIR, f"KM_{km_num}_{mid}.png")
                plot_kaplan_meier_curves(km_data, km_title, km_filename, min_x_extent=KM_MAX_DISPLAY_TIME)
                print(f"  --> Saved survival curve [{km_tag}]: {km_filename}")
                
        # Export Global Pareto Frontier plot with numbered badges
        global_pareto_file = os.path.join(PLOTS_DIR, "pareto_frontier_global_numbered.png")
        plot_global_pareto_numbered(df_all, "Global Pareto Frontier with Numbered KM Curves", global_pareto_file)
        print(f"\nSaved Global Pareto Frontier plot: {global_pareto_file}")
        
        # Save complete results table to CSV
        out_csv = os.path.join(BASE_DIR, "final_systematic_ffnn_triage_pareto_results.csv")
        df_all.to_csv(out_csv, index=False)
        print(f"Full results exported to: {out_csv}")
        
        # Summary display of non-dominated Pareto models
        pareto_table = df_all[df_all["is_pareto"]].sort_values(by="P_bar_G")
        disp_cols = ["Pareto_ID", "Model_ID", "Parameters", "Train_MAE_days", "Test_MAE_days", 
                     "Diff_Train_Test", "P_R", "P_G", "P_bar_G", "N_Red", "N_Green"]
        print("\n" + "="*125)
        print(" PARETO OPTIMAL MODELS (MAPPED WITH KM IDENTIFIERS)")
        print("="*125)
        print(pareto_table[disp_cols].to_string(index=False))
        print("="*125)


if __name__ == "__main__":
    main()