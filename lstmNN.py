import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- Configurations ---
CSV_PATH = r"c:\Users\vince\Desktop\NN\NN_training_dataset.csv"
CACHE_DIR = r"c:\Users\vince\Desktop\NN"

# Cache paths for the preprocessed LSTM pipeline (misure only)
X_ALL_PATH = os.path.join(CACHE_DIR, "X_all_lstm_misure.npy")
Y_ALL_PATH = os.path.join(CACHE_DIR, "y_all_lstm_misure.npy")
PID_ALL_PATH = os.path.join(CACHE_DIR, "pid_all_lstm_misure.npy")
TIME_ALL_PATH = os.path.join(CACHE_DIR, "time_all_lstm_misure.npy")

EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 5e-5

# --- 1. Custom Hybrid Loss Function (MAPE + Weighted MAE) ---
class HybridMAPEMAELoss(nn.Module):
    def __init__(self, min_val=10.0, mae_weight=0.01):
        """
        Loss Ibrida: Calcola il MAPE con clamp a denominatore e vi aggiunge una
        penalizzazione ponderata sull'errore assoluto medio (MAE) in giorni.
        """
        super().__init__()
        self.min_val = min_val
        self.mae_weight = mae_weight

    def forward(self, outputs, targets):
        # Componente MAPE
        denom = torch.clamp(targets, min=self.min_val)
        mape = torch.mean(torch.abs(outputs - targets) / denom) * 100.0
        
        # Componente MAE Ponderata
        mae = torch.mean(torch.abs(outputs - targets))
        
        return mape + (self.mae_weight * mae)

# --- 2. Data Preparation and Caching ---
def load_and_preprocess_data():
    cache_exists = all(os.path.exists(p) for p in [X_ALL_PATH, Y_ALL_PATH, PID_ALL_PATH, TIME_ALL_PATH])
    
    if cache_exists:
        print("Loading preprocessed dataset from LSTM numpy cache files...")
        t0 = time.time()
        X_all = np.load(X_ALL_PATH, allow_pickle=True)
        y_all = np.load(Y_ALL_PATH)
        pid_all = np.load(PID_ALL_PATH, allow_pickle=True)
        time_all = np.load(TIME_ALL_PATH, allow_pickle=True)
        print(f"Loaded datasets from cache in {time.time() - t0:.2f} seconds.")
        return X_all, y_all, pid_all, time_all

    print(f"Cache files not found. Parsing CSV file: {CSV_PATH}")
    t_start = time.time()
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV file in {time.time() - t_start:.2f}s. Initial row count: {len(df)}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("Sorting rows by patient_id and chronologically by timestamp...")
    df = df.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
    
    def parse_vector_column(series):
        t0 = time.time()
        all_str = ",".join(series.str.strip('{}'))
        parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
        reshaped = parsed.reshape(len(series), -1)
        print(f"Parsed column '{series.name}' in {time.time() - t0:.2f}s. Feature size: {reshaped.shape[1]}")
        return reshaped

    print("Parsing vector column 'misure'...")
    t_parse = time.time()
    misure = parse_vector_column(df['misure'])
    print(f"Total vector parsing completed in {time.time() - t_parse:.2f}s.")
    
    X_all = misure
    y_all = df['tte'].values.astype(np.float32)
    pid_all = df['patient_id'].values
    time_all = df['timestamp'].values
    
    print("Caching global preprocessed arrays for LSTM...")
    np.save(X_ALL_PATH, X_all)
    np.save(Y_ALL_PATH, y_all)
    np.save(PID_ALL_PATH, pid_all)
    np.save(TIME_ALL_PATH, time_all)
    print("Preprocessed dataset successfully saved to cache.")
    
    return X_all, y_all, pid_all, time_all

# --- 3. Custom Dataset with Feature Delta Augmentation (Delta X) ---
class EnhancedDialysisDataset(Dataset):
    def __init__(self, X, y, pids, allowed_pids, T, use_deltas=True):
        """
        Costruisce finestre 3D aggiungendo opzionalmente le differenze prime (differenziali)
        tra sedute consecutive, raddoppiando il numero delle feature.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.T = T
        self.use_deltas = use_deltas
        
        mask = np.isin(pids, allowed_pids)
        pids_arr = np.array(pids)
        same_patient = pids_arr[T - 1:] == pids_arr[:- (T - 1)]
        target_mask = mask[T - 1:]
        
        valid_flags = same_patient & target_mask
        self.valid_indices = np.where(valid_flags)[0] + (T - 1)
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        X_seq = self.X[target_idx - self.T + 1 : target_idx + 1]  # [T, feature_dim]
        
        if self.use_deltas:
            # Calcolo differenze prime temporali lungo la finestra T
            deltas = torch.zeros_like(X_seq)
            deltas[1:] = X_seq[1:] - X_seq[:-1]
            X_seq = torch.cat([X_seq, deltas], dim=-1)  # [T, feature_dim * 2]
            
        y_target = self.y[target_idx]
        return X_seq, y_target

# --- 4. FIXED WINNING ARCHITECTURE: LSTM_BiLSTM_32_1L ---
class FixedBiLSTM(nn.Module):
    def __init__(self, input_dim):
        """
        Architettura fissa BiLSTM a 32 unita' per direzione (64 totali) con LayerNorm.
        Adattata dinamicamente alla dimensione delle feature in ingresso.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.head(last_step)

# --- Early Stopping Helper ---
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- 5. Training Loop ---
def train_and_evaluate_bilstm(model, train_loader, val_loader, test_loader, exp_config, exp_name, device):
    if exp_config["loss_type"] == "hybrid":
        criterion = HybridMAPEMAELoss(min_val=10.0, mae_weight=0.01)
    else:
        criterion = HybridMAPEMAELoss(min_val=10.0, mae_weight=0.0)  # Pure MAPE
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    if exp_config["scheduler"] == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses, val_losses, val_maes, test_losses, test_maes = [], [], [], [], []
    
    best_val_loss = float('inf')  
    best_val_mae = float('inf')
    best_test_loss = float('inf') 
    best_test_mae = float('inf')
    
    early_stopping = EarlyStopping(patience=30)
    eval_mape_calc = HybridMAPEMAELoss(min_val=10.0, mae_weight=0.0)
    
    print(f"\n--- Training {exp_name} on {device} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_train_loss += loss.item() * X_batch.size(0)
            
        scheduler.step()
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluation step
        model.eval()
        epoch_val_loss, epoch_val_mae, epoch_test_loss, epoch_test_mae = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                loss = eval_mape_calc(outputs, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
                epoch_val_mae += torch.abs(outputs - y_batch).sum().item()
                
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                loss = eval_mape_calc(outputs, y_batch)
                epoch_test_loss += loss.item() * X_batch.size(0)
                epoch_test_mae += torch.abs(outputs - y_batch).sum().item()
                
        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_mae /= len(val_loader.dataset)
        epoch_test_loss /= len(test_loader.dataset)
        epoch_test_mae /= len(test_loader.dataset)
        
        val_losses.append(epoch_val_loss)
        val_maes.append(epoch_val_mae)
        test_losses.append(epoch_test_loss)
        test_maes.append(epoch_test_mae)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_mae = epoch_val_mae
            best_test_loss = epoch_test_loss
            best_test_mae = epoch_test_mae
            torch.save(model.state_dict(), f"best_weights_{exp_name}.pth")
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{EPOCHS:03d} | LR: {current_lr:.1e} | Train Loss: {epoch_train_loss:.2f} | "
              f"Val MAPE: {epoch_val_loss:.2f}% (MAE: {epoch_val_mae:.2f}gg) | "
              f"Test MAPE: {epoch_test_loss:.2f}% (MAE: {epoch_test_mae:.2f}gg)")
        
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model weights...")
            break
            
    if os.path.exists(f"best_weights_{exp_name}.pth"):
        model.load_state_dict(torch.load(f"best_weights_{exp_name}.pth"))
        
    return train_losses, val_losses, val_maes, test_losses, test_maes, best_test_loss, best_test_mae, best_val_loss, best_val_mae

# --- 6. Main Execution Pipeline ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load Data
    X_all, y_all, pid_all, time_all = load_and_preprocess_data()
    
    # 2. Patient-wise Split (50% train / 10% val / 40% test)
    unique_pids = np.unique(pid_all)
    np.random.seed(42)
    shuffled_pids = unique_pids.copy()
    np.random.shuffle(shuffled_pids)
    
    n_patients = len(shuffled_pids)
    train_end = int(n_patients * 0.5)
    val_end = int(n_patients * 0.6)
    
    train_pids = shuffled_pids[:train_end]
    val_pids = shuffled_pids[train_end:val_end]
    test_pids = shuffled_pids[val_end:]
    
    print(f"\nPatient-wise split summary: Total Unique Patients = {n_patients}")
    print(f"Train: {len(train_pids)} patients | Val: {len(val_pids)} patients | Test: {len(test_pids)} patients")
    
    train_mask = np.isin(pid_all, train_pids)
    
    # Standardization
    mean = X_all[train_mask].mean(axis=0, keepdims=True)
    std = X_all[train_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_scaled = (X_all - mean) / std
    
    seq_len = 15
    
    # 3. Definiamo i 4 Esperimenti Avanzati sulla Pipeline della BiLSTM_32_1L
    experiments = [
        {
            "name": "BiLSTM_32_1L_Winner_Baseline",
            "use_deltas": False,
            "loss_type": "pure_mape",
            "scheduler": "cosine"
        },
        {
            "name": "BiLSTM_32_1L_FeatureDeltas_52Dim",
            "use_deltas": True,
            "loss_type": "pure_mape",
            "scheduler": "cosine"
        },
        {
            "name": "BiLSTM_32_1L_WarmRestarts",
            "use_deltas": False,
            "loss_type": "pure_mape",
            "scheduler": "warm_restarts"
        },
        {
            "name": "BiLSTM_32_1L_Deltas_HybridLoss",
            "use_deltas": True,
            "loss_type": "hybrid",
            "scheduler": "warm_restarts"
        }
    ]
    
    results = {}
    plt.figure(figsize=(12, 8))
    
    for exp in experiments:
        name = exp["name"]
        print(f"\nConfiguring Advanced Pipeline Experiment for {name}...")
        
        train_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, train_pids, T=seq_len, use_deltas=exp["use_deltas"])
        val_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, val_pids, T=seq_len, use_deltas=exp["use_deltas"])
        test_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, test_pids, T=seq_len, use_deltas=exp["use_deltas"])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        num_features = X_scaled.shape[1]
        input_dim = num_features * 2 if exp["use_deltas"] else num_features
        model = FixedBiLSTM(input_dim=input_dim).to(device)
        
        train_hist, val_loss_hist, val_mae_hist, test_loss_hist, test_mae_hist, best_test_loss, best_test_mae, best_val_loss, best_val_mae = train_and_evaluate_bilstm(
            model, train_loader, val_loader, test_loader, exp, name, device
        )
        
        results[name] = {
            "train_history": train_hist,
            "val_loss_history": val_loss_hist,
            "val_mae_history": val_mae_hist,
            "test_loss_history": test_loss_hist,
            "test_mae_history": test_mae_hist,
            "best_mape": best_test_loss,        
            "best_mae": best_test_mae,          
            "best_val_mape": best_val_loss,    
            "best_val_mae": best_val_mae       
        }
        
        plt.plot(range(1, len(test_loss_hist) + 1), test_loss_hist, label=f"{name} (Best Test MAPE: {best_test_loss:.2f}%)")
        
    plt.title("BiLSTM_32_1L Advanced Pipeline Enhancements Test MAPE (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Percentage Error (MAPE) in %")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plot_path = os.path.join(CACHE_DIR, "lstm_mape_loss_training.png")
    plt.savefig(plot_path)
    print(f"\nMAPE training curve plot saved to: {plot_path}")
    
    print("\n" + "="*110)
    print(f"{'Pipeline Configuration Name':<40} | {'Best Val MAPE':<15} | {'Best Val MAE':<15} | {'Best Test MAPE':<15} | {'Best Test MAE':<15}")
    print("-"*110)
    best_exp_name = None
    best_mape = float('inf')
    
    for name, stats in results.items():
        print(f"{name:<40} | {stats['best_val_mape']:<14.2f}% | {stats['best_val_mae']:<12.2f} gg | {stats['best_mape']:<14.2f}% | {stats['best_mae']:<12.2f} gg")
        if stats['best_mape'] < best_mape:
            best_mape = stats['best_mape']
            best_exp_name = name
            
    print("="*110)
    print(f"Recommended Best Pipeline Setup: {best_exp_name} with Test MAPE of {best_mape:.2f}%.")
    print("="*110)

if __name__ == "__main__":
    main()