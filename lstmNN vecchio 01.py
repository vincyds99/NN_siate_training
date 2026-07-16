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

# New unique cache paths specific to the filtered LSTM pipeline (misure only)
X_ALL_PATH = os.path.join(CACHE_DIR, "X_all_lstm_no2017_misure.npy")
Y_ALL_PATH = os.path.join(CACHE_DIR, "y_all_lstm_no2017_misure.npy")
PID_ALL_PATH = os.path.join(CACHE_DIR, "pid_all_lstm_no2017_misure.npy")
TIME_ALL_PATH = os.path.join(CACHE_DIR, "time_all_lstm_no2017_misure.npy")

EPOCHS = 500
BATCH_SIZE = 512 # Andiamo a 256/128
LEARNING_RATE = 1e-3


# --- 1. Data Preparation, Filtering and Caching ---
def load_and_preprocess_data():
    """
    Loads pre-processed numpy arrays if they exist.
    Otherwise, reads the CSV, filters out the year 2017, parses vectors,
    sorts by patient and time, and caches the global arrays.
    """
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
    
    # Filter out sessions from the year 2017
    print("Filtering out rows from the year 2017...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.year != 2017].reset_index(drop=True)
    print(f"Rows remaining after 2017 exclusion: {len(df)}")
    
    # Sort primarily by patient_id and secondarily by timestamp for sequence consistency
    print("Sorting rows by patient_id and chronologically by timestamp...")
    df = df.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
    
    def parse_vector_column(series):
        t0 = time.time()
        all_str = ",".join(series.str.strip('{}'))
        parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
        reshaped = parsed.reshape(len(series), -1)
        print(f"Parsed column '{series.name}' in {time.time() - t0:.2f}s. Feature size: {reshaped.shape[1]}")
        return reshaped

    print("Parsing vector columns...")
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

# --- 2. Custom Dataset for Sliding Window Temporal Ingestion ---
class DialysisLSTMDataset(Dataset):
    def __init__(self, X, y, pids, timestamps, train_split_date, val_split_date, split_type, T):
        """
        Custom Dataset that builds 3D sliding windows on the fly.
        Ensures a window never contains data from more than one patient.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.T = T
        
        # Determine train/val/test mask based on target timestamp position
        ts_series = pd.to_datetime(timestamps)
        if split_type == "train":
            mask = np.asarray(ts_series <= train_split_date)
        elif split_type == "val":
            mask = np.asarray((ts_series > train_split_date) & (ts_series <= val_split_date))
        elif split_type == "test":
            mask = np.asarray(ts_series > val_split_date)
        else:
            raise ValueError(f"Unknown split_type: {split_type}")
            
        # Vectorized check: verify if patient ID at index i matches patient ID at index i - (T - 1)
        pids_arr = np.array(pids)
        same_patient = pids_arr[T - 1:] == pids_arr[:- (T - 1)]
        target_mask = mask[T - 1:]
        
        # Combine filters to get global indices of valid sequence endpoints
        valid_flags = same_patient & target_mask
        self.valid_indices = np.where(valid_flags)[0] + (T - 1)
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        X_seq = self.X[target_idx - self.T + 1 : target_idx + 1]
        y_target = self.y[target_idx]
        return X_seq, y_target

# --- 3. LSTM Neural Network Architectures ---
class FlexibleLSTM(nn.Module):
    def __init__(self, input_dim, arch_type="light", dropout_prob=0.0):
        super().__init__()
        self.arch_type = arch_type
        
        if arch_type == "light":
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=1, batch_first=True)
            self.fc = nn.Linear(32, 1)
            
        elif arch_type == "deep":
            self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=1, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
            self.fc = nn.Linear(32, 1)
            
        elif arch_type == "regularized":
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(dropout_prob)
            self.fc = nn.Linear(128, 1)

    def forward(self, x):
        if self.arch_type == "light":
            lstm_out, _ = self.lstm(x)
            last_timestep = lstm_out[:, -1, :]
            return self.fc(last_timestep)
            
        elif self.arch_type == "deep":
            out1, _ = self.lstm1(x)
            out2, _ = self.lstm2(out1)
            last_timestep = out2[:, -1, :]
            return self.fc(last_timestep)
            
        elif self.arch_type == "regularized":
            lstm_out, _ = self.lstm(x)
            last_timestep = lstm_out[:, -1, :]
            last_timestep = self.dropout(last_timestep)
            return self.fc(last_timestep)

# --- Early Stopping Helper ---
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
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

# --- 4. Training and Evaluation Loop ---
def train_and_evaluate_lstm(model, train_loader, val_loader, test_loader, model_name, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses = []
    val_losses = []
    val_maes = []
    test_losses = []
    test_maes = []
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_test_loss = float('inf')
    best_test_mae = float('inf')
    
    early_stopping = EarlyStopping(patience=20) # 20 Epochs of tolerance
    
    print(f"\n--- Training {model_name} on {device} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * X_batch.size(0)
            
        scheduler.step()
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation and Test Evaluation steps
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae = 0.0
        epoch_test_loss = 0.0
        epoch_test_mae = 0.0
        
        with torch.no_grad():
            # Validate
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
                mae = torch.abs(outputs - y_batch).sum().item()
                epoch_val_mae += mae
                
            # Test
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_test_loss += loss.item() * X_batch.size(0)
                mae = torch.abs(outputs - y_batch).sum().item()
                epoch_test_mae += mae
                
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
            torch.save(model.state_dict(), f"best_weights_{model_name}.pth")
            
        print(f"Epoch {epoch+1:03d}/{EPOCHS:03d} | Train MSE: {epoch_train_loss:.2f} | Val MSE: {epoch_val_loss:.2f} | Val MAE: {epoch_val_mae:.2f} days | Test MSE: {epoch_test_loss:.2f} | Test MAE: {epoch_test_mae:.2f} days")
        
        # Check early stopping criterion on validation loss
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model weights...")
            break
            
    if os.path.exists(f"best_weights_{model_name}.pth"):
        model.load_state_dict(torch.load(f"best_weights_{model_name}.pth"))
        
    return train_losses, val_losses, val_maes, test_losses, test_maes, best_test_loss, best_test_mae, best_val_loss, best_val_mae

# --- 5. Main Execution Pipeline ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load Data (from CSV or Numpy Cache)
    X_all, y_all, pid_all, time_all = load_and_preprocess_data()
    
    # 2. Determine Chronological Split Points (50% training / 10% validation / 40% testing)
    sorted_times = np.sort(time_all)
    train_split_idx = int(len(sorted_times) * 0.5)
    val_split_idx = int(len(sorted_times) * 0.6)
    train_split_date = pd.to_datetime(sorted_times[train_split_idx])
    val_split_date = pd.to_datetime(sorted_times[val_split_idx])
    print(f"Chronological split dates: Train <= {train_split_date} | Val <= {val_split_date} | Test > {val_split_date}")
    
    # 3. Standardization based strictly on training statistics to prevent leakage
    print("\nStandardizing features based on training segment...")
    ts_series = pd.to_datetime(time_all)
    train_mask = ts_series <= train_split_date
    
    mean = X_all[train_mask].mean(axis=0, keepdims=True)
    std = X_all[train_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    
    X_scaled = (X_all - mean) / std
    print("Features normalized cleanly (Mean = 0, Std = 1).")
    
    input_dim = X_scaled.shape[1]
    print(f"Input dimensions: {input_dim}")
    
    # 4. Define the Three LSTM Architectures
    architectures = [
        ("Architecture_1_Light", "light", 5, 0.0),
        ("Architecture_2_Deep", "deep", 10, 0.0),
        ("Architecture_3_Regularized", "regularized", 15, 0.2)
    ]
    
    results = {}
    plt.figure(figsize=(12, 8))
    
    # 5. Iterative Experimentation Loop
    for name, arch_type, seq_len, dropout in architectures:
        print(f"\nConfiguring Dataset for {name} with Sequence Length T = {seq_len}...")
        
        train_dataset = DialysisLSTMDataset(X_scaled, y_all, pid_all, time_all, train_split_date, val_split_date, split_type="train", T=seq_len)
        val_dataset = DialysisLSTMDataset(X_scaled, y_all, pid_all, time_all, train_split_date, val_split_date, split_type="val", T=seq_len)
        test_dataset = DialysisLSTMDataset(X_scaled, y_all, pid_all, time_all, train_split_date, val_split_date, split_type="test", T=seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = FlexibleLSTM(
            input_dim=input_dim,
            arch_type=arch_type,
            dropout_prob=dropout
        ).to(device)
        
        train_hist, val_loss_hist, val_mae_hist, test_loss_hist, test_mae_hist, best_test_loss, best_test_mae, best_val_loss, best_val_mae = train_and_evaluate_lstm(
            model, train_loader, val_loader, test_loader, name, device
        )
        
        results[name] = {
            "train_history": train_hist,
            "val_loss_history": val_loss_hist,
            "val_mae_history": val_mae_hist,
            "test_loss_history": test_loss_hist,
            "test_mae_history": test_mae_hist,
            "best_mse": best_test_loss,
            "best_rmse": np.sqrt(best_test_loss),
            "best_mae": best_test_mae,
            "best_val_mae": best_val_mae
        }
        
        # Plot matches the exact length of test_mae_hist due to Early Stopping truncation
        plt.plot(range(1, len(test_mae_hist) + 1), test_mae_hist, label=f"{name} (Best MAE: {best_test_mae:.2f})")
        
    # Styling evaluation plots
    plt.title("LSTM Test MAE Loss Comparison Across Temporal Architectures (with Early Stopping)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (MAE) in Days")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plot_path = os.path.join(CACHE_DIR, "lstm_mae_comparison.png")
    plt.savefig(plot_path)
    print(f"\nMAE comparison curve plot saved to: {plot_path}")
    
    # 6. Final Experimental Evaluation Summary Table
    print("\n" + "="*95)
    print(f"{'LSTM Architecture Model':<30} | {'Best Val MAE':<12} | {'Best Test MSE':<13} | {'Best Test RMSE':<14} | {'Best Test MAE':<13}")
    print("-"*95)
    best_model_name = None
    best_rmse = float('inf')
    
    for name, stats in results.items():
        print(f"{name:<30} | {stats['best_val_mae']:<12.2f} | {stats['best_mse']:<13.2f} | {stats['best_rmse']:<14.2f} | {stats['best_mae']:<13.2f} days")
        if stats['best_rmse'] < best_rmse:
            best_rmse = stats['best_rmse']
            best_model_name = name
            
    print("="*95)
    print(f"Recommended Best Sequential Architecture: {best_model_name} with RMSE of {best_rmse:.2f} days.")
    print("="*95)

if __name__ == "__main__":
    main()