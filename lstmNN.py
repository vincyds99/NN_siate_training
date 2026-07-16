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
X_ALL_PATH = os.path.join(CACHE_DIR, "X_all_lstm_no2017_misure.npy")
Y_ALL_PATH = os.path.join(CACHE_DIR, "y_all_lstm_no2017_misure.npy")
PID_ALL_PATH = os.path.join(CACHE_DIR, "pid_all_lstm_no2017_misure.npy")
TIME_ALL_PATH = os.path.join(CACHE_DIR, "time_all_lstm_no2017_misure.npy")

EPOCHS = 500
BATCH_SIZE = 256  # Ridotto a 256 per favorire una migliore regolarizzazione e convergenza dei gradienti
LEARNING_RATE = 1e-3

# --- 1. Data Preparation, Filtering and Caching ---
def load_and_preprocess_data():
    """
    Carica i vettori pre-elaborati se presenti in cache.
    In caso contrario, esegue il parsing del CSV filtrando il 2017 e isolando la colonna 'misure'.
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
    
    # Filtro anno 2017
    print("Filtering out rows from the year 2017...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.year != 2017].reset_index(drop=True)
    print(f"Rows remaining after 2017 exclusion: {len(df)}")
    
    # Ordinamento cronologico per paziente
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

# --- 2. Custom Dataset for Sliding Window Temporal Ingestion ---
class DialysisLSTMDataset(Dataset):
    def __init__(self, X, y, pids, timestamps, train_split_date, val_split_date, split_type, T):
        """
        Dataset personalizzato per la costruzione on-the-fly di finestre temporali 3D.
        Garantisce in modo stringente che nessuna finestra contenga dati di pazienti diversi.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.T = T
        
        # Split cronologico
        ts_series = pd.to_datetime(timestamps)
        if split_type == "train":
            mask = np.asarray(ts_series <= train_split_date)
        elif split_type == "val":
            mask = np.asarray((ts_series > train_split_date) & (ts_series <= val_split_date))
        elif split_type == "test":
            mask = np.asarray(ts_series > val_split_date)
        else:
            raise ValueError(f"Unknown split_type: {split_type}")
            
        # Controllo di integrita del paziente:
        # Verifica se l'ID del paziente all'indice i corrisponde a quello all'indice i - (T - 1)
        pids_arr = np.array(pids)
        same_patient = pids_arr[T - 1:] == pids_arr[:- (T - 1)]
        target_mask = mask[T - 1:]
        
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
    def __init__(self, input_dim, arch_type="light_32_1l"):
        """
        Architetture LSTM parametrizzate senza l'uso del Dropout, 
        per testare differenti livelli di capacita rappresentativa su sequenze fisse T = 15.
        """
        super().__init__()
        self.arch_type = arch_type
        
        if arch_type == "light_32_1l":
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=1, batch_first=True)
            self.fc = nn.Linear(32, 1)
            
        elif arch_type == "medium_64_2l":
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, batch_first=True)
            self.fc = nn.Linear(64, 1)
            
        elif arch_type == "wide_128_1l":
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=1, batch_first=True)
            self.fc = nn.Linear(128, 1)
            
        elif arch_type == "complex_128_2l":
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=2, batch_first=True)
            self.fc = nn.Linear(128, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Estrazione dell'ultimo passo temporale della finestra (T = 15)
        last_timestep = lstm_out[:, -1, :]
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
    val_mapes = []
    test_losses = []
    test_maes = []
    test_mapes = []
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_mape = float('inf')
    best_test_loss = float('inf')
    best_test_mae = float('inf')
    best_test_mape = float('inf')
    
    early_stopping = EarlyStopping(patience=20)
    
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
        
        # Evaluation step
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae = 0.0
        epoch_val_mape = 0.0
        epoch_test_loss = 0.0
        epoch_test_mae = 0.0
        epoch_test_mape = 0.0
        
        with torch.no_grad():
            # Validation
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
                
                # MAE in giorni
                mae = torch.abs(outputs - y_batch).sum().item()
                epoch_val_mae += mae
                
                # MAPE (con clamp preventivo del denominatore a minimo 1.0 giorno)
                mape = (torch.abs(outputs - y_batch) / torch.clamp(y_batch, min=1.0)).sum().item()
                epoch_val_mape += mape
                
            # Test
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_test_loss += loss.item() * X_batch.size(0)
                
                # MAE in giorni
                mae = torch.abs(outputs - y_batch).sum().item()
                epoch_test_mae += mae
                
                # MAPE
                mape = (torch.abs(outputs - y_batch) / torch.clamp(y_batch, min=1.0)).sum().item()
                epoch_test_mape += mape
                
        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_mae /= len(val_loader.dataset)
        epoch_val_mape = (epoch_val_mape / len(val_loader.dataset)) * 100.0
        
        epoch_test_loss /= len(test_loader.dataset)
        epoch_test_mae /= len(test_loader.dataset)
        epoch_test_mape = (epoch_test_mape / len(test_loader.dataset)) * 100.0
        
        val_losses.append(epoch_val_loss)
        val_maes.append(epoch_val_mae)
        val_mapes.append(epoch_val_mape)
        test_losses.append(epoch_test_loss)
        test_maes.append(epoch_test_mae)
        test_mapes.append(epoch_test_mape)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_mae = epoch_val_mae
            best_val_mape = epoch_val_mape
            best_test_loss = epoch_test_loss
            best_test_mae = epoch_test_mae
            best_test_mape = epoch_test_mape
            torch.save(model.state_dict(), f"best_weights_{model_name}.pth")
            
        print(f"Epoch {epoch+1:03d}/{EPOCHS:03d} | Train MSE: {epoch_train_loss:.2f} | "
              f"Val MSE: {epoch_val_loss:.2f} | Val MAE: {epoch_val_mae:.2f} (MAPE: {epoch_val_mape:.2f}%) | "
              f"Test MAE: {epoch_test_mae:.2f} (MAPE: {epoch_test_mape:.2f}%)")
        
        # Early stopping basata sulla perdita di validazione
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model weights...")
            break
            
    if os.path.exists(f"best_weights_{model_name}.pth"):
        model.load_state_dict(torch.load(f"best_weights_{model_name}.pth"))
        
    return train_losses, val_losses, val_maes, val_mapes, test_losses, test_maes, test_mapes, best_test_loss, best_test_mae, best_test_mape, best_val_loss, best_val_mae, best_val_mape

# --- 5. Main Execution Pipeline ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load Data
    X_all, y_all, pid_all, time_all = load_and_preprocess_data()
    
    # 2. Determine Chronological Split Points (50% train / 10% val / 40% test)
    sorted_times = np.sort(time_all)
    train_split_idx = int(len(sorted_times) * 0.5)
    val_split_idx = int(len(sorted_times) * 0.6)
    train_split_date = pd.to_datetime(sorted_times[train_split_idx])
    val_split_date = pd.to_datetime(sorted_times[val_split_idx])
    print(f"Chronological split dates: Train <= {train_split_date} | Val <= {val_split_date} | Test > {val_split_date}")
    
    # 3. Standardization based on training statistics to prevent leakage
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
    
    # 4. Definizione delle quattro architetture da testare a finestra fissa T = 15 (1 mese)
    # Rimosso completamente il dropout in tutti i modelli
    architectures = [
        ("LSTM_Light_32_1L", "light_32_1l", 15),
        ("LSTM_Medium_64_2L", "medium_64_2l", 15),
        ("LSTM_Wide_128_1L", "wide_128_1l", 15),
        ("LSTM_Complex_128_2L", "complex_128_2l", 15)
    ]
    
    results = {}
    plt.figure(figsize=(12, 8))
    
    # 5. Iterative Experimentation Loop
    for name, arch_type, seq_len in architectures:
        print(f"\nConfiguring Dataset for {name} with Sequence Length T = {seq_len}...")
        
        train_dataset = DialysisLSTMDataset(X_scaled, y_all, pid_all, time_all, train_split_date, val_split_date, split_type="train", T=seq_len)
        val_dataset = DialysisLSTMDataset(X_scaled, y_all, pid_all, time_all, train_split_date, val_split_date, split_type="val", T=seq_len)
        test_dataset = DialysisLSTMDataset(X_scaled, y_all, pid_all, time_all, train_split_date, val_split_date, split_type="test", T=seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = FlexibleLSTM(
            input_dim=input_dim,
            arch_type=arch_type
        ).to(device)
        
        train_hist, val_loss_hist, val_mae_hist, val_mape_hist, test_loss_hist, test_mae_hist, test_mape_hist, best_test_loss, best_test_mae, best_test_mape, best_val_loss, best_val_mae, best_val_mape = train_and_evaluate_lstm(
            model, train_loader, val_loader, test_loader, name, device
        )
        
        results[name] = {
            "train_history": train_hist,
            "val_loss_history": val_loss_hist,
            "val_mae_history": val_mae_hist,
            "val_mape_history": val_mape_hist,
            "test_loss_history": test_loss_hist,
            "test_mae_history": test_mae_hist,
            "test_mape_history": test_mape_hist,
            "best_mse": best_test_loss,
            "best_rmse": np.sqrt(best_test_loss),
            "best_mae": best_test_mae,
            "best_mape": best_test_mape,
            "best_val_mae": best_val_mae,
            "best_val_mape": best_val_mape
        }
        
        # Plot basato sull'andamento del Test MAPE (%) lungo le epoche (fino all'Early Stopping)
        plt.plot(range(1, len(test_mape_hist) + 1), test_mape_hist, label=f"{name} (Best MAPE: {best_test_mape:.2f}%)")
        
    # Styling dei grafici
    plt.title("LSTM Test MAPE (%) Comparison Across Architectural Configurations (Fixed T = 15, No Dropout)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Percentage Error (MAPE) in %")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plot_path = os.path.join(CACHE_DIR, "lstm_mape_comparison.png")
    plt.savefig(plot_path)
    print(f"\nMAPE comparison curve plot saved to: {plot_path}")
    
    # 6. Tabella finale di riepilogo scientifico per la tesi
    print("\n" + "="*125)
    print(f"{'LSTM Architecture Model':<25} | {'Best Val MAE':<12} | {'Best Val MAPE':<13} | {'Best Test MSE':<13} | {'Best Test MAE':<13} | {'Best Test MAPE':<14}")
    print("-"*125)
    best_model_name = None
    best_mape = float('inf')
    
    for name, stats in results.items():
        print(f"{name:<25} | {stats['best_val_mae']:<12.2f} | {stats['best_val_mape']:<12.2f}% | {stats['best_mse']:<13.2f} | {stats['best_mae']:<13.2f} days | {stats['best_mape']:<13.2f}%")
        if stats['best_mape'] < best_mape:
            best_mape = stats['best_mape']
            best_model_name = name
            
    print("="*125)
    print(f"Recommended Best Architectural Configuration (T=15): {best_model_name} with MAPE of {best_mape:.2f}%.")
    print("="*125)

if __name__ == "__main__":
    main()