import os
import math
from typing import Tuple, List

import numpy as np
import psycopg2
from scipy.signal import decimate
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import time
from collections import deque
import datetime

# --- Hyperparameters ---
DSN = os.getenv("DATABASE_URL", "postgres://pico:pass@localhost:5432/piezo_data")
WINDOW_SEC = 1.0
SLIDE_SEC = 0.1                # Stride for data augmentation
SAMPLE_RATE = 200_000
DOWNSAMPLE_FACTOR = 1  # No downsampling for higher resolution

DS_RATE = SAMPLE_RATE // DOWNSAMPLE_FACTOR  # 200 kHz
DS_LEN = int(WINDOW_SEC * DS_RATE)      # 200,000 samples

# FFT Hyperparameters
NFFT = 131_072  # FFT size (power of 2 for efficiency)
FFT_LOW_HZ = 5_000.0
FFT_HIGH_HZ = 60_000.0
# Calculate actual bins available in frequency range
freq_resolution = DS_RATE / NFFT
low_bin = int(FFT_LOW_HZ / freq_resolution)
high_bin = int(FFT_HIGH_HZ / freq_resolution)
NUM_BINS_SPECTRO = high_bin - low_bin  # Should be ~36,045 bins

# GPU-friendly hyperparameters
EPOCHS = 20  # Increased for better training
BATCH_SIZE = 64 # Larger batch size for GPU
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2 # 20% of data for validation
EARLY_STOP_PATIENCE = 3 # Stop if val loss doesn't improve for 3 epochs


def get_windows() -> List[np.ndarray]:
    """Fetch all 'voltages' arrays from the windows table."""
    try:
        conn = psycopg2.connect(DSN)
        print("✅ Connected to PostgreSQL.")
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to PostgreSQL: {e}")
        print("Please ensure the Docker container is running and the DSN is correct.")
        return []
        
    with conn.cursor() as cur:
        cur.execute("SELECT voltages FROM windows WHERE label_normal = TRUE ORDER BY id")
        rows = cur.fetchall()
    print(f"✅ Fetched {len(rows)} normal windows from the database.")
    conn.close()
    return [np.asarray(r[0], dtype=np.float32) for r in rows]


def make_dataset() -> np.ndarray:
    """
    Processes raw windows into a downsampled, augmented dataset.
    It concatenates all data, then creates overlapping sliding windows.
    """
    windows = get_windows()
    if not windows:
        print("❌ No data found. Exiting.")
        return np.array([])

    # 1. Concatenate all database rows into a single continuous signal
    continuous_signal = np.concatenate(windows)
    print(f"✅ Combined data into a single signal of length {len(continuous_signal)} samples.")

    # 2. Create overlapping slices from the continuous signal
    win_len_samples = int(WINDOW_SEC * SAMPLE_RATE)
    stride_samples = int(SLIDE_SEC * SAMPLE_RATE)
    
    slices = []
    for start_idx in range(0, len(continuous_signal) - win_len_samples + 1, stride_samples):
        end_idx = start_idx + win_len_samples
        slices.append(continuous_signal[start_idx:end_idx])

    if not slices:
        print("❌ Signal is not long enough to create any 1-second windows.")
        return np.array([])
        
    print(f"✅ Generated {len(slices)} overlapping windows for data augmentation.")

    # Skip downsampling since DOWNSAMPLE_FACTOR=1
    ds_windows = slices  # Already at full resolution
    ds_windows = [arr[:DS_LEN] for arr in ds_windows]  # Ensure exact length

    # 4. Compute FFT features for each window
    print("⏳ Computing FFT features...")
    features = []
    for ds in ds_windows:
        # Compute FFT on the raw downsampled data
        fft_result = np.fft.rfft(ds, n=NFFT)
        freqs = np.fft.rfftfreq(NFFT, d=1.0 / DS_RATE)

        # Filter frequencies in the range
        mask = (freqs >= FFT_LOW_HZ) & (freqs <= FFT_HIGH_HZ)
        band = np.abs(fft_result[mask])

        # Use full spectrum - no binning for 55k bins
        if len(band) >= NUM_BINS_SPECTRO:
            spectro_bins = band[:NUM_BINS_SPECTRO]
        else:
            # Pad with zeros if needed
            spectro_bins = np.pad(band, (0, NUM_BINS_SPECTRO - len(band)), 'constant')

        # Concatenate downsampled time data + spectro bins
        combined = np.concatenate([ds, spectro_bins])
        features.append(combined)

    # Final feature matrix
    X = np.vstack(features).astype(np.float32)
    print(f"✅ Final augmented dataset shape: {X.shape}")  # (N, 200000 + 55000)
    return X

class VoltageDataset(Dataset):
    """Dataset for time-domain vectors kept on CPU."""
    def __init__(self, X_time: np.ndarray):
        self.X = torch.from_numpy(X_time).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class SpectroDataset(Dataset):
    """Dataset for spectral vectors kept on CPU."""
    def __init__(self, X_spectro: np.ndarray):
        self.X = torch.from_numpy(X_spectro).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class ConvAutoEncoder1D(nn.Module):
    """
    A 1D convolutional auto-encoder.
    The encoder compresses the input vector, and the decoder attempts to reconstruct it.
    """
    def __init__(self, input_len: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), # (B, 1, L) -> (B, 16, L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), # (B, 16, L/2) -> (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), # (B, 32, L/4) -> (B, 64, L/8)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh() # Use Tanh to keep output values between -1 and 1
        )
        self.input_len = input_len

    def forward(self, x):  # Input shape: (Batch, Length)
        # Add a channel dimension for Conv1D
        x = x.unsqueeze(1) # (B, 1, L)
        
        z = self.encoder(x)
        recon = self.decoder(z)
        
        # Remove channel dimension
        recon = recon.squeeze(1) # (B, L)

        # Ensure output length matches input, cropping if necessary
        if recon.size(1) > self.input_len:
            recon = recon[:, : self.input_len]
        return recon


def train():
    # --- 1. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Training will be on CPU and may be slow.")

    # --- 2. Create Dataset ---
    X = make_dataset()
    if X.size == 0: return

    print(f"📊 Total samples created: {X.shape[0]}")

    # Split into time and spectro parts
    X_time = X[:, :DS_LEN]
    X_spectro = X[:, DS_LEN:]

    # Normalise each part separately to [-1,1]
    X_time /= np.max(np.abs(X_time), axis=1, keepdims=True)
    X_spectro /= np.max(np.abs(X_spectro), axis=1, keepdims=True)

    time_dataset = VoltageDataset(X_time)
    spectro_dataset = SpectroDataset(X_spectro)

    # Split datasets into train/val
    val_size = int(len(time_dataset) * VALIDATION_SPLIT)
    train_size = len(time_dataset) - val_size
    time_train, time_val = random_split(time_dataset, [train_size, val_size])
    spectro_train, spectro_val = random_split(spectro_dataset, [train_size, val_size])
    
    print(f" splitting into {len(time_train)} training samples and {len(time_val)} validation samples.")

    time_loader = DataLoader(
        time_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    time_val_loader = DataLoader(
        time_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    spectro_loader = DataLoader(spectro_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    spectro_val_loader = DataLoader(spectro_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # --- 3. Initialize Models ---
    time_model = ConvAutoEncoder1D(DS_LEN).to(device)

    class SpectroAutoEncoder(nn.Module):
        def __init__(self, input_len=NUM_BINS_SPECTRO):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), # (B, 1, L) -> (B, 16, L/2)
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), # (B, 16, L/2) -> (B, 32, L/4)
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), # (B, 32, L/4) -> (B, 64, L/8)
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.Tanh()
            )
            self.input_len = input_len
            
        def forward(self, x):  # Input shape: (Batch, Length)
            # Add a channel dimension for Conv1D
            x = x.unsqueeze(1) # (B, 1, L)
            
            z = self.encoder(x)
            recon = self.decoder(z)
            
            # Remove channel dimension
            recon = recon.squeeze(1) # (B, L)

            # Ensure output length matches input, cropping if necessary
            if recon.size(1) > self.input_len:
                recon = recon[:, : self.input_len]
            return recon

    spectro_model = SpectroAutoEncoder().to(device)

    loss_fn = nn.MSELoss()

    optim_time = torch.optim.Adam(time_model.parameters(), lr=LEARNING_RATE)
    optim_spectro = torch.optim.Adam(spectro_model.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting autoencoder training...")
    best_val_loss = float('inf')
    best_model_path = None
    epochs_no_improve = 0
    
    # --- 4. Training Loop ---
    def train_one(model, optim, train_loader, val_loader, tag):
        best_val = float('inf'); best_path=None
        for epoch in range(EPOCHS):
            model.train(); tot=0.0
            for xb in train_loader:
                xb = xb.to(device, non_blocking=True)
                recon = model(xb)
                loss = loss_fn(recon, xb)
                optim.zero_grad(); loss.backward(); optim.step(); tot += loss.item()
            avg_train = tot/len(train_loader)

            model.eval(); tot_v=0.0
            with torch.no_grad():
                for xb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    val_loss = loss_fn(model(xb), xb)
                    tot_v += val_loss.item()
            avg_val = tot_v/len(val_loader)
            print(f"[{tag}] Epoch {epoch+1}/{EPOCHS} | Train {avg_train:.6f} | Val {avg_val:.6f}")

            if avg_val < best_val:
                best_val = avg_val
                # delete old
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                best_path = str(MODELS_DIR / f"autoencoder_{tag}_best_{ts}.pt")
                torch.save(model.state_dict(), best_path)
                print(f"   ✨ saved {best_path}")

    # create models dir
    from pathlib import Path
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    train_one(time_model, optim_time, time_loader, time_val_loader, tag="time")
    train_one(spectro_model, optim_spectro, spectro_loader, spectro_val_loader, tag="spectro")
    print("🏁 Training complete for both models.")

if __name__ == "__main__":
    train() 