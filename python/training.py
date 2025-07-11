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
NUM_BINS = 200   # Number of frequency bins to extract

# GPU-friendly hyperparameters
EPOCHS = 20
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

        # Bin the frequencies
        bin_size = len(band) // NUM_BINS
        fft_bins = [np.mean(band[i*bin_size:(i+1)*bin_size]) for i in range(NUM_BINS)]

        # Concatenate downsampled time data + FFT bins
        combined = np.concatenate([ds, fft_bins])
        features.append(combined)

    X = np.vstack(features)
    print(f"✅ Final augmented dataset shape: {X.shape}")  # Should be (N, 200000 + 56)
    return X

class VoltageDataset(Dataset):
    """PyTorch dataset for voltage window vectors kept on CPU; each batch is moved to GPU on-the-fly."""
    def __init__(self, X: np.ndarray):
        # Keep tensor on CPU to avoid huge GPU allocations
        self.X = torch.from_numpy(X).float()  # (N, 200056)

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

    # Normalize data to [-1, 1] range, suitable for Tanh activation
    X /= np.max(np.abs(X), axis=1, keepdims=True)

    full_dataset = VoltageDataset(X)
    
    # Split dataset into training and validation sets
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f" splitting into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    # --- 3. Initialize Model and Optimizer ---
    model = ConvAutoEncoder1D(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("🚀 Starting autoencoder training...")
    best_val_loss = float('inf')
    best_model_path = None
    epochs_no_improve = 0
    
    # --- 4. Training Loop ---
    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        total_train_loss = 0.0
        
        for i, xb in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb_val in val_loader:
                xb_val = xb_val.to(device, non_blocking=True)
                recon_val = model(xb_val)
                val_loss = loss_fn(recon_val, xb_val)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"✅ Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- Early Stopping and Model Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset counter
            # Remove previous best if it exists
            if best_model_path and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except OSError as e:
                    print(f"⚠️ Could not delete old model {best_model_path}: {e}")

            os.makedirs("models", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            best_model_path = f"models/autoencoder_1d_best_{ts}.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✨ New best model saved to {best_model_path} (Val Loss: {avg_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"   🛑 Early stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
                break # Exit training loop

    print(f"🏁 Training complete. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train() 