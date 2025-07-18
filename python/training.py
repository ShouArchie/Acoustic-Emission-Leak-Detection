import os
import math
from typing import Tuple, List
# debug: using external postgres at 192.168.10.212
import numpy as np
import psycopg2
from scipy.signal import decimate
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import time
from collections import deque
import datetime

# Params
DSN = os.getenv("DATABASE_URL", "postgres://pico:pass@192.168.10.212:5432/piezo_data")
WINDOW_SEC = 1.0
SLIDE_SEC = 0.1                # Stride for data augmentation
SAMPLE_RATE = 200_000
DOWNSAMPLE_FACTOR = 1  # No downsampling for higher resolution

DS_RATE = SAMPLE_RATE // DOWNSAMPLE_FACTOR  # 200 kHz
DS_LEN = int(WINDOW_SEC * DS_RATE)      # 200,000 samples

# FFT Hyperparameters – 1 Hz resolution over the 1-second window
# Use NFFT = 200 000 so each FFT bin ≈ 1 Hz (SAMPLE_RATE / NFFT)
NFFT = 200_000
FFT_LOW_HZ = 200.0
FFT_HIGH_HZ = 100_000.0

# --- New spectral setup ---
# Average every 25 Hz bin from 200 Hz → 100 kHz
BUCKET_HZ = 25
NUM_BINS_SPECTRO = int((FFT_HIGH_HZ - FFT_LOW_HZ) / BUCKET_HZ)  # (100k-200)/25 = 3992
FINE_BINS_HIGH_HZ = 5_000.0   # Up-weighted range upper bound
NUM_UPWEIGHT = int((FINE_BINS_HIGH_HZ - FFT_LOW_HZ) / BUCKET_HZ)  # (5000-200)/25 = 192

# GPU-friendly hyperparameters
# --- Training schedule ---
# Train up to 50 epochs, with early stopping after 7 epochs without improvement.
EPOCHS = 50
BATCH_SIZE = 64 # Larger batch size for GPU
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

EARLY_STOP_PATIENCE = 7  # epochs without improvement
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# --- Training mode selector ---
# 0 = train both auto-encoders
# 1 = train time-domain auto-encoder only
# 2 = train FFT / spectrogram auto-encoder only
TRAIN_MODE = 0


def get_windows() -> List[np.ndarray]:
    """Fetch all 'voltages' arrays from the windows table."""
    print(f"[debug] attempting db connect via {DSN}")
    try:
        conn = psycopg2.connect(DSN)
        print("✅ connected to postgres host ✔")
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

    # 4. Compute FFT-based spectral features
    print("⏳ Computing FFT features…")
    features = []
    for ds in ds_windows:
        # FFT (1 Hz resolution) and magnitude
        fft_result = np.fft.rfft(ds, n=NFFT)
        mags = np.abs(fft_result)  # len 100 001

        # Bucket-average every 25 Hz from 200 Hz → 100 kHz
        buckets = []
        start_bin = int(FFT_LOW_HZ)  # 200
        for i in range(NUM_BINS_SPECTRO):
            bin_start = start_bin + i * BUCKET_HZ
            bin_end   = bin_start + BUCKET_HZ
            bucket_vals = mags[bin_start:bin_end]
            buckets.append(bucket_vals.mean() if bucket_vals.size else 0.0)

        spectro_bins = np.asarray(buckets, dtype=np.float32)

        # Sanity check / pad if rare rounding issues
        if spectro_bins.size < NUM_BINS_SPECTRO:
            spectro_bins = np.pad(spectro_bins, (0, NUM_BINS_SPECTRO - spectro_bins.size))

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

    # Split into train/val/test
    total = len(time_dataset)
    train_size = int(total * TRAIN_FRAC)
    val_size   = int(total * VAL_FRAC)
    test_size  = total - train_size - val_size

    time_train, time_val, time_test = random_split(time_dataset, [train_size, val_size, test_size])
    spectro_train, spectro_val, spectro_test = random_split(spectro_dataset, [train_size, val_size, test_size])

    print(f" Dataset split: {train_size} train | {val_size} val | {test_size} test")

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

    spectro_loader      = DataLoader(spectro_train, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    spectro_val_loader  = DataLoader(spectro_val,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    spectro_test_loader = DataLoader(spectro_test,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # --- 3. Initialize Models ---
    time_model    = ConvAutoEncoder1D(DS_LEN).to(device)
    spectro_model = ConvAutoEncoder1D(NUM_BINS_SPECTRO).to(device)

    optim_time    = torch.optim.AdamW(time_model.parameters(),    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optim_spectro = torch.optim.AdamW(spectro_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Shared loss
    loss_fn = nn.MSELoss(reduction="none")  # we'll apply weights manually

    # Precompute spectral weights (tensor on device)
    weights_np = np.ones(NUM_BINS_SPECTRO, dtype=np.float32)
    weights_np[:NUM_UPWEIGHT] = 5.0  # up-weight 200 Hz – 5 kHz region
    weights_tensor = torch.from_numpy(weights_np).to(device)

    # --- General training helper for an auto-encoder ---
    def train_one(model, optim, train_loader, val_loader, test_loader, tag: str):
        """Train a single auto-encoder with early stopping and TorchScript export."""
        best_val = float('inf')
        best_path: str | None = None
        epochs_no_improve = 0
        epoch = 0

        while epoch < EPOCHS:
            epoch += 1
            model.train(); total_train = 0.0
            for xb in train_loader:
                xb = xb.to(device, non_blocking=True)
                recon = model(xb)
                raw = loss_fn(recon, xb)  # shape (B, L)
                if tag == "spectro":
                    weighted = raw * weights_tensor  # broadcast (1,L)
                    loss = weighted.sum() / weights_tensor.sum() / raw.size(0)
                else:
                    loss = raw.mean()
                optim.zero_grad(); loss.backward(); optim.step()
                total_train += loss.item()
            avg_train = total_train / len(train_loader)

            # Validation
            model.eval(); total_val = 0.0
            with torch.no_grad():
                for xb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    recon_v = model(xb)
                    raw_v = loss_fn(recon_v, xb)
                    if tag == "spectro":
                        val_loss = (raw_v * weights_tensor).sum() / weights_tensor.sum()
                    else:
                        val_loss = raw_v.mean()
                    total_val += val_loss.item()
            avg_val = total_val / len(val_loader)
            print(f"[{tag}] Epoch {epoch} | Train {avg_train:.6f} | Val {avg_val:.6f}")

            # Check for improvement
            if avg_val < best_val - 1e-6:
                best_val = avg_val; epochs_no_improve = 0
                # Remove previous best
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                ts_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                best_path = str(MODELS_DIR / f"autoencoder_{tag}_best_{ts_stamp}.ts")
                scripted = torch.jit.script(model.cpu())
                scripted.save(best_path)
                model.to(device)
                print(f"   ✨ saved {best_path}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"   ⏹ Early stopping after {epoch} epochs (no val improvement)")
                break

        # Evaluate on test set
        model.eval(); total_test = 0.0
        with torch.no_grad():
            for xb in test_loader:
                xb = xb.to(device, non_blocking=True)
                recon_t = model(xb)
                raw_t = loss_fn(recon_t, xb)
                if tag == "spectro":
                    test_loss = (raw_t * weights_tensor).sum() / weights_tensor.sum()
                else:
                    test_loss = raw_t.mean()
                total_test += test_loss.item()
        print(f"[{tag}] Test MSE: {total_test / len(test_loader):.6f}")

    # create models dir
    from pathlib import Path
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    if TRAIN_MODE in (0, 1):
        train_one(
            time_model, optim_time,
            time_loader, time_val_loader,
            DataLoader(time_test, batch_size=BATCH_SIZE, shuffle=False),
            tag="time")
 
    # --------- Spectro auto-encoder training loop ---------
    if TRAIN_MODE in (0, 2):
        train_one(
            spectro_model, optim_spectro,
            spectro_loader, spectro_val_loader,
            spectro_test_loader,
            tag="spectro")

    print("🏁 Training complete for both models.")

if __name__ == "__main__":
    train() 