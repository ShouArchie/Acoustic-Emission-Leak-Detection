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
import datetime, os

DSN = os.getenv("DATABASE_URL", "postgres://pico:pass@localhost:5432/piezo_data")
WINDOW_SEC = 5.0
SAMPLE_RATE = 200_000          # Hz
SLIDE_SEC = 0.2                # stride
DOWNSAMPLE_FACTOR = 5          # 200 kHz → 40 kHz
FFT_LOW_HZ = 5_000
FFT_HIGH_HZ = 60_000
FFT_BIN_SPACING = 1_000        # 1 kHz

DS_RATE = SAMPLE_RATE // DOWNSAMPLE_FACTOR   # 40 kHz
DS_LEN  = int(WINDOW_SEC * DS_RATE)          # 200 000

EPOCHS = 15


def get_windows() -> List[np.ndarray]:
    """Fetch voltages arrays from the windows table."""
    conn = psycopg2.connect(DSN)
    print("Connecting to PostgreSQL …")
    cur = conn.cursor()
    cur.execute("SELECT voltages FROM windows ORDER BY id")
    rows = cur.fetchall()
    print(f"Fetched {len(rows)} base windows.")
    cur.close()
    conn.close()
    return [np.asarray(r[0], dtype=np.float32) for r in rows]


def iter_slices(window: np.ndarray) -> List[np.ndarray]:
    """Yield overlapped 5-s slices inside a 15-minute (or smaller) array."""
    stride = int(SLIDE_SEC * SAMPLE_RATE)      # 40 000 samples
    win_len = int(WINDOW_SEC * SAMPLE_RATE)    # 1 000 000 samples
    for start in range(0, len(window) - win_len + 1, stride):
        yield window[start:start + win_len]


def fft_feature(raw: np.ndarray) -> np.ndarray:
    # 131 072-pt FFT (next power-of-two ≥ 1e6) is heavy; instead use 65 536 (≈ 327 ms) tiles
    NFFT = 131_072
    fft = np.fft.rfft(raw, n=NFFT)
    freqs = np.fft.rfftfreq(NFFT, d=1 / SAMPLE_RATE)

    band_mask = (freqs >= FFT_LOW_HZ) & (freqs <= FFT_HIGH_HZ)
    band = np.abs(fft[band_mask])

    n_target = int((FFT_HIGH_HZ - FFT_LOW_HZ) / 1000) + 1  # 56
    bin_size = len(band) // n_target
    trim_len = bin_size * n_target
    band = band[:trim_len]
    mag = band.reshape(n_target, bin_size).mean(axis=1)
    return mag.astype(np.float32)


def make_dataset() -> Tuple[np.ndarray, np.ndarray]:
    rows = get_windows()  # 180 rows
    win_len = int(WINDOW_SEC * SAMPLE_RATE)
    stride = int(SLIDE_SEC * SAMPLE_RATE)

    buf = deque()  # keep growing; we manually pop stride samples
    features, labels = [], []

    total_rows = len(rows)
    processed_rows = 0

    for row in rows:
        # append new samples
        buf.extend(row)

        while len(buf) >= win_len:
            # take first win_len samples
            slice_arr = np.array(list(buf)[:win_len], dtype=np.float32)

            # generate feature
            v_ds = decimate(slice_arr, DOWNSAMPLE_FACTOR, ftype='fir')
            fft_mag = fft_feature(slice_arr)
            feat_vec = np.concatenate([v_ds, fft_mag])
            features.append(feat_vec)
            labels.append(1)

            # remove stride samples from left
            for _ in range(stride):
                buf.popleft()

        processed_rows += 1
        if processed_rows % 20 == 0 or processed_rows == total_rows:
            pct = processed_rows * 100 / total_rows
            print(f"Row progress: {pct:.1f}% ({processed_rows}/{total_rows})")

    X = np.vstack(features)
    y = np.array(labels, dtype=np.int64)
    print(f"Final dataset shapes  X:{X.shape}  y:{y.shape}")
    return X, y

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleCNN(nn.Module):
    def __init__(self, in_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # B,1,L
        return self.net(x)

def train():
    X, y = make_dataset()
    ds = WindowDataset(X, y)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = SimpleCNN(X.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    print("Starting training …")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            preds = model(xb.float()).squeeze()
            loss = loss_fn(preds, yb.float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        # simple val loss
        model.eval()
        with torch.no_grad():
            v_loss = 0.0
            for xb, yb in val_loader:
                preds = model(xb.float()).squeeze()
                v_loss += loss_fn(preds, yb.float()).item()
        print(f"Epoch {epoch+1}/{EPOCHS}: train_loss {avg_loss:.4f}  val_loss {v_loss/len(val_loader):.4f}")
    os.makedirs("models", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"models/model_{ts}.pt"
    torch.save(model.state_dict(), fname)
    print(f"Model saved to {fname}")

if __name__ == "__main__":
    train() 