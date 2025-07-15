import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# --- Constants (must match training.py) ---
WINDOW_SEC = 1.0
SAMPLE_RATE = 200_000
DOWNSAMPLE_FACTOR = 1
DS_RATE = SAMPLE_RATE // DOWNSAMPLE_FACTOR
# FFT constants
NFFT = 131_072
FFT_LOW_HZ = 5_000.0
FFT_HIGH_HZ = 60_000.0
# Calculate actual bins available in frequency range
freq_resolution = DS_RATE / NFFT
low_bin = int(FFT_LOW_HZ / freq_resolution)
high_bin = int(FFT_HIGH_HZ / freq_resolution)
NUM_BINS_SPECTRO = high_bin - low_bin  # Should be ~36,045 bins
DS_LEN = int(WINDOW_SEC * DS_RATE) + NUM_BINS_SPECTRO  # 200,000 time + ~36k spectro bins


# ----- model definitions -----
class ConvAutoEncoder1D(nn.Module):
    """
    A 1D convolutional auto-encoder.
    The encoder compresses the input vector, and the decoder attempts to reconstruct it.
    This definition MUST be kept in sync with training.py
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


class SpectroAutoEncoder(nn.Module):
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


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

def _find_latest(pattern: str) -> str | None:
    MODELS_DIR.mkdir(exist_ok=True)
    cands = list(MODELS_DIR.glob(pattern))
    if not cands:
        return None
    return str(max(cands, key=os.path.getmtime))


def convert(pt_path: str, model: nn.Module):
    print(f"⏳ Loading {pt_path} ...")
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    ts_path = os.path.splitext(pt_path)[0] + ".ts"
    torch.jit.script(model).save(ts_path)
    print(f"✅ Wrote {ts_path}")


def main():
    time_pt = _find_latest("autoencoder_time_best_*.pt")
    spectro_pt = _find_latest("dsvdd_spectro_best_*.pt")

    if not time_pt or not spectro_pt:
        print("❌ Could not find both time and spectro checkpoints", file=sys.stderr)
        sys.exit(1)

    convert(time_pt, ConvAutoEncoder1D(200_000))
    convert(spectro_pt, SpectroAutoEncoder(55_000))


if __name__ == "__main__":
    main() 