import glob
import os
import re
import sys
from datetime import datetime

import torch
import torch.nn as nn

# --- Constants (must match training.py) ---
WINDOW_SEC = 1.0
SAMPLE_RATE = 200_000
DOWNSAMPLE_FACTOR = 1
DS_RATE = SAMPLE_RATE // DOWNSAMPLE_FACTOR
DS_LEN = int(WINDOW_SEC * DS_RATE) + 56  # 200,000 time + 56 FFT bins


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


def _find_latest_model(models_dirs=("models", "../models")) -> str | None:
    """Find most recently modified model file in candidate directories."""
    candidates = []
    for mdir in models_dirs:
        candidates.extend(glob.glob(os.path.join(mdir, "autoencoder_1d_*.pt")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    print("🔎 Searching for the latest trained model...")
    latest_pt = _find_latest_model()
    if latest_pt is None:
        print("❌ No 'autoencoder_1d*.pt' model found in models/ folder.", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Found latest model: {latest_pt}")

    model = ConvAutoEncoder1D(DS_LEN)
    
    print("⏳ Loading model state from disk...")
    # Load onto CPU explicitly, as the Rust environment might not have a GPU
    state = torch.load(latest_pt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("✅ Model state loaded successfully.")

    print("⏳ Compiling model to TorchScript...")
    scripted = torch.jit.script(model)
    
    # Create the .ts path from the .pt path
    ts_path = os.path.splitext(latest_pt)[0] + ".ts"
    scripted.save(ts_path)
    print(f"✅ Saved TorchScript model to {ts_path}")


if __name__ == "__main__":
    main() 