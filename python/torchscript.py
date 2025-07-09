import glob
import os
import re
import sys
from datetime import datetime

import torch

# AutoEncoder definition must match training; keep a minimal copy here.
class AutoEncoder(torch.nn.Module):
    def __init__(self, in_len: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
        self.in_len = in_len

    def forward(self, x):
        z = self.encoder(x.unsqueeze(1))
        recon = self.decoder(z).squeeze(1)
        if recon.size(1) > self.in_len:
            recon = recon[:, : self.in_len]
        elif recon.size(1) < self.in_len:
            recon = torch.nn.functional.pad(recon, (0, self.in_len - recon.size(1)))
        return recon


def _find_latest_pt(models_dir: str = "models") -> str | None:
    pts = sorted(
        glob.glob(os.path.join(models_dir, "model_*.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    return pts[0] if pts else None


def main() -> None:
    latest_pt = _find_latest_pt()
    if latest_pt is None:
        print("[torchscript] No model_*.pt found in models/.", file=sys.stderr)
        sys.exit(1)

    IN_LEN = 200_056  # Length used during training (200k + 56 FFT bins)

    model = AutoEncoder(IN_LEN)
    state = torch.load(latest_pt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scripted = torch.jit.script(model)
    ts_path = re.sub(r"\.pt$", ".ts", latest_pt)
    scripted.save(ts_path)
    print(f"[torchscript] Saved TorchScript model to {ts_path}")


if __name__ == "__main__":
    main() 