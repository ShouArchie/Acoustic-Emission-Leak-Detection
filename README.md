# Piezo Monitor

**Real-time high-speed piezoelectric sensor visualizer**

This application acquires 200 kHz voltage data from a Raspberry Pi Pico over USB, stores the stream in a lock-free circular buffer, and displays both the live waveform and its FFT using a Rust egui/eframe desktop GUI. Designed for leak detection within piping systems.

---

## Features

* 🛰️  200 kHz continuous sampling (2 000 samples per batch, 0.01 s intervals). Adjustable based on max USB sample rate
* 🔵 Live time-domain plot with automatic down-sampling
* 🔴 Frequency-domain plot (FFT, 10–100 kHz window) updated in real-time
* 🛡️  Circular buffer keeps only the most recent window (10 s / 1 000 000 samples)

---

## Hardware

| Part | Notes |
|------|-------|
| **Raspberry Pi Pico** | Firmware in `pico_adc_sampler.cpp` (200 kHz ADC + DMA) |
| **Piezo sensor** | Connected to ADC0 (GPIO 26) |
| **USB Cable** | High-speed; attach Pico to host PC |
| **LM386** | Opamp used for piezo |

> 💡 **Windows drivers**: both Pico and Teensy appear as standard USB CDC/Serial devices; no extra driver is usually required on Win 10+.

---

## Prerequisites

1. **Rust toolchain** (stable ‑ 1.70+ recommended)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update
   ```
2. **Git** (to clone the project)
3. **C/C++ compiler/builder** (only needed if you plan to rebuild the Pico firmware)

---

## Building & Running (desktop app)

```bash
# clone
$ git clone https://github.com/your-org/piezo-monitor.git
$ cd piezo-monitor

# build optimized release binary
$ cargo build --release

# run (the binary will be in target/release/)
$ cargo run --release
```

If everything is wired correctly, click **🔌 Connect** inside the GUI. The program auto-detects the first Pico/Teensy serial port (VID 0x2E8A / 0x16C0). Press **▶️ Start** to begin streaming.

### Development build

For faster incremental builds while hacking on the GUI:
```bash
cargo run
```

---

## Firmware (Pico)

The C++ firmware in `pico_adc_sampler.cpp` configures the RP2040 ADC + DMA for 200 kHz sampling and sends framed batches (header `0xA5 0x5A`, 32-bit batch-id + 2 000 × u16 samples).

To compile & flash (using the Pico SDK and `picotool`):
```bash
mkdir build && cd build
cmake ..
make -j4
picotool load -v pico_adc_sampler.uf2
```
(see RP2040 documentation for full environment setup).

---

## Usage tips

* **Down-sample** – use the spinner to manually reduce points if your GPU struggles.
* **Auto Scale** – toggles y-axis auto-bounds.
* **Show FFT** – hides the frequency plot when you only care about waveforms.
* If the status light shows **🔴 STALE**, your MCU isn’t sending data or the USB link is saturated.

---

## Troubleshooting

| Problem | Remedy |
|---------|--------|
| *No ports detected* | Ensure Pico is in normal run mode (not BOOTSEL) and shows up in *Device Manager* as USB Serial and not seen via file explorer. |
| *Desync detected* messages | Check cable quality; make sure firmware and desktop packet sizes match. |
| GUI freezes | Lower sample rate or down-sample factor. |

---

## Roadmap / TODO

* 🧠 Integrate 1-D CNN leak-detection model
* 🔁 Add RNN for predictive maintenance
* 🌐 Optional Web-view build via `eframe::Web` (WASM)

---

## License

MIT © 2025 Archie Shou / Aalo Atomics 