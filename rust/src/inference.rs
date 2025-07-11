use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use tch::{kind::Kind, CModule, Tensor};
use tch::IValue;

// --- Constants (must match python/training.py) ---
const WINDOW_SEC: f32 = 1.0;
const SAMPLE_RATE: f32 = 200_000.0;
const DOWNSAMPLE_FACTOR: usize = 1;
const WINDOW_SAMPLES: usize = (WINDOW_SEC * SAMPLE_RATE) as usize; // 200,000 raw samples
const DS_LEN: usize = WINDOW_SAMPLES / DOWNSAMPLE_FACTOR; // 200,000 downsampled
// FFT constants
const NFFT: usize = 131_072;
const FFT_LOW_HZ: f32 = 5_000.0;
const FFT_HIGH_HZ: f32 = 60_000.0;
const NUM_BINS: usize = 200;

pub struct InferenceEngine {
    model: CModule,
}

impl InferenceEngine {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn Error>> {
        let model = CModule::load(model_path.as_ref())?;
        println!("[Inference] Loaded TorchScript model from {}", model_path.as_ref().display());
        Ok(Self { model })
    }

    pub fn load_latest() -> Result<(Self, PathBuf), Box<dyn Error>> {
        let path = find_latest_model().ok_or("No autoencoder_1d_best*.ts model found")?;
        let engine = Self::new(path.clone())?;
        Ok((engine, path))
    }

    pub fn predict(&self, window: &[f32]) -> Result<f32, Box<dyn Error>> {
        if window.len() < WINDOW_SAMPLES {
            return Err(format!("Window too short: got {}, expected {}", window.len(), WINDOW_SAMPLES).into());
        }

        // Prepare input tensor exactly as in Python
        let mut input_vec = downsample(&window[window.len() - WINDOW_SAMPLES..]);
        let fft_bins = compute_fft_bins(&input_vec);
        input_vec.extend(fft_bins);
        normalize(&mut input_vec);
        
        let input_tensor = Tensor::from_slice(&input_vec)
            .to_kind(Kind::Float)
            .unsqueeze(0); // Shape: [1, 200056]

        // Run inference
        let out = self.model.forward_is(&[IValue::from(input_tensor.copy())])?;
        let recon: Tensor = out.try_into()?;
        
        // Calculate Mean Squared Error for reconstruction
        let diff = recon - &input_tensor;
        // Split diff into time-domain part and FFT-bin part
        let diff_time = diff.narrow(1, 0, DS_LEN as i64);
        let diff_fft  = diff.narrow(1, DS_LEN as i64, NUM_BINS as i64);

        let mse_time = diff_time.pow_tensor_scalar(2.0).mean(Kind::Float).double_value(&[]);
        let mse_fft  = diff_fft.pow_tensor_scalar(2.0).mean(Kind::Float).double_value(&[]);

        // Give FFT error a higher weight to make tonal anomalies more visible
        const FFT_WEIGHT: f64 = 5.0;  // tune as needed
        let mse = mse_time + FFT_WEIGHT * mse_fft;

        // Debug log
        println!("[Inference] mse_time={:.6e} mse_fft={:.6e} (weighted) mse_total={:.6e}", mse_time, mse_fft, mse);
        // Simple threshold mapping to get a "confidence" score (0=anomaly, 1=normal)
        // This threshold might need retuning now that we weight FFT heavily.
        const THRESH: f64 = 0.01; 
        let conf = (THRESH / (THRESH + mse)).clamp(0.0, 1.0) as f32;
        
        println!("[Inference] Recon MSE: {:.6} | Confidence: {:.3}", mse, conf);
        Ok(conf)
    }
}

fn find_latest_model() -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &["models", "../models"];

    for dir in CANDIDATES {
        if let Ok(iter) = fs::read_dir(dir) {
            let latest = iter
                .filter_map(Result::ok)
                .filter(|e| {
                    e.path().extension().map_or(false, |ext| ext == "ts") &&
                    e.file_name().to_string_lossy().starts_with("autoencoder_1d")
                })
                .max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));
            
            if let Some(best) = latest {
                return Some(best.path());
            }
        }
    }
    None
}

/// Compute FFT bins matching Python
fn compute_fft_bins(raw: &[f32]) -> Vec<f32> {
    use realfft::RealFftPlanner;
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(NFFT);
    let mut input = vec![0.0f32; NFFT];
    let copy_len = raw.len().min(NFFT);
    input[0..copy_len].copy_from_slice(&raw[0..copy_len]);
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).unwrap();

    let bin_freq = SAMPLE_RATE / NFFT as f32;
    let mut band: Vec<f32> = spectrum.iter().enumerate().filter_map(|(i, c)| {
        let freq = i as f32 * bin_freq;
        if freq >= FFT_LOW_HZ && freq <= FFT_HIGH_HZ {
            Some((c.re * c.re + c.im * c.im).sqrt())
        } else {
            None
        }
    }).collect();

    let bin_size = band.len() / NUM_BINS;
    let trim_len = bin_size * NUM_BINS;
    band.truncate(trim_len);

    let mut fft_bins = Vec::with_capacity(NUM_BINS);
    for i in 0..NUM_BINS {
        let start = i * bin_size;
        let end = start + bin_size;
        let avg = band[start..end].iter().sum::<f32>() / bin_size as f32;
        fft_bins.push(avg);
    }
    fft_bins
}

/// Downsample the raw signal by taking every Nth sample.
/// This matches the Python `decimate` function's behavior for this use case.
fn downsample(raw: &[f32]) -> Vec<f32> {
    raw.iter()
        .step_by(DOWNSAMPLE_FACTOR)
        .cloned()
        .take(DS_LEN)
        .collect()
}

/// Normalize the data to the [-1, 1] range, matching the Python script.
fn normalize(data: &mut [f32]) {
    let max_abs = data.iter().fold(0.0f32, |max, &val| val.abs().max(max));
    if max_abs > 0.0 {
        for val in data.iter_mut() {
            *val /= max_abs;
        }
    }
} 