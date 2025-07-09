use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use realfft::RealFftPlanner;
use tch::{kind::Kind, CModule, Tensor};
use tch::IValue;

const WINDOW_SAMPLES: usize = 1_000_000;
const DOWNSAMPLE_FACTOR: usize = 5;
const DS_LEN: usize = WINDOW_SAMPLES / DOWNSAMPLE_FACTOR; // 200 000
const NFFT: usize = 131_072;
const FFT_LOW_HZ: f32 = 5_000.0;
const FFT_HIGH_HZ: f32 = 60_000.0;
const SAMPLE_RATE: f32 = 200_000.0;
const NUM_BINS: usize = 56;

pub struct InferenceEngine {
    model: CModule,
}

impl InferenceEngine {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn Error>> {
        let model = CModule::load(model_path.as_ref())?;
        println!("[Inference] Loaded TorchScript model from {}", model_path.as_ref().display());
        Ok(Self { model })
    }

    pub fn load_latest() -> Result<Self, Box<dyn Error>> {
        let path = find_latest_model().ok_or("No TorchScript .ts model found")?;
        Self::new(path)
    }

    pub fn predict(&self, window: &[f32]) -> Result<f32, Box<dyn Error>> {
        if window.len() < WINDOW_SAMPLES {
            return Err("window too short".into());
        }

        let features = compute_features(&window[window.len() - WINDOW_SAMPLES..]);
        let input = Tensor::from_slice(&features)
            .to_kind(Kind::Float)
            .unsqueeze(0); // (1, L)
        let out = self.model.forward_is(&[IValue::from(input.copy())])?;
        let recon: Tensor = out.try_into()?;
        // MSE reconstruction error
        let diff = recon - &input;
        let mse = diff.pow_tensor_scalar(2.0).mean(Kind::Float).double_value(&[]);

        // Simple threshold mapping → confidence (0..1)
        const THRESH: f64 = 0.01; // tune later
        let conf = (THRESH / (THRESH + mse)).clamp(0.0, 1.0) as f32;
        println!("[Inference] recon_mse={:.6}  confidence={:.3}", mse, conf);
        Ok(conf)
    }
}

fn find_latest_model() -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &["models", "../models"];

    for dir in CANDIDATES {
        if let Ok(iter) = fs::read_dir(dir) {
            if let Some(best) = iter
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "ts")
                        .unwrap_or(false)
                })
                .max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
            {
                return Some(best.path());
            }
        }
    }
    None
}

fn compute_features(raw: &[f32]) -> Vec<f32> {
    // Downsample by decimation
    let mut v_ds = Vec::with_capacity(DS_LEN);
    for i in (0..raw.len()).step_by(DOWNSAMPLE_FACTOR) {
        if v_ds.len() == DS_LEN {
            break;
        }
        v_ds.push(raw[i]);
    }

    // FFT computation
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(NFFT);
    let mut input = vec![0.0f32; NFFT];
    input.copy_from_slice(&raw[..NFFT]);
    let mut spectrum = r2c.make_output_vec();
    let _ = r2c.process(&mut input, &mut spectrum);

    let bin_freq = SAMPLE_RATE / NFFT as f32;
    let mut band: Vec<f32> = spectrum
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let freq = i as f32 * bin_freq;
            if freq >= FFT_LOW_HZ && freq <= FFT_HIGH_HZ {
                Some((c.re * c.re + c.im * c.im).sqrt())
            } else {
                None
            }
        })
        .collect();

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

    let mut features = v_ds;
    features.extend(fft_bins);
    features
} 