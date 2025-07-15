use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tch::{kind::Kind, CModule, Tensor};
use tch::IValue;

use realfft::RealFftPlanner;
use realfft::num_complex::Complex32;

// --- Spectrogram bucketing constants (must match python/training.py) ---
const FFT_LOW_HZ_USIZE: usize = 200;   // 200 Hz lower bound
const FFT_HIGH_HZ_USIZE: usize = 100_000; // 100 kHz upper bound
const BUCKET_HZ: usize = 25;           // bucket width
const NUM_BINS_SPECTRO: usize = (FFT_HIGH_HZ_USIZE - FFT_LOW_HZ_USIZE) / BUCKET_HZ; // 3_992
const UPWEIGHT_BINS: usize = (5_000 - FFT_LOW_HZ_USIZE) / BUCKET_HZ; // first 192 buckets

// --- Constants (must match python/training.py) ---
const WINDOW_SEC: f32 = 1.0;
const SAMPLE_RATE: f32 = 200_000.0;
const DOWNSAMPLE_FACTOR: usize = 1;
const WINDOW_SAMPLES: usize = (WINDOW_SEC * SAMPLE_RATE) as usize; // 200,000 raw samples
const DS_LEN: usize = WINDOW_SAMPLES / DOWNSAMPLE_FACTOR; // 200,000 downsampled

pub struct InferenceEngine {
    time_model: CModule,
    spectro_model: CModule, // Deep SVDD wrapper
}

impl InferenceEngine {
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>>(time_path: P, spectro_path: Q) -> Result<Self, Box<dyn Error>> {
        let time_model = CModule::load(time_path.as_ref())?;
        println!("[Inference] Loaded time model {}", time_path.as_ref().display());

        let spectro_model = CModule::load(spectro_path.as_ref())?;
        println!("[Inference] Loaded spectro model {}", spectro_path.as_ref().display());

        Ok(Self { 
            time_model,
            spectro_model,
        })
    }

    pub fn load_latest() -> Result<(Self, PathBuf), Box<dyn Error>> {
        let time_path = find_latest_ts("autoencoder_time_best_*.ts").ok_or("No time .ts found")?;
        let spectro_path = find_latest_ts("autoencoder_spectro_best_*.ts").ok_or("No spectro AE .ts found")?;
        Self::new(&time_path, &spectro_path).map(|e| (e, time_path))
    }
    
    // Helper to compute spectrogram features matching the training pipeline
    fn compute_spectro_features(&self, window: &[f32]) -> Vec<f32> {
        // 1-second FFT at 1 Hz resolution, then average 25 Hz buckets
        const NFFT: usize = WINDOW_SAMPLES; // 200 000

        // Prepare buffer (latest 1 s) and zero-pad/trim to NFFT
        let mut buf = vec![0.0f32; NFFT];
        let start = window.len().saturating_sub(NFFT);
        buf.copy_from_slice(&window[start..start + NFFT]);

        // FFT
        let mut planner = RealFftPlanner::<f32>::new();
        let rfft = planner.plan_fft_forward(NFFT);
        let mut spectrum: Vec<Complex32> = rfft.make_output_vec();
        rfft.process(&mut buf, &mut spectrum).unwrap();

        let mags: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect(); // len 100 001

        // Bucket-average every 25 Hz from 200 Hz → 100 kHz
        let mut features: Vec<f32> = Vec::with_capacity(NUM_BINS_SPECTRO);
        for i in 0..NUM_BINS_SPECTRO {
            let start_hz = FFT_LOW_HZ_USIZE + i * BUCKET_HZ;
            let end_hz = start_hz + BUCKET_HZ;
            let slice = &mags[start_hz..end_hz.min(mags.len())];
            let mean = if !slice.is_empty() {
                slice.iter().sum::<f32>() / slice.len() as f32
            } else { 0.0 };
            features.push(mean);
        }

        // Normalize to [0,1]
        let maxv = features
            .iter()
            .cloned()
            .fold(0.0_f32, |m, v| m.max(v));
        if maxv > 0.0 {
            for v in features.iter_mut() {
                *v /= maxv;
            }
        }

        features
    }

    pub fn predict(&self, window: &[f32]) -> Result<(f32, f32, f32), Box<dyn Error>> {
        if window.len() < WINDOW_SAMPLES {
            return Err(format!("Window too short: got {}, expected {}", window.len(), WINDOW_SAMPLES).into());
        }

        // Time-domain inference
        let mut input_vec = downsample(&window[window.len() - WINDOW_SAMPLES..]);
        
        // Normalize time data
        let max_time = input_vec.iter().fold(0.0f32, |max, &val| val.abs().max(max));
        if max_time > 0.0 {
            for val in input_vec.iter_mut() {
                *val /= max_time;
            }
        }

        // Create input tensor for time model
        let input_time = Tensor::from_slice(&input_vec).to_kind(Kind::Float).unsqueeze(0);

        // Run time model
        let recon_time: Tensor = self
            .time_model
            .forward_is(&[IValue::from(input_time.copy())])?
            .try_into()?;

        let diff_time = recon_time - input_time;
        let mse_time = diff_time.pow_tensor_scalar(2.0).mean(Kind::Float).double_value(&[]);

        // Time confidence
        const THRESH_TIME: f64 = 0.001;
        let conf_time = (THRESH_TIME / (THRESH_TIME + mse_time)).clamp(0.0, 1.0) as f32;

        // Spectrogram inference using local auto-encoder
        let spectro_features = self.compute_spectro_features(window);
        let input_spectro = Tensor::from_slice(&spectro_features).to_kind(Kind::Float).unsqueeze(0);

        // Spectro auto-encoder reconstruction
        let recon_spec: Tensor = self
            .spectro_model
            .forward_is(&[IValue::from(input_spectro.copy())])?
            .try_into()?;

        let diff_spec = recon_spec - input_spectro;

        // Weighted MSE
        let mut weights = vec![1.0f32; NUM_BINS_SPECTRO];
        for w in &mut weights[..UPWEIGHT_BINS] { *w = 5.0; }
        let w_tensor = Tensor::from_slice(&weights).to_kind(Kind::Float).unsqueeze(0);

        let mse_spec = (diff_spec.pow_tensor_scalar(2.0) * &w_tensor)
            .sum(Kind::Float)
            .double_value(&[]) / w_tensor.sum(Kind::Float).double_value(&[]);

        const THRESH_SPEC: f64 = 0.001;
        let conf_spec = (THRESH_SPEC / (THRESH_SPEC + mse_spec)).clamp(0.0, 1.0) as f32;

        // Combined confidence
        const TIME_WEIGHT: f32 = 0.5;
        const SPEC_WEIGHT: f32 = 0.5;
        let conf_combined = TIME_WEIGHT * conf_time + SPEC_WEIGHT * conf_spec;

        println!("[Inf] mse_t={:.3e} mse_s={:.3e} | conf_t={:.2} conf_spec={:.2} conf_c={:.2}",
            mse_time, mse_spec, conf_time, conf_spec, conf_combined);
        
        Ok((conf_combined, conf_time, conf_spec))
    }
}

fn find_latest_ts(pattern: &str) -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &["models", "../models"];

    for dir in CANDIDATES {
        if let Ok(iter) = fs::read_dir(dir) {
            let latest = iter
                .filter_map(Result::ok)
                .filter(|e| {
                    e.path().extension().map_or(false, |ext| ext == "ts") &&
                    e.file_name().to_string_lossy().starts_with(pattern.trim_end_matches("*.ts"))
                })
                .max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));
            
            if let Some(best) = latest {
                return Some(best.path());
            }
        }
    }
    None
}

/// Downsample the raw signal by taking every Nth sample.
fn downsample(raw: &[f32]) -> Vec<f32> {
    raw.iter()
        .step_by(DOWNSAMPLE_FACTOR)
        .cloned()
        .take(DS_LEN)
        .collect()
} 