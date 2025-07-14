use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tch::{kind::Kind, CModule, Tensor};
use tch::IValue;

use realfft::RealFftPlanner;
use realfft::num_complex::Complex32;

// --- Constants (must match python/training.py) ---
const WINDOW_SEC: f32 = 1.0;
const SAMPLE_RATE: f32 = 200_000.0;
const DOWNSAMPLE_FACTOR: usize = 1;
const WINDOW_SAMPLES: usize = (WINDOW_SEC * SAMPLE_RATE) as usize; // 200,000 raw samples
const DS_LEN: usize = WINDOW_SAMPLES / DOWNSAMPLE_FACTOR; // 200,000 downsampled

pub struct InferenceEngine {
    time_model: CModule,
    spectro_model: CModule,
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
        let spectro_path = find_latest_ts("autoencoder_spectro_best_*.ts").ok_or("No spectro .ts found")?;
        Self::new(&time_path, &spectro_path).map(|e| (e, time_path))
    }
    
    // Helper to compute spectrogram features matching the training pipeline
    fn compute_spectro_features(&self, window: &[f32]) -> Vec<f32> {
        const NFFT: usize = 131_072;
        const FFT_LOW_HZ: f32 = 5_000.0;
        const FFT_HIGH_HZ: f32 = 60_000.0;

        let freq_resolution = SAMPLE_RATE / NFFT as f32;
        let low_bin = (FFT_LOW_HZ / freq_resolution).floor() as usize;
        let high_bin = (FFT_HIGH_HZ / freq_resolution).floor() as usize;
        let num_bins = high_bin - low_bin;

        // Prepare input of length NFFT (take latest samples)
        let mut buf = vec![0.0f32; NFFT];
        let start = window.len().saturating_sub(NFFT);
        buf.copy_from_slice(&window[start..start + NFFT]);

        // FFT
        let mut planner = RealFftPlanner::<f32>::new();
        let rfft = planner.plan_fft_forward(NFFT);
        let mut spectrum: Vec<Complex32> = rfft.make_output_vec();
        rfft.process(&mut buf, &mut spectrum).unwrap();

        // Magnitude and slice band
        let mut mags: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();
        mags.truncate(high_bin + 1);
        mags.drain(0..low_bin);

        // Ensure exact length
        if mags.len() > num_bins {
            mags.truncate(num_bins);
        } else if mags.len() < num_bins {
            mags.extend(std::iter::repeat(0.0).take(num_bins - mags.len()));
        }

        // Normalize to [-1,1]
        if let Some(maxv) = mags.iter().cloned().fold(None, |acc, v| {
            Some(acc.map_or(v, |m: f32| m.max(v)))
        }) {
            if maxv > 0.0 {
                for v in mags.iter_mut() { *v /= maxv; }
            }
        }

        mags
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

        let recon_spec: Tensor = self
            .spectro_model
            .forward_is(&[IValue::from(input_spectro.copy())])?
            .try_into()?;

        let diff_spec = recon_spec - input_spectro;
        let mse_spec = diff_spec.pow_tensor_scalar(2.0).mean(Kind::Float).double_value(&[]);

        const THRESH_SPEC: f64 = 0.001;
        let conf_spec = (THRESH_SPEC / (THRESH_SPEC + mse_spec)).clamp(0.0, 1.0) as f32;

        let conf_ml = conf_spec; // rename for compatibility

        // Combined confidence (weighted average of time and ML)
        const TIME_WEIGHT: f32 = 0.5;
        const SPEC_WEIGHT: f32 = 0.5;
        let conf_combined = TIME_WEIGHT * conf_time + SPEC_WEIGHT * conf_ml;

        println!("[Inf] mse_t={:.3e} | conf_t={:.2} conf_s={:.2} conf_c={:.2}",
            mse_time, conf_time, conf_ml, conf_combined);
        
        // Return (combined, time, spectro)
        Ok((conf_combined, conf_time, conf_ml))
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