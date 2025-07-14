use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::io::Write;

use tch::{kind::Kind, CModule, Tensor};
use tch::IValue;
use serde_json::Value;

// --- Constants (must match python/training.py) ---
const WINDOW_SEC: f32 = 1.0;
const SAMPLE_RATE: f32 = 200_000.0;
const DOWNSAMPLE_FACTOR: usize = 1;
const WINDOW_SAMPLES: usize = (WINDOW_SEC * SAMPLE_RATE) as usize; // 200,000 raw samples
const DS_LEN: usize = WINDOW_SAMPLES / DOWNSAMPLE_FACTOR; // 200,000 downsampled

pub struct InferenceEngine {
    time_model: CModule,
    ml_service_available: bool,
}

impl InferenceEngine {
    pub fn new<P: AsRef<Path>>(time_path: P) -> Result<Self, Box<dyn Error>> {
        let time_model = CModule::load(time_path.as_ref())?;
        println!("[Inference] Loaded time model {}", time_path.as_ref().display());
        
        // Check if ML service is available
        let ml_available = Self::check_ml_service();
        if ml_available {
            println!("[Inference] ML spectral service available");
        } else {
            println!("[Inference] ML spectral service not available, using time-only inference");
        }
        
        Ok(Self { 
            time_model, 
            ml_service_available: ml_available 
        })
    }

    pub fn load_latest() -> Result<(Self, PathBuf), Box<dyn Error>> {
        let time_path = find_latest_ts("autoencoder_time_best_*.ts").ok_or("No time .ts found")?;
        Self::new(&time_path).map(|e| (e, time_path))
    }
    
    fn check_ml_service() -> bool {
        // Check if Python and required packages are available
        match Command::new("python")
            .arg("-c")
            .arg("import sklearn, numpy, pickle; print('OK')")
            .output() 
        {
            Ok(output) => {
                let result = String::from_utf8_lossy(&output.stdout);
                result.trim() == "OK"
            }
            Err(_) => false
        }
    }
    
    fn call_ml_service(&self, window: &[f32]) -> Result<(f32, f32, f32), Box<dyn Error>> {
        // Save signal to temporary file
        let temp_dir = std::env::temp_dir();
        let signal_file = temp_dir.join("signal_temp.npy");
        
        // Convert to numpy format and save
        // For simplicity, we'll write as CSV and read in Python
        let csv_file = temp_dir.join("signal_temp.csv");
        let mut file = std::fs::File::create(&csv_file)?;
        for (i, &value) in window.iter().enumerate() {
            if i > 0 { write!(file, ",")?; }
            write!(file, "{}", value)?;
        }
        writeln!(file)?;
        
        // Call Python ML service
        let python_script = Path::new("../python/ml_inference_service.py");
        let output = Command::new("python")
            .arg(python_script)
            .arg(&csv_file)
            .output()?;
        
        // Clean up temp file
        let _ = std::fs::remove_file(&csv_file);
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("ML service failed: {}", stderr).into());
        }
        
        // Parse JSON response
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: Value = serde_json::from_str(&stdout)?;
        
        if !result["success"].as_bool().unwrap_or(false) {
            let error = result["error"].as_str().unwrap_or("Unknown error");
            return Err(format!("ML prediction failed: {}", error).into());
        }
        
        let conf_combined = result["confidence_combined"].as_f64().unwrap_or(0.5) as f32;
        let conf_if = result["confidence_isolation_forest"].as_f64().unwrap_or(0.5) as f32;
        let conf_svm = result["confidence_one_class_svm"].as_f64().unwrap_or(0.5) as f32;
        
        Ok((conf_combined, conf_if, conf_svm))
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

        // ML spectral inference
        let (conf_ml, conf_if, conf_svm) = if self.ml_service_available {
            match self.call_ml_service(window) {
                Ok(result) => result,
                Err(e) => {
                    println!("[Inference] ML service error: {}", e);
                    (0.5, 0.5, 0.5) // Fallback values
                }
            }
        } else {
            (0.5, 0.5, 0.5) // No ML service available
        };

        // Combined confidence (weighted average of time and ML)
        const TIME_WEIGHT: f32 = 0.4;
        const ML_WEIGHT: f32 = 0.6;
        let conf_combined = TIME_WEIGHT * conf_time + ML_WEIGHT * conf_ml;

        println!("[Inf] mse_t={:.3e} | conf_t={:.2} conf_ml={:.2} (IF:{:.2} SVM:{:.2}) conf_c={:.2}",
            mse_time, conf_time, conf_ml, conf_if, conf_svm, conf_combined);
        
        // Return (combined, time, ml_combined)
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