import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import glob

class MLSpectralInference:
    def __init__(self):
        self.isolation_forest_model = None
        self.one_class_svm_model = None
        self.models_loaded = False
        
    def find_latest_model(self, pattern: str) -> Optional[str]:
        """Find the latest model file matching the pattern."""
        models_dir = Path(__file__).resolve().parent.parent / "models"
        candidates = list(models_dir.glob(pattern))
        if not candidates:
            return None
        return str(max(candidates, key=lambda p: p.stat().st_mtime))
    
    def load_models(self):
        """Load both ML models."""
        print("🔄 Loading ML spectral models...", file=sys.stderr)
        
        # Load Isolation Forest
        if_path = self.find_latest_model("isolation_forest_spectro_*.pkl")
        if if_path:
            with open(if_path, 'rb') as f:
                self.isolation_forest_model = pickle.load(f)
            print(f"✅ Loaded Isolation Forest: {Path(if_path).name}", file=sys.stderr)
        else:
            print("❌ No Isolation Forest model found", file=sys.stderr)
        
        # Load One-Class SVM
        svm_path = self.find_latest_model("one_class_svm_spectro_*.pkl")
        if svm_path:
            with open(svm_path, 'rb') as f:
                self.one_class_svm_model = pickle.load(f)
            print(f"✅ Loaded One-Class SVM: {Path(svm_path).name}", file=sys.stderr)
        else:
            print("❌ No One-Class SVM model found", file=sys.stderr)
        
        self.models_loaded = (self.isolation_forest_model is not None or 
                             self.one_class_svm_model is not None)
        
        if self.models_loaded:
            print("✅ ML models ready for inference", file=sys.stderr)
        else:
            print("❌ No ML models loaded", file=sys.stderr)
    
    def compute_spectrum_features(self, signal: np.ndarray, model_data: dict) -> np.ndarray:
        """Compute spectral features matching the training preprocessing."""
        fft_params = model_data['fft_params']
        downsample_factor = model_data['downsample_factor']
        
        # Ensure signal length
        sample_rate = fft_params['sample_rate']
        if len(signal) > sample_rate:
            signal = signal[-sample_rate:]
        elif len(signal) < sample_rate:
            signal = np.pad(signal, (0, sample_rate - len(signal)), 'constant')
            
        # Compute FFT
        nfft = fft_params['nfft']
        fft_result = np.fft.rfft(signal, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        
        # Filter to frequency range
        low_hz = fft_params['low_hz']
        high_hz = fft_params['high_hz']
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        spectrum = np.abs(fft_result[mask])
        
        # Log transform and downsample
        log_spectrum = np.log10(spectrum + 1e-8)
        downsampled = log_spectrum[::downsample_factor]
        
        return downsampled
    
    def predict_isolation_forest(self, signal: np.ndarray) -> Tuple[float, dict]:
        """Run Isolation Forest prediction."""
        if self.isolation_forest_model is None:
            return 0.5, {"error": "Isolation Forest model not loaded"}
        
        try:
            # Extract features
            features = self.compute_spectrum_features(signal, self.isolation_forest_model)
            
            # Preprocess
            X = features.reshape(1, -1)
            X_scaled = self.isolation_forest_model['scaler'].transform(X)
            
            # Predict
            model = self.isolation_forest_model['model']
            anomaly_score = model.decision_function(X_scaled)[0]
            is_anomaly = model.predict(X_scaled)[0] == -1
            
            # Convert to confidence (0 = anomaly, 1 = normal)
            confidence = 1.0 / (1.0 + np.exp(-anomaly_score * 2))
            
            details = {
                "anomaly_score": float(anomaly_score),
                "is_anomaly": bool(is_anomaly),
                "method": "isolation_forest"
            }
            
            return float(confidence), details
            
        except Exception as e:
            return 0.5, {"error": f"Isolation Forest prediction failed: {str(e)}"}
    
    def predict_one_class_svm(self, signal: np.ndarray) -> Tuple[float, dict]:
        """Run One-Class SVM prediction."""
        if self.one_class_svm_model is None:
            return 0.5, {"error": "One-Class SVM model not loaded"}
        
        try:
            # Extract features
            features = self.compute_spectrum_features(signal, self.one_class_svm_model)
            
            # Preprocess
            X = features.reshape(1, -1)
            X_scaled = self.one_class_svm_model['scaler'].transform(X)
            X_pca = self.one_class_svm_model['pca'].transform(X_scaled)
            
            # Predict
            model = self.one_class_svm_model['model']
            decision_score = model.decision_function(X_pca)[0]
            is_anomaly = model.predict(X_pca)[0] == -1
            
            # Convert to confidence
            confidence = 1.0 / (1.0 + np.exp(-decision_score))
            
            details = {
                "decision_score": float(decision_score),
                "is_anomaly": bool(is_anomaly),
                "method": "one_class_svm"
            }
            
            return float(confidence), details
            
        except Exception as e:
            return 0.5, {"error": f"One-Class SVM prediction failed: {str(e)}"}
    
    def predict(self, signal: np.ndarray) -> Dict:
        """Run prediction with both models and return combined results."""
        if not self.models_loaded:
            return {
                "success": False,
                "error": "No models loaded",
                "confidence_combined": 0.5,
                "confidence_isolation_forest": 0.5,
                "confidence_one_class_svm": 0.5
            }
        
        results = {"success": True}
        
        # Run Isolation Forest
        if_conf, if_details = self.predict_isolation_forest(signal)
        results["confidence_isolation_forest"] = if_conf
        results["isolation_forest_details"] = if_details
        
        # Run One-Class SVM
        svm_conf, svm_details = self.predict_one_class_svm(signal)
        results["confidence_one_class_svm"] = svm_conf
        results["one_class_svm_details"] = svm_details
        
        # Combine confidences (simple average for now)
        confidences = []
        if self.isolation_forest_model is not None:
            confidences.append(if_conf)
        if self.one_class_svm_model is not None:
            confidences.append(svm_conf)
        
        combined_conf = np.mean(confidences) if confidences else 0.5
        results["confidence_combined"] = float(combined_conf)
        
        return results

def main():
    """Main function for command-line interface."""
    if len(sys.argv) != 2:
        print("Usage: python ml_inference_service.py <signal_file>", file=sys.stderr)
        sys.exit(1)
    
    signal_file = sys.argv[1]
    
    # Load signal (support both .npy and .csv formats)
    try:
        if signal_file.endswith('.npy'):
            signal = np.load(signal_file)
        else:
            # Assume CSV format
            signal = np.loadtxt(signal_file, delimiter=',')
    except Exception as e:
        result = {"success": False, "error": f"Failed to load signal: {str(e)}"}
        print(json.dumps(result))
        sys.exit(1)
    
    # Initialize and run inference
    inference = MLSpectralInference()
    inference.load_models()
    
    result = inference.predict(signal)
    print(json.dumps(result))

if __name__ == "__main__":
    main() 