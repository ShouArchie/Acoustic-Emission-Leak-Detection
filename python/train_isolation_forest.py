import os
import numpy as np
import psycopg2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import List
import datetime

# --- Hyperparameters ---
DSN = os.getenv("DATABASE_URL", "postgres://pico:pass@localhost:5432/piezo_data")
WINDOW_SEC = 1.0
SAMPLE_RATE = 200_000
NFFT = 131_072
FFT_LOW_HZ = 5_000.0
FFT_HIGH_HZ = 60_000.0

# Isolation Forest parameters
CONTAMINATION = 0.05  # Expected fraction of anomalies (5%)
N_ESTIMATORS = 100
MAX_SAMPLES = 256
RANDOM_STATE = 42
DOWNSAMPLE_FACTOR = 50  # Take every 50th frequency bin for efficiency

def get_windows() -> List[np.ndarray]:
    """Fetch all normal windows from database."""
    try:
        conn = psycopg2.connect(DSN)
        print("✅ Connected to PostgreSQL.")
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to PostgreSQL: {e}")
        return []
        
    with conn.cursor() as cur:
        cur.execute("SELECT voltages FROM windows WHERE label_normal = TRUE ORDER BY id")
        rows = cur.fetchall()
    print(f"✅ Fetched {len(rows)} normal windows from the database.")
    conn.close()
    return [np.asarray(r[0], dtype=np.float32) for r in rows]

def compute_spectrum_features(signal: np.ndarray) -> np.ndarray:
    """Compute spectral features for a signal."""
    # Ensure signal length
    if len(signal) > SAMPLE_RATE:
        signal = signal[-SAMPLE_RATE:]
    elif len(signal) < SAMPLE_RATE:
        signal = np.pad(signal, (0, SAMPLE_RATE - len(signal)), 'constant')
        
    # Compute FFT
    fft_result = np.fft.rfft(signal, n=NFFT)
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / SAMPLE_RATE)
    
    # Filter to frequency range of interest
    mask = (freqs >= FFT_LOW_HZ) & (freqs <= FFT_HIGH_HZ)
    spectrum = np.abs(fft_result[mask])
    
    # Log transform and downsample
    log_spectrum = np.log10(spectrum + 1e-8)
    downsampled = log_spectrum[::DOWNSAMPLE_FACTOR]
    
    return downsampled

def train_isolation_forest():
    print("🚀 Training Isolation Forest for spectral anomaly detection...")
    
    # Get training data
    windows = get_windows()
    if not windows:
        print("❌ No training data found.")
        return
    
    print(f"📊 Processing {len(windows)} windows...")
    
    # Extract spectral features
    all_features = []
    for i, window in enumerate(windows):
        if i % 50 == 0:
            print(f"Processing window {i+1}/{len(windows)}")
            
        features = compute_spectrum_features(window)
        all_features.append(features)
    
    # Convert to numpy array
    X = np.array(all_features)
    print(f"📊 Feature matrix shape: {X.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    print("🔄 Training Isolation Forest...")
    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Test on training data
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)
    n_outliers = np.sum(predictions == -1)
    
    print(f"✅ Training complete!")
    print(f"📊 Anomaly scores - Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
    print(f"📊 Score range: {np.min(scores):.3f} to {np.max(scores):.3f}")
    print(f"📊 Outliers detected: {n_outliers}/{len(predictions)} ({n_outliers/len(predictions)*100:.1f}%)")
    
    # Save model
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_path = models_dir / f"isolation_forest_spectro_{ts}.pkl"
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'downsample_factor': DOWNSAMPLE_FACTOR,
        'fft_params': {
            'nfft': NFFT,
            'low_hz': FFT_LOW_HZ,
            'high_hz': FFT_HIGH_HZ,
            'sample_rate': SAMPLE_RATE
        },
        'training_stats': {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    train_isolation_forest() 