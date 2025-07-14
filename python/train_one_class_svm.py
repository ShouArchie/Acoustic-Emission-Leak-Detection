import os
import numpy as np
import psycopg2
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# One-Class SVM parameters
NU = 0.05  # Expected fraction of anomalies (5%)
KERNEL = 'rbf'  # Radial basis function kernel
GAMMA = 'scale'  # Kernel coefficient
PCA_COMPONENTS = 100  # Reduce dimensionality
DOWNSAMPLE_FACTOR = 30  # Take every 30th frequency bin

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

def train_one_class_svm():
    print("🚀 Training One-Class SVM for spectral anomaly detection...")
    
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
    
    # Apply PCA for dimensionality reduction
    print("🔄 Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)
    print(f"📊 After PCA: {X_pca.shape}")
    print(f"📊 PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # Train One-Class SVM
    print("🔄 Training One-Class SVM...")
    model = OneClassSVM(
        nu=NU,
        kernel=KERNEL,
        gamma=GAMMA
    )
    model.fit(X_pca)
    
    # Test on training data
    scores = model.decision_function(X_pca)
    predictions = model.predict(X_pca)
    n_outliers = np.sum(predictions == -1)
    
    print(f"✅ Training complete!")
    print(f"📊 Decision scores - Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
    print(f"📊 Score range: {np.min(scores):.3f} to {np.max(scores):.3f}")
    print(f"📊 Outliers detected: {n_outliers}/{len(predictions)} ({n_outliers/len(predictions)*100:.1f}%)")
    
    # Save model
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_path = models_dir / f"one_class_svm_spectro_{ts}.pkl"
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'downsample_factor': DOWNSAMPLE_FACTOR,
        'pca_components': PCA_COMPONENTS,
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
            'max_score': np.max(scores),
            'explained_variance': np.sum(pca.explained_variance_ratio_)
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    train_one_class_svm() 