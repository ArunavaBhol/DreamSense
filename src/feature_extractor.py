import numpy as np
from typing import Dict

class EEGFeatureExtractor:
    """Handles feature engineering and digital signal processing on raw EEG signals."""
    
    def __init__(self, sampling_rate: int = 256):
        self.sampling_rate = sampling_rate

    def compute_fft_bands(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Computes the absolute power spectral density across standard brainwave bands.
        Alpha: 8-12 Hz | Beta: 12-30 Hz | Theta: 4-8 Hz
        """
        if len(signal) == 0:
            raise ValueError("Signal chunk cannot be empty.")
            
        fft_values = np.abs(np.fft.fft(signal))
        frequencies = np.fft.fftfreq(len(signal), d=1.0/self.sampling_rate)
        
        # Take positive frequencies
        positive_idx = frequencies >= 0
        freqs = frequencies[positive_idx]
        fft_pow = fft_values[positive_idx] ** 2

        bands = {
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30)
        }
        
        extracted_features = {}
        for band, (low, high) in bands.items():
            # Find indices mapping to specific bands
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            extracted_features[f"band_power_{band}"] = float(np.sum(fft_pow[idx])) if len(idx) > 0 else 0.0
            
        return extracted_features
