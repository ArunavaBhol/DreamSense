import pytest
import numpy as np
from src.feature_extractor import EEGFeatureExtractor

def test_compute_fft_bands_shapes():
    extractor = EEGFeatureExtractor(sampling_rate=200)
    # Generate 1 second of dummy noise data
    dummy_signal = np.random.normal(0, 1, 200)
    
    features = extractor.compute_fft_bands(dummy_signal)
    
    assert "band_power_alpha" in features
    assert "band_power_beta" in features
    assert "band_power_theta" in features
    assert isinstance(features["band_power_alpha"], float)

def test_empty_signal_throws_error():
    extractor = EEGFeatureExtractor()
    with pytest.raises(ValueError):
        extractor.compute_fft_bands(np.array([]))
