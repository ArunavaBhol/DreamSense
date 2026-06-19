import pytest
import numpy as np
from src.classifier import EEGEmotionClassifier

def test_untrained_prediction_raises_error():
    clf = EEGEmotionClassifier()
    dummy_input = np.array([0.5, 1.2, -0.4])
    with pytest.raises(RuntimeError):
        clf.predict(dummy_input)

def test_model_training_flow():
    clf = EEGEmotionClassifier()
    # Mock small dataset: 10 samples, 4 features
    X = np.random.rand(10, 4)
    y = np.array(["NEUTRAL", "POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", 
                  "NEGATIVE", "NEUTRAL", "POSITIVE", "NEGATIVE", "NEUTRAL"])
    
    metrics = clf.train(X, y)
    assert clf.is_trained is True
    assert "accuracy" in metrics
    
    # Test valid single prediction
    test_sample = np.random.rand(4)
    pred = clf.predict(test_sample)
    assert pred in ["NEUTRAL", "POSITIVE", "NEGATIVE"]
