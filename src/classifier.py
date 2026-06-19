from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EEGEmotionClassifier:
    """Manages the lifetime of the ML model used to classify brain emotional states."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Splits data, trains a Random Forest model, and returns performance metrics."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Training RandomForest Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        logger.info(f"Model trained successfully. Test Accuracy: {accuracy:.4f}")
        return {
            "accuracy": accuracy,
            "report": classification_report(y_test, predictions, output_dict=True)
        }

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predicts the emotion of a given processing feature row."""
        if not self.is_trained:
            raise RuntimeError("Cannot predict. Model has not been trained yet.")
            
        # Reshape for single sample inference
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)
            
        prediction = self.model.predict(feature_vector)
        return str(prediction[0])
