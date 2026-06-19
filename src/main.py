from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import os

from src.data_loader import EEGDataLoader
from src.classifier.py import EEGEmotionClassifier

app = FastAPI(title="DreamSense API", description="Production-ready EEG Emotion Classification Service", version="1.0.0")

# Lazy initialize our classifier service instance
DATA_PATH = os.getenv("DATASET_PATH", "emotions.csv")
classifier = EEGEmotionClassifier()

class InferenceRequest(BaseModel):
    features: List[float]

@app.on_event("startup")
def startup_event():
    """Pre-trains the model on deployment startup so it's ready for immediate use."""
    if os.path.exists(DATA_PATH):
        try:
            loader = EEGDataLoader(DATA_PATH)
            X, y = loader.split_features_target()
            classifier.train(X, y)
        except Exception as e:
            print(f"Warning: Failed to automatically train model on startup: {str(e)}")
    else:
        print(f"Dataset not found at {DATA_PATH}. Model must be trained via the API.")

@app.post("/train", tags=["Model Management"])
def train_model() -> Dict[str, Any]:
    """Triggers model training from the configured CSV path dataset."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail=f"Dataset file '{DATA_PATH}' not found.")
    try:
        loader = EEGDataLoader(DATA_PATH)
        X, y = loader.split_features_target()
        metrics = classifier.train(X, y)
        return {"status": "success", "metrics": {"accuracy": metrics["accuracy"]}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", tags=["Inference"])
def predict_emotion(request: InferenceRequest) -> Dict[str, str]:
    """Accepts features array and returns predicted emotional state label."""
    if not classifier.is_trained:
        raise HTTPException(status_code=400, detail="Model is currently untrained. Call /train first.")
    try:
        vector = np.array(request.features)
        prediction = classifier.predict(vector)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference processing failed: {str(e)}")
