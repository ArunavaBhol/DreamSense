# 🧠 DreamSense: EEG-Based Dream Emotion Analysis API

## 📌 Overview

DreamSense is a production-ready, AI-based backend system that analyzes EEG signals during sleep to interpret underlying emotional states. Transitioning from experimental notebooks to a scalable REST API architecture, it focuses on extracting DSP (Digital Signal Processing) patterns from brainwaves and serving emotion-based insights in real-time.

## 🎯 Objectives

* **Analyze EEG sleep data** robustly using modular data pipelines.
* **Extract meaningful brain signal features** (Alpha, Beta, Theta bands) via Fast Fourier Transform (FFT).
* **Serve Real-Time Predictions** by deploying a machine learning classifier through a fast, asynchronous REST API.
* **Maintain Engineering Rigor** through strict unit testing, continuous integration (CI/CD), and containerization.

## 🏗️ Project Architecture

The codebase follows modern software engineering principles, separating data ingestion, signal processing, machine learning, and API routing.

```text
dreamsense/
├── .github/workflows/ci.yml  # Automated testing pipeline
├── src/
│   ├── data_loader.py        # Safe data ingestion & matrix preparation
│   ├── feature_extractor.py  # DSP and FFT brainwave band extraction
│   ├── classifier.py         # ML model lifecycle (train, predict)
│   └── main.py               # FastAPI application routing
├── tests/
│   ├── test_feature_extractor.py
│   └── test_classifier.py
├── requirements.txt          # Dependency management
└── Dockerfile                # Containerization blueprint
```

## 🛠️ Tech Stack

**Data Science & Machine Learning:**
* Python 3.10
* NumPy & Pandas (Data Manipulation)
* Scikit-Learn (Random Forest Classification)
* MNE-Python / SciPy (Signal Processing)

**Software Engineering & DevOps:**
* **FastAPI & Uvicorn:** High-performance REST API routing.
* **Pytest:** Automated unit testing suite.
* **Docker:** Containerization for consistent cross-platform deployment.
* **GitHub Actions:** CI/CD pipelines for automated code validation.

## 🚀 Getting Started

### Option 1: Run via Docker (Recommended)
The easiest way to run the DreamSense API is via Docker, ensuring no dependency conflicts.

```bash
# Build the Docker image
docker build -t dreamsense-api .

# Run the container mapping port 8000
docker run -p 8000:8000 dreamsense-api
```

### Option 2: Local Python Environment
```bash
# Clone the repository
git clone [https://github.com/ArunavaBhol/DreamSense.git](https://github.com/ArunavaBhol/DreamSense.git)
cd DreamSense

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn src.main:app --reload
```

*The interactive API documentation will be available at `http://localhost:8000/docs`.*

## 🔌 API Endpoints

Once the server is running, you can interact with the following REST endpoints:

* **`POST /train`**
  Triggers the backend to load `emotions.csv`, extract features, and train the Random Forest classifier. Returns training accuracy and classification metrics.

* **`POST /predict`**
  Accepts a JSON payload of extracted EEG features and returns the predicted emotional state in real-time.
  ```json
  {
    "features": [12.4, -0.5, 3.14, 8.9]
  }
  ```

## 🧪 Testing & CI/CD

Quality assurance is maintained through automated testing. Every push to the `main` branch triggers GitHub Actions to run the full test suite.

To run tests locally:
```bash
pytest tests/ -v
```

## 🔭 Future Work

* **Cloud Deployment:** Deploy the Dockerized API to AWS EC2 or Google Cloud Run.
* **Database Integration:** Implement PostgreSQL/MongoDB to log real-time predictions and user sessions.
* **Advanced Deep Learning:** Upgrade from Random Forest to an EEGNet (CNN-based) architecture for raw signal classification without manual feature extraction.
* **WebSocket Integration:** Support continuous, high-frequency streaming of EEG data rather than single-batch REST payloads.

## ⚠️ Note

DreamSense does not reconstruct exact visual dreams. It focuses strictly on interpreting and classifying the underlying emotional patterns derived from raw EEG signal frequencies.

## 📫 Contact

* **Developer:** Arunava Bhol
* **Project:** DreamSense
