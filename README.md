
# Heart Disease Prediction – End-to-End MLOps Pipeline

## Overview
End-to-end MLOps pipeline using the UCI Heart Disease dataset:
- EDA
- Feature engineering
- Model training & tracking (MLflow)
- CI/CD (GitHub Actions)
- Dockerized FastAPI inference
- Kubernetes deployment
- Monitoring with Prometheus metrics

## Prerequisites
- Python 3.10+
- Docker
- Git
- (Optional) Kubernetes / Minikube

## Setup Instructions

```bash
git clone <your-repo-url>
cd heart-disease-mlops
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step-by-Step Execution

### 1. Download Dataset
```bash
python data/raw/download_data.py
```

### 2. Run EDA
```bash
python notebooks/eda.py
```

### 3. Train Models + MLflow Tracking
```bash
mlflow ui
python src/train.py
```

### 4. Run Tests
```bash
pytest
```

### 5. Run API Locally
```bash
uvicorn api.app:app --reload
```

### 6. Docker Build & Run
```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

### 7. Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### 8. API Test
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
-d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

