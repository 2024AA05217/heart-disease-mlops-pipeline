
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("mlruns/0/latest/model")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    return {"prediction": int(prob > 0.5), "confidence": prob}
