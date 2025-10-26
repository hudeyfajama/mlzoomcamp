# FastAPI code for serving the model (save this as app.py)

from fastapi import FastAPI
import pickle

app = FastAPI()

# Load the pipeline at startup
with open('pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Lead scoring model is running"}

@app.post("/predict")
def predict(client: dict):
    """
    Predict conversion probability for a client
    """
    # Make prediction
    prediction_proba = pipeline.predict_proba([client])[0, 1]
    
    return {
        "conversion_probability": float(prediction_proba),
        "conversion_probability_rounded": round(float(prediction_proba), 3)
    }


