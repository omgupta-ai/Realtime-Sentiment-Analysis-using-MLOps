# File: api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Sentiment Analysis API")

# --- Load Model ---

# Corrected paths for the Docker container
model_path = 'model/logistic_regression.joblib'
vectorizer_path = 'model/tfidf_vectorizer.joblib'

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# --- API Models ---
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    sentiment_score: int

# --- Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    # Vectorize the input text
    processed_text = vectorizer.transform([request.text])

    # Predict with the model
    prediction = model.predict(processed_text)[0]

    sentiment_label = "Positive" if prediction == 1 else "Negative"

    return SentimentResponse(
        sentiment=sentiment_label,
        sentiment_score=int(prediction)
    )