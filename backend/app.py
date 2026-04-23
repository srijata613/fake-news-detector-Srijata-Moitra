from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import string
from pathlib import Path


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Model Paths 
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(MODEL_DIR / "tfidf_logreg_model.pkl")


#  Request Schema 
class NewsRequest(BaseModel):
    text: str


#  Text Cleaning
def clean_text(text: str):

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    return text


#  Root Endpoint 
@app.get("/")
def root():
    return {"message": "Fake News Detector API running"}


# Prediction Endpoint 
@app.post("/predict")
def predict(data: NewsRequest):

    if not data.text.strip():
        return {"error": "No text provided"}

    # Clean text before vectorization
    cleaned_text = clean_text(data.text)

    text_vec = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_vec)[0]

    prob = model.predict_proba(text_vec)[0].max()

    # Correct mapping
    label = "REAL" if prediction == 1 else "FAKE"

    return {
        "label": label,
        "confidence": round(prob * 100, 2)
    }