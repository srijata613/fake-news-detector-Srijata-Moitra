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


REAL_PATTERNS = [
    r"\b(reuters|cnn|nyt|bbc|associated press|ap news)\b",
    r"\b(pentagon|white house|supreme court|senator|congress)\b",
    r"\b(said|announced|according to|reported)\b",
]

FAKE_PATTERNS = [
    r"\b(secret|shocking|insane|miracle|disturbing)\b",
    r"\b(will make you sick|what happens next)\b",
    r"\b(conspiracy|exposed|they don't want you to know)\b",
]


def rule_based_check(text):

    t = text.lower()

    for p in REAL_PATTERNS:
        if re.search(p, t):
            return "REAL", 0.75

    for p in FAKE_PATTERNS:
        if re.search(p, t):
            return "FAKE", 0.85

    return None


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

    text = data.text.strip()

    if not text:
        return {"error": "No text provided"}

    # RULE LAYER FIRST
    rule_result = rule_based_check(text)
    if rule_result:
        label, conf = rule_result
        return {
            "label": label,
            "confidence": round(conf*100,2)
        }

    # ML MODEL
    cleaned = clean_text(text)
    text_vec = vectorizer.transform([cleaned])
    
    pred = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0].max()
    
    label = "REAL" if pred == 1 else "FAKE"
    
    # confidence guard
    if prob < 0.60:
        label = "UNCERTAIN"

    return {
        "label": label,
        "confidence": round(prob*100,2)
    }