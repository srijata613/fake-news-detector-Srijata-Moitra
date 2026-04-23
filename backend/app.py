from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/tfidf_logreg_model.pkl")

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake News Detector API running"}

@app.post("/predict")
def predict(data: NewsRequest):

    if not data.text.strip():
        return {"error": "No text provided"}

    text_vec = vectorizer.transform([data.text])

    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0].max()

    label = "FAKE" if prediction == 1 else "REAL"

    return {
        "label": label,
        "confidence": round(prob * 100, 2)
    }