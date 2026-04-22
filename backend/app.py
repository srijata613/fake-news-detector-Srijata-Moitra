from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load lightweight TF-IDF model ----------
print("Loading TF-IDF model...")

vectorizer = joblib.load("backend/model/tfidf_vectorizer.pkl")
tfidf_model = joblib.load("backend/model/tfidf_logreg_model.pkl")


# ---------- DistilBERT lazy loading ----------
bert_model = None
tokenizer = None


def load_bert():
    global bert_model, tokenizer

    if bert_model is None:
        print("Loading DistilBERT model from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
        )
        bert_model.eval()


class NewsRequest(BaseModel):
    text: str
    mode: str = "fast"   # fast or bert


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: NewsRequest):

    text = data.text.strip()

    if not text:
        return {"error": "No text provided"}

    # ---------- FAST MODE (default) ----------
    if data.mode == "fast":

        X = vectorizer.transform([text])

        pred = tfidf_model.predict(X)[0]
        prob = tfidf_model.predict_proba(X)[0].max()

        label = "REAL" if pred == 1 else "FAKE"

        return {
            "model": "TF-IDF",
            "label": label,
            "confidence": round(prob * 100, 2)
        }

    # ---------- BERT MODE ----------
    elif data.mode == "bert":

        load_bert()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

        label = "REAL" if predicted.item() == 1 else "FAKE"

        return {
            "model": "DistilBERT",
            "label": label,
            "confidence": round(confidence.item() * 100, 2)
        }