from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Enable CORS so frontend can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "distilbert-base-uncased"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model from HuggingFace...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Reduce memory usage
model.eval()

class NewsRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: NewsRequest):

    text = data.text.strip()

    if not text:
        return {"error": "No text provided"}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    confidence, predicted = torch.max(probs, dim=1)

    label = "REAL" if predicted.item() == 1 else "FAKE"

    return {
        "label": label,
        "confidence": round(confidence.item() * 100, 2)
    }