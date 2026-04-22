# Importing Necessary Libraries

import pandas as pd
import re
import string
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np


# Load Dataset

data_dir = Path("data")

fake = pd.read_csv(data_dir / "Fake.csv")
true = pd.read_csv(data_dir / "True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["content"] = df["title"] + " " + df["text"]

print("Total samples:", len(df))
print("Fake samples:", sum(df["label"] == 0))
print("Real samples:", sum(df["label"] == 1))


# Clean Text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["content"] = df["content"].apply(clean_text)


# Train/Test Split

train_text, test_text, train_labels, test_labels = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# BASELINE MODEL (TF-IDF + Logistic Regression)

print("\nTraining TF-IDF + Logistic Regression...\n")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

lr_model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

lr_model.fit(X_train, train_labels)

preds = lr_model.predict(X_test)

print("\nBaseline Model Evaluation")
print("--------------------------")
print("Accuracy:", accuracy_score(test_labels, preds))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, preds))
print("\nClassification Report:")
print(classification_report(test_labels, preds))


# Save TF-IDF Model

Path("model").mkdir(exist_ok=True)

joblib.dump(lr_model, "model/tfidf_logreg_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nTF-IDF model and vectorizer saved.")


# DISTILBERT MODEL

print("\nTraining DistilBERT...\n")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_dataset = Dataset.from_dict({
    "text": list(train_text),
    "label": list(train_labels)
})

test_dataset = Dataset.from_dict({
    "text": list(test_text),
    "label": list(test_labels)
})


# Tokenization

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)


# Load DistilBERT Model

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# Training Configuration

training_args = TrainingArguments(
    output_dir="model/bert_output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
)


# Trainer Setup

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# Train Model

trainer.train()


# Evaluate DistilBERT

predictions = trainer.predict(test_dataset)

pred_labels = np.argmax(predictions.predictions, axis=1)

print("\nDistilBERT Evaluation")
print("----------------------")
print("Accuracy:", accuracy_score(test_labels, pred_labels))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, pred_labels))
print("\nClassification Report:")
print(classification_report(test_labels, pred_labels))


# Save DistilBERT Model

trainer.save_model("model/distilbert_fake_news")

print("\nDistilBERT model saved successfully.")