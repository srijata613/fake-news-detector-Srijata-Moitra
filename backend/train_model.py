import pandas as pd
import joblib
import re
import string
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

# Combine headline + article
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

df = pd.concat([fake, true])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean
df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2),
    max_features=100000,
    min_df=2,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Models
logreg = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    C=2,
    solver="liblinear"
)

nb = MultinomialNB()


# Hybrid ensemble
model = VotingClassifier(
    estimators=[
        ("lr", logreg),
        ("nb", nb)
    ],
    voting="soft"
)

model.fit(X_train_vec, y_train)


# Evaluate
pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# Save
Path("model").mkdir(exist_ok=True)

joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
joblib.dump(model, "model/tfidf_logreg_model.pkl")

print("\nHybrid model saved.")