# Fake News Detector

A web application that detects whether a news headline or article snippet is **Real or Fake** using a trained Machine Learning model.

## Features

- DistilBERT Fake News Classification
- TF-IDF + Logistic Regression baseline
- FastAPI backend with `/predict` endpoint
- Interactive frontend UI
- Dark mode
- History panel
- Confidence score visualization

## Tech Stack

Frontend:
- HTML
- CSS
- Vanilla JavaScript
- Bootstrap

Backend:
- FastAPI
- PyTorch
- HuggingFace Transformers

Machine Learning:
- ISOT Fake News Dataset
- TF-IDF
- Logistic Regression
- DistilBERT

## Run Backend

```bash
uvicorn backend.app:app --reload

API:

POST /predict

Example request:

{
"text": "NASA confirms presence of water on Mars"
}

Example response:

{
"label": "REAL",
"confidence": 96.4
}

---

![Figure 1: User Interface of the Fake News Detection System]({8A678D71-571A-47D7-B61B-AFDAA5ABF69B}.png)

![Figure 2: Prediction output for real news]({66BE494B-6691-4E0E-B87C-F59179A7D3DD}.png)

![Figure 3: Detection of fake news article]({88F8EF62-5999-4C45-ABDC-4114EDE02D1A}.png)

![Figure 4: Prediction history of recent analyses]({0EE7C7AC-6D27-4E0A-B693-28098B90C1FC}.png)

![Figure 5: FastAPI interactive API documentation]({230A9458-37B3-4648-981A-ABAA7770757E}.png)