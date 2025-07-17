#!/usr/bin/env python3

import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, RootModel
from typing import List
import joblib
import os
from sentence_transformers import SentenceTransformer
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "svm.joblib"
EMBEDDING_MODEL_PATH = "/opt/huggingface_models/all-MiniLM-L6-v2"  
EMBEDDING_MODEL_PATH = "all-MiniLM-L6-v2"

app = FastAPI()

model = None
embedder = None

class HeadlineList(RootModel[List[str]]):
    pass

@app.on_event("startup")
def load_models():
    global model, embedder
    logger.info("Starting up: Loading SVM and embedding models...")

    if not os.path.exists(MODEL_PATH):
        logger.critical(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError("SVM model file not found.")

    model = joblib.load(MODEL_PATH)
    logger.info("SVM model loaded.")

    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
        logger.info("Sentence transformer model loaded.")
    except Exception as e:
        logger.critical(f"Error loading embedding model: {e}")
        raise RuntimeError("Failed to load sentence transformer model.")

@app.get("/status")
def status():
    logger.info("Status check requested.")
    return {"status": "OK"}

class HeadlinesData(BaseModel):
    headlines: List[str]

@app.post("/score_headlines")
def score_headlines(data: HeadlinesData):
    headlines = data.headlines
    logger.info(f"Received {len(headlines)} headlines for scoring")

    # Model inference
    vectors = embedder.encode(headlines)
    predictions = model.predict(vectors).tolist()

    return {"labels": predictions}

if __name__ == "__main__":
    try:
        uvicorn.run("score_headlines_api:app", host="0.0.0.0", port=8011, log_level="info")
    except KeyboardInterrupt:
        logging.info("Server shutdown requested via KeyboardInterrupt (Ctrl+C). Exiting cleanly.")


#pip install --upgrade pip
#pip install -r requirements.txt
#uvicorn score_headlines_api:app --host 0.0.0.0 --port 8011 --reload
