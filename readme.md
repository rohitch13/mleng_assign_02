# Headline Sentiment Scoring API

This service provides real-time sentiment scoring for news headlines using a pre-trained SVM model with sentence embeddings.

---

## Overview

The API exposes two endpoints:

- **GET** `/status`  
  Returns a JSON confirming the service is running:  
  ```json
  { "status": "OK" }
  ```

- **POST** /score_headlines
Accepts a JSON payload containing a list of headlines and returns sentiment labels predicted by the model.

## Input Format for /score_headlines

```json
{
  "headlines": [
    "what is going on",
    "I hate this and that stuff"
  ]
}
```

## Usage
```bash
uvicorn score_headlines_api:app --host 0.0.0.0 --port 8011
```

```bash
curl http://localhost:8011/status
```

```bash
curl -X POST http://localhost:8011/score_headlines -H "Content-Type: application/json" -d '{"headlines": ["what is going on", "I hate this and that stuff"]}'
```