
# 🚀 Lung Cancer Prediction ML Project

---

## 🗂️ Project Overview

This project predicts lung cancer risk using a machine learning model trained on a custom dataset. It features:
- Model training and versioning
- Secure API access via FastAPI gateway (API key authentication)
- User-friendly Flask web frontend
- CLI script for quick predictions

---

## 🟢 Endpoints & API Reference

**TensorFlow Serving Endpoint:**
```
http://localhost:8501/v1/models/my_model:predict
```

**FastAPI Gateway Endpoint (with API key):**
```
http://localhost:8000/predict
Header: X-API-Key: <your_api_key>
```

---

## 🔑 API Key Authentication

- The FastAPI gateway requires an API key in the `X-API-Key` header.
- The key is loaded from a `.env` file (use `python-dotenv`). Example:
  ```
  API_KEY=your_secret_key
  ```
- Update your `.env` file to set/change the key.

---

## 🧑‍💻 Sample Request (Python)

Example using CLI script (`model.py`):
```python 
  python model.py
```

## 🌐 Flask Web Frontend

- Located in `frontend/app.py`
- Users enter feature values and their API key in a styled web form
- Prediction is displayed as "Positive" or "Negative"

---

## 🔄 Model Training & Promotion

- `train.py` trains the model and saves it in a versioned `models` folder if accuracy ≥ 90%
- TensorFlow Serving automatically serves the highest version
- Previous versions are retained for rollback

---


## 📊 Monitoring (Prometheus)

Prometheus metrics for your FastAPI gateway are available at:
```
http://127.0.0.1:8000/metrics
```
**Prometheus Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'fastapi_gateway'
    static_configs:
      - targets: ['127.0.0.1:8000']
```

---

## 🧪 Canary Testing

- Send requests to `/v1/models/my_model/versions/<version>:predict`
- Compare predictions with previous version

---

## 🤖 Automation (GitHub Actions)

Automate retraining & deployment with GitHub Actions:


