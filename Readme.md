
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import json
import numpy as np

print("Loading Lung Cancer Dataset ...")
df = pd.read_csv('./data/Lung Cancer Dataset.csv')

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Convert target to numeric if needed
if y.dtype == object or y.apply(lambda v: isinstance(v, str)).any():
    y = y.map({'YES': 1, 'NO': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

test_samples = X_test_scaled[0:5]

data = json.dumps({"instances": test_samples.tolist()})

url = 'http://localhost:8000/predict'  # FastAPI gateway endpoint
api_key = input("Enter API Key: ")
headers = {"content-type": "application/json", "X-API-Key": api_key}

json_response = requests.post(url, data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']

predicted_classes = [1 if pred[0] > 0.5 else 0 for pred in predictions]
actual_classes = y_test.values.tolist()[0:5]
print("\nPredicted classes:")
print(predicted_classes)
print("Actual classes: ")
print(actual_classes)
```

---

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

TensorFlow Serving exposes Prometheus metrics at:
```
http://localhost:8501/monitoring/prometheus
```
**Prometheus Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'tensorflow_serving'
    static_configs:
      - targets: ['localhost:8501']
```

---

## 🧪 Canary Testing

- Send requests to `/v1/models/my_model/versions/<version>:predict`
- Compare predictions with previous version

---

## 🤖 Automation (GitHub Actions)

Automate retraining & deployment with GitHub Actions:
```yaml
name: Retrain and Promote Model
on:
  schedule:
    - cron: '0 0 * * 0'  # Runs weekly on Sunday at midnight
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run training script
        run: |
          python train.py
      - name: Upload exported model
        uses: actions/upload-artifact@v3
        with:
          name: exported-model
          path: models/
```

---
