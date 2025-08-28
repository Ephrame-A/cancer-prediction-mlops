## 1. Endpoint and API Documentation

### **URL Structure**
The endpoint for making predictions on your deployed model is:
`http://localhost:8501/v1/models/my_model:predict`

-   `http://localhost:8501`: This is the host and port of your running TensorFlow Serving instance.
-   `/v1/models`: This is the standard path for the TensorFlow Serving REST API.
-   `/my_model`: This is the **model name** (`MODEL_NAME`) you set when running the Docker container.
-   `:predict`: This is the **verb** used to request an inference from the model.

---

## 2. Sample Request Code

The following Python script demonstrates how to prepare data and send a POST request to your model endpoint. It also includes error handling by converting the NumPy array to a JSON-serializable list.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import json
import numpy as np

# Load and prepare data (Lung Cancer Dataset)
print("Loading Lung Cancer Dataset ...")
df = pd.read_csv('./data/Lung Cancer Dataset.csv')

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Convert target to numeric if needed
if y.dtype == object or y.apply(lambda v: isinstance(v, str)).any():
  y = y.map({'YES': 1, 'NO': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

test_samples = X_test_scaled[0:5]

# Prepare the JSON payload by converting the NumPy array to a list
data_payload = json.dumps({"instances": test_samples.tolist()})

# Define the endpoint and headers
url = 'http://localhost:8501/v1/models/my_model:predict'
headers = {"content-type": "application/json"}


# 🚀 Lung Cancer Prediction Model API & Deployment Guide


## 🟢 1. Endpoint & API Reference

**Prediction Endpoint:**

```text
http://localhost:8501/v1/models/my_model:predict
```

| Component         | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `localhost:8501`  | Host & port of TensorFlow Serving                                  |
| `/v1/models`      | REST API base path                                                 |
| `/my_model`       | Model name (set via Docker `MODEL_NAME` env)                      |
| `:predict`        | REST verb for inference                                            |


## 🧑‍💻 2. Sample Request (Python)

> Prepares data, sends a POST request, and handles errors. Uses the Lung Cancer Dataset.

```python
...existing code...
```


## 🔄 3. Promotion & Rollback Logic

- `train.py` checks if new model accuracy ≥ **90%**
- If passed, model is saved in a new versioned folder (e.g., `models/2/`)
- TensorFlow Serving automatically serves the highest version
- If failed, previous version remains active (no rollback needed)


## 📊 4. Monitoring (Prometheus)

TensorFlow Serving exposes Prometheus metrics at:

```text
http://localhost:8501/monitoring/prometheus
```

**Prometheus Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'tensorflow_serving'
    static_configs:
      - targets: ['localhost:8501']
```


## 🧪 5. Canary Testing

To canary test a new model version:
- Send requests to `/v1/models/my_model/versions/<version>:predict`
- Compare predictions with previous version

**Example:**
```python
...existing code...
```


## 🤖 6. Automation (GitHub Actions)

Automate retraining & deployment with GitHub Actions:

```yaml
...existing code...
```


## 🧬 7. Canary Test Script Example

```python
...existing code...
```

        run: |
          python train.py
      - name: Upload exported model
        uses: actions/upload-artifact@v3
        with:
          name: exported-model
          path: exports/
```

---

## 7. Canary Test Script Example

```python
import requests
import json
import numpy as np

# Prepare test data (replace with your own test samples)
test_samples = np.random.rand(5, 17)  # Example shape for lung cancer dataset

# Predict with previous version
url_prev = 'http://localhost:8501/v1/models/my_model/versions/1:predict'
headers = {"content-type": "application/json"}
data = json.dumps({"instances": test_samples.tolist()})
response_prev = requests.post(url_prev, data=data, headers=headers)
preds_prev = json.loads(response_prev.text)['predictions']

# Predict with new version
url_new = 'http://localhost:8501/v1/models/my_model/versions/2:predict'
response_new = requests.post(url_new, data=data, headers=headers)
preds_new = json.loads(response_new.text)['predictions']

print('Previous version predictions:', preds_prev)
print('New version predictions:', preds_new)
```

---
