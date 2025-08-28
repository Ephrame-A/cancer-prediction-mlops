
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import json
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

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

test_samples = X_test_scaled[0:10]

data = json.dumps({"instances": test_samples.tolist()})


url = 'http://localhost:8000/predict'  # FastAPI gateway endpoint
api_key = os.environ.get("API_KEY", "cancermodel")
headers = {"content-type": "application/json", "x-api-key": api_key}

json_response = requests.post(url, data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']

predicted_classes = [1 if pred[0] > 0.5 else 0 for pred in predictions]
actual_classes = y_test.values.tolist()[0:10]
print("\nPredicted classes:")
print(predicted_classes)
print("Actual classes: ")
print(actual_classes)