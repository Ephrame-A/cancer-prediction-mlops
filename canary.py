import requests
import os
from dotenv import load_dotenv
load_dotenv()

url = 'http://localhost:8000/canary'
api_key = os.environ.get("API_KEY", "cancermodel")
headers = {"content-type": "application/json", "x-api-key": api_key}

# Example: 5 samples, 17 features each
instances = [[0.1]*17 for _ in range(5)]
payload = {"instances": instances}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
