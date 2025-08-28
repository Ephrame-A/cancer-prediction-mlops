
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
import requests
import os
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator
load_dotenv()

API_KEY = os.environ.get("API_KEY", "cancermodel")
API_KEY_NAME = "x-api-key"
TF_SERVING_URL = "http://localhost:8501/v1/models/my_model:predict"

app = FastAPI()
Instrumentator().instrument(app).expose(app)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


@app.post("/predict")
async def predict(request: Request, api_key: str = Depends(get_api_key)):
    payload = await request.json()
    try:
        response = requests.post(TF_SERVING_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Canary endpoint: compare predictions from two model versions
@app.post("/canary")
async def canary(request: Request, api_key: str = Depends(get_api_key)):
    payload = await request.json()
    instances = payload.get("instances", [])
    if not instances:
        raise HTTPException(status_code=400, detail="Missing 'instances' in payload")
    # URLs for previous and new model versions
    prev_url = "http://localhost:8501/v1/models/my_model/versions/1:predict"
    new_url = "http://localhost:8501/v1/models/my_model/versions/2:predict"
    try:
        prev_resp = requests.post(prev_url, json={"instances": instances})
        new_resp = requests.post(new_url, json={"instances": instances})
        prev_resp.raise_for_status()
        new_resp.raise_for_status()
        prev_preds = prev_resp.json().get("predictions", [])
        new_preds = new_resp.json().get("predictions", [])
        return {
            "previous_version_predictions": prev_preds,
            "new_version_predictions": new_preds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
