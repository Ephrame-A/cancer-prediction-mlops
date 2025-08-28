
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "cancermodel")
API_KEY_NAME = "X-API-Key"
TF_SERVING_URL = "http://localhost:8501/v1/models/my_model:predict"

app = FastAPI()
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
