"""
deployment/api.py

Creates a FastAPI application that serves real-time predictions from the trained RL agent.
The API accepts market data as JSON and returns the predicted trading action.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from models.rl_agent import load_agent
import uvicorn

app = FastAPI(title="RL Trading Agent API")

# Load the trained RL model
try:
    model = load_agent()
except Exception as e:
    print(f"Error loading RL model: {e}")
    model = None  # Prevent API from crashing

# Define a root endpoint to avoid "Not Found" error
@app.get("/")
def home():
    return {"message": "Welcome to the RL Trading API! Use /predict to get predictions."}

# Define the data model for incoming requests
class MarketData(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    rsi: float
    macd: float
    sma_50: float
    sma_200: float
    balance: float
    shares_held: float

@app.post("/predict")
def predict(data: MarketData):
    if model is None:
        return {"error": "Model is not loaded. Train the model first!"}

    # Build the observation vector for the model
    obs = np.array([[data.Open, data.High, data.Low, data.Close, data.Volume,
                     data.rsi, data.macd, data.sma_50, data.sma_200,
                     data.balance, data.shares_held]], dtype=np.float32)

    # Get the predicted action from the RL agent
    try:
        action, _ = model.predict(obs)
        action_mapping = {0: "hold", 1: "buy", 2: "sell"}
        return {"action": action_mapping.get(int(action[0]), "unknown")}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# Only run Uvicorn when executing the script directly
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))  # Cloud deployment compatibility
    uvicorn.run(app, host="0.0.0.0", port=port)
