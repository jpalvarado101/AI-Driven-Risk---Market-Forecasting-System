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

# Load the trained RL model (ensure you have trained the agent before starting the API)
model = load_agent()

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
    # Build the observation vector for the model
    obs = np.array([[data.Open, data.High, data.Low, data.Close, data.Volume,
                     data.rsi, data.macd, data.sma_50, data.sma_200,
                     data.balance, data.shares_held]], dtype=np.float32)
    # Get the predicted action from the RL agent
    action, _ = model.predict(obs)
    # Map numeric actions to human-readable form
    action_mapping = {0: "hold", 1: "buy", 2: "sell"}
    predicted_action = int(action[0])
    return {"action": action_mapping.get(predicted_action, "unknown")}

if __name__ == "__main__":
    # Run the API with uvicorn on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
