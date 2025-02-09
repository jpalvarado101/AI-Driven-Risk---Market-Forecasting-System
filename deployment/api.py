"""
deployment/api.py

Creates a FastAPI application that serves real-time predictions from the trained RL agent.
The API accepts market data as JSON, computes extra features from historical data,
and returns the predicted trading action.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from models.rl_agent import load_agent
import uvicorn
import os

app = FastAPI(title="RL Trading Agent API")

# Dictionary to cache models per ticker
ticker_models = {}

def get_model_for_ticker(ticker: str):
    if ticker not in ticker_models:
        model_path = f"./models/ppo_trading_agent_{ticker}.zip"
        print(f"Loading model from {model_path}")
        ticker_models[ticker] = load_agent(model_path)
    return ticker_models[ticker]

def compute_extra_features(ticker: str, new_data: dict):
    """
    Computes extra features (vol_spike, drawdown, price_zscore) for the new observation,
    using historical data from the ticker's CSV file.
    
    Parameters:
      ticker: the ticker symbol (e.g., "AAPL")
      new_data: a dict with keys: Open, High, Low, Close, Volume
      
    Returns:
      vol_spike, drawdown, price_zscore
    """
    file_path = f"./data/processed_data_{ticker}.csv"
    if not os.path.exists(file_path):
        raise ValueError(f"Historical data file {file_path} does not exist.")
    
    # Load historical data for this ticker
    df = pd.read_csv(file_path)
    
    # Append the new observation as a new row
    # new_data should include only the numeric fields required to compute extra features.
    new_df = pd.DataFrame([new_data])
    df_combined = pd.concat([df, new_df], ignore_index=True)
    
    # Compute 30-day rolling average volume (use min_periods=1 to avoid NaNs)
    df_combined['avg_volume'] = df_combined['Volume'].rolling(window=30, min_periods=1).mean()
    vol_spike = df_combined['Volume'].iloc[-1] / df_combined['avg_volume'].iloc[-1]
    
    # Compute drawdown:
    # Drawdown = (Peak Price - Current Price) / Peak Price, where Peak Price is the cumulative max of Close.
    df_combined['peak_price'] = df_combined['Close'].cummax()
    drawdown = (df_combined['peak_price'].iloc[-1] - df_combined['Close'].iloc[-1]) / df_combined['peak_price'].iloc[-1]
    
    # Compute price z-score over a 30-day rolling window.
    df_combined['price_mean'] = df_combined['Close'].rolling(window=30, min_periods=1).mean()
    # Replace zero standard deviations with 1 to avoid division by zero.
    df_combined['price_std'] = df_combined['Close'].rolling(window=30, min_periods=1).std().replace(0, 1)
    price_zscore = (df_combined['Close'].iloc[-1] - df_combined['price_mean'].iloc[-1]) / df_combined['price_std'].iloc[-1]
    
    return vol_spike, drawdown, price_zscore

# Define the data model for incoming requests (without extra features)
class MarketData(BaseModel):
    ticker: str
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
    print(f"Received prediction request for ticker: {data.ticker}")
    
    try:
        # Get the model for the provided ticker
        model = get_model_for_ticker(data.ticker)
    except Exception as e:
        error_msg = f"Model loading failed: {e}"
        print(error_msg)
        return {"error": error_msg}
    
    # Prepare a dict with the fields needed to compute extra features
    new_obs_data = {
        "Open": data.Open,
        "High": data.High,
        "Low": data.Low,
        "Close": data.Close,
        "Volume": data.Volume,
    }
    
    try:
        # Compute extra features from historical data
        vol_spike, drawdown, price_zscore = compute_extra_features(data.ticker, new_obs_data)
        print(f"Computed extra features - vol_spike: {vol_spike}, drawdown: {drawdown}, price_zscore: {price_zscore}")
    except Exception as e:
        error_msg = f"Failed to compute extra features: {e}"
        print(error_msg)
        return {"error": error_msg}
    
    # Build the observation vector for the model (14 features)
    # Order must match what the model was trained on:
    # [Open, High, Low, Close, Volume, rsi, macd, sma_50, sma_200, vol_spike, drawdown, price_zscore, balance, shares_held]
    obs = np.array([[data.Open, data.High, data.Low, data.Close, data.Volume,
                     data.rsi, data.macd, data.sma_50, data.sma_200,
                     vol_spike, drawdown, price_zscore,
                     data.balance, data.shares_held]], dtype=np.float32)
    
    try:
        # Get the predicted action from the RL agent
        prediction = model.predict(obs)
        print("Prediction output:", prediction)
        action, _ = prediction
        # Map numeric actions to human-readable form
        action_mapping = {0: "hold", 1: "buy", 2: "sell"}
        predicted_action = int(action[0])
        print("Predicted action:", predicted_action)
        return {"action": action_mapping.get(predicted_action, "unknown")}
    except Exception as e:
        error_msg = f"Prediction failed: {e}"
        print(error_msg)
        return {"error": error_msg}

@app.get("/")
def home():
    return {"message": "Welcome to the RL Trading API! Use /predict to get predictions."}

# Only run Uvicorn when executing the script directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # For cloud deployment compatibility
    uvicorn.run(app, host="0.0.0.0", port=port)
