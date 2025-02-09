"""
models/rl_agent.py

Implements and trains the RL agent using PPO from stable-baselines3.
Provides functions for training and loading the trained model.
"""

import pandas as pd
from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(ticker="AAPL", total_timesteps=10000):
    # Load processed market data for the given ticker
    df = pd.read_csv(f"./data/processed_data_{ticker}.csv")
    
    # Create the trading environment
    env = TradingEnv(df)
    env = DummyVecEnv([lambda: env])

    # Initialize the PPO agent with a multilayer perceptron policy
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    print(f"Starting training for {ticker}...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")

    # Save the trained model
    model.save(f"./models/ppo_trading_agent_{ticker}")
    print(f"Model saved to ./models/ppo_trading_agent_{ticker}.zip")
    return model

def load_agent(model_path):
    # Load a previously trained agent from the given model_path
    model = PPO.load(model_path)
    return model

if __name__ == "__main__":
    train_agent()
