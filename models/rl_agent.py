"""
models/rl_agent.py

Implements and trains the RL agent using PPO from stable-baselines3.
Provides functions for training and loading the trained model.
"""

import pandas as pd
from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(total_timesteps=10000):
    # Load processed market data
    df = pd.read_csv("./data/processed_data.csv")
    
    # Create the trading environment
    env = TradingEnv(df)
    env = DummyVecEnv([lambda: env])

    # Initialize the PPO agent with a multilayer perceptron policy
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")

    # Save the trained model
    model.save("./models/ppo_trading_agent")
    print("Model saved to ./models/ppo_trading_agent.zip")
    return model

def load_agent(model_path="./models/ppo_trading_agent.zip"):
    # Load a previously trained agent
    model = PPO.load(model_path)
    return model

if __name__ == "__main__":
    train_agent()
