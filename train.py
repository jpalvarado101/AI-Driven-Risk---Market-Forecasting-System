"""
train.py

Entry-point script to train the RL agents for multiple tickers.
This script loops over a list of tickers, loads their data, initializes the environment,
trains the PPO agent, and saves the trained model for each ticker.
"""

from models.rl_agent import train_agent

def main():
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "META"]
    for ticker in tickers:
        print(f"Starting training for {ticker} RL Trading Agent...")
        train_agent(ticker=ticker, total_timesteps=10000)
        print(f"Training for {ticker} completed.\n")

if __name__ == "__main__":
    main()
