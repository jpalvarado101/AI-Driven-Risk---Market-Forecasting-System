"""
train.py

Entry-point script to train the RL agent.
This script loads the data, initializes the environment, trains the PPO agent,
and saves the trained model.
"""

from models.rl_agent import train_agent

def main():
    print("Starting training of the RL Trading Agent...")
    # Adjust total_timesteps as needed (default set to 10,000)
    train_agent(total_timesteps=10000)
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
