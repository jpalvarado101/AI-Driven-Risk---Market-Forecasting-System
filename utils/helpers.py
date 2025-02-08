"""
utils/helpers.py

Contains utility functions used across the project.
For example: plotting training rewards.
"""

import matplotlib.pyplot as plt

def plot_rewards(rewards, filename="rewards_plot.png"):
    """
    Plots training rewards over episodes and saves the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Reward plot saved as {filename}")
