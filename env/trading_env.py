"""
env/trading_env.py

Defines a custom Gym environment for simulating trading based on historical market data.
The agent can take three actions: 0 (hold), 1 (buy), or 2 (sell).

This version includes additional features:
- vol_spike: Volume spike indicator (Current Volume / 30-day Average Volume)
- drawdown: Drawdown percentage from peak price
- price_zscore: Z-score of the price over a 30-day window

It also adjusts the reward function:
- Rewards extra when buying during oversold conditions (drawdown > 20% or price_zscore < -2)
- Penalizes selling during high drawdown conditions.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(self.df) - 1

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # [Open, High, Low, Close, Volume, RSI, MACD, SMA50, SMA200,
        #  vol_spike, drawdown, price_zscore, balance, shares_held]
        self.observation_dim = 14
        low = -np.inf * np.ones(self.observation_dim, dtype=np.float32)
        high = np.inf * np.ones(self.observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _next_observation(self):
        # Get the current market data row
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
            row['rsi'], row['macd'], row['sma_50'], row['sma_200'],
            row['vol_spike'], row['drawdown'], row['price_zscore'],
            self.balance,
            self.shares_held
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        # Get current row features (for reward adjustments)
        current_row = self.df.iloc[self.current_step]

        # Execute the action
        if action == 1:  # Buy one share
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2:  # Sell one share
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price

        self.current_step += 1

        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price

        # Base reward: change in net worth from initial balance
        reward = self.net_worth - self.initial_balance

        # ---------------------------
        # Scenario-Based Reward Adjustments
        # ---------------------------
        if self.current_step > 0:
            prev_row = self.df.iloc[self.current_step - 1]
            # Bonus for buying during oversold conditions:
            # If drawdown > 20% or price_zscore < -2, add bonus reward.
            if action == 1 and (prev_row['drawdown'] > 0.20 or prev_row['price_zscore'] < -2):
                reward += 10
            # Penalize selling during high drawdown (panic sell-off)
            if action == 2 and (prev_row['drawdown'] > 0.20):
                reward -= 5

        done = self.current_step >= self.max_steps

        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._next_observation()

    def render(self, mode='human', close=False):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}, Shares held: {self.shares_held}, Net Worth: {self.net_worth:.2f}")
