"""
risk_management/risk_metrics.py

Provides functions to compute financial risk metrics including:
- Sharpe Ratio
- Value at Risk (VaR)
- Maximum Drawdown
"""

import numpy as np
import pandas as pd

def compute_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Compute the Sharpe Ratio given a series of returns.
    Assumes daily returns and annualizes the ratio.
    """
    excess_returns = returns - risk_free_rate / 252
    std = np.std(excess_returns) + 1e-9  # avoid division by zero
    sharpe_ratio = np.mean(excess_returns) / std * np.sqrt(252)
    return sharpe_ratio

def compute_var(returns, confidence_level=0.95):
    """
    Compute Value at Risk (VaR) at the given confidence level.
    """
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def compute_max_drawdown(net_worth_series):
    """
    Compute the maximum drawdown from a series of net worth values.
    """
    roll_max = net_worth_series.cummax()
    drawdown = (net_worth_series - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown

if __name__ == "__main__":
    # Example usage with dummy data
    sample_returns = np.random.normal(0, 1, 1000)
    print("Sharpe Ratio:", compute_sharpe_ratio(sample_returns))
    print("VaR (95%):", compute_var(sample_returns))
    
    net_worth_series = pd.Series(10000 + np.cumsum(np.random.randn(1000)))
    print("Max Drawdown:", compute_max_drawdown(net_worth_series))
