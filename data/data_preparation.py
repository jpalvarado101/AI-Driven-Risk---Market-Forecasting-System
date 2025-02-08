"""
data/data_preparation.py

Downloads historical stock data using yfinance, computes technical indicators,
and saves the processed data to CSV.
"""

import yfinance as yf
import pandas as pd
import ta  # Technical Analysis library

def fetch_data(ticker="AAPL", period="5y", interval="1d"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

def add_technical_indicators(df):
    print("Adding technical indicators...")
    # Ensure that the 'Close' column is a 1-dimensional Series.
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()
    
    # Compute Relative Strength Index (RSI)
    df['rsi'] = ta.momentum.rsi(close_series, window=14)
    # Compute MACD (Moving Average Convergence Divergence)
    df['macd'] = ta.trend.macd(close_series)
    # Compute Simple Moving Averages
    df['sma_50'] = close_series.rolling(window=50).mean()
    df['sma_200'] = close_series.rolling(window=200).mean()
    # Fill missing values
    df = df.fillna(method='bfill')
    return df

def prepare_data():
    df = fetch_data()
    df = add_technical_indicators(df)
    output_path = "J:\John Alvarado\Documents\projects\AI-Driven Risk & Market Forecasting System\AI-Driven-Risk---Market-Forecasting-System\data\processed_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Data prepared and saved to {output_path}")

if __name__ == "__main__":
    prepare_data()
