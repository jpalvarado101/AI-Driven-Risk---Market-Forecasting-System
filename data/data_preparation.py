"""
data/data_preparation.py

Downloads historical stock data using yfinance for multiple tickers,
computes technical indicators and additional crisis features, and saves the processed data to CSV files.
"""

import yfinance as yf
import pandas as pd
import ta  # Technical Analysis library

def fetch_data(ticker="AAPL", period="20y", interval="1d"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

def add_technical_indicators(df):
    print("Adding technical indicators and extra features...")
    # Ensure that the 'Close' column is a 1-dimensional Series.
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()
    
    # Basic technical indicators:
    rsi_values = ta.momentum.rsi(close_series, window=14).to_numpy().flatten()
    df.loc[:, 'rsi'] = pd.Series(rsi_values, index=df.index)
    
    macd_values = ta.trend.macd(close_series).to_numpy().flatten()
    df.loc[:, 'macd'] = pd.Series(macd_values, index=df.index)
    
    sma_50 = close_series.rolling(window=50).mean().to_numpy().flatten()
    df.loc[:, 'sma_50'] = pd.Series(sma_50, index=df.index)
    
    sma_200 = close_series.rolling(window=200).mean().to_numpy().flatten()
    df.loc[:, 'sma_200'] = pd.Series(sma_200, index=df.index)
    
    # Additional Crisis Features:
    # 1. Volume Spike Indicator: Current Volume / (30-day Average Volume)
    avg_volume = df['Volume'].rolling(window=30).mean().to_numpy().flatten()
    df.loc[:, 'avg_volume'] = pd.Series(avg_volume, index=df.index)
    
    volume_values = df['Volume'].to_numpy().flatten()
    vol_spike = volume_values / avg_volume
    df.loc[:, 'vol_spike'] = pd.Series(vol_spike, index=df.index)
    
    # 2. Drawdown: (Peak Price - Current Price) / Peak Price
    peak_price = df['Close'].cummax().to_numpy().flatten()
    df.loc[:, 'peak_price'] = pd.Series(peak_price, index=df.index)
    
    close_values = df['Close'].to_numpy().flatten()
    drawdown = (peak_price - close_values) / peak_price
    df.loc[:, 'drawdown'] = pd.Series(drawdown, index=df.index)
    
    # 3. Price Z-Score (for mean reversion):
    #    Z-Score = (Current Price - 30-day Mean) / (30-day Std Dev)
    price_mean = close_series.rolling(window=30).mean().to_numpy().flatten()
    df.loc[:, 'price_mean'] = pd.Series(price_mean, index=df.index)
    
    price_std = close_series.rolling(window=30).std().to_numpy().flatten()
    df.loc[:, 'price_std'] = pd.Series(price_std, index=df.index)
    
    price_zscore = (close_values - price_mean) / price_std
    df.loc[:, 'price_zscore'] = pd.Series(price_zscore, index=df.index)
    
    # Fill missing values (backward fill any NaNs from rolling calculations)
    df = df.fillna(method='bfill')
    return df

def prepare_data_for_ticker(ticker):
    df = fetch_data(ticker=ticker)
    df = add_technical_indicators(df)
    output_path = f"./data/processed_data_{ticker}.csv"
    df.to_csv(output_path, index=False)
    print(f"Data for {ticker} prepared and saved to {output_path}")

def prepare_data_multi(tickers=["AAPL", "NVDA", "MSFT", "TSLA", "META"]):
    for ticker in tickers:
        prepare_data_for_ticker(ticker)

if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "META"]
    prepare_data_multi(tickers)
