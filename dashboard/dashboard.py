"""
dashboard/dashboard.py

A Streamlit dashboard for interacting with the RL trading agent API.
Users can select a ticker, input market data, receive a trading action prediction, and view risk metrics.
"""

import streamlit as st
import requests

st.title("RL Trading Agent Dashboard")

st.header("Trading Action Prediction")

st.text("Current Implementation: Assumes the new observation is the latest. It appends the new observation to the end and then computes rolling features. \n Historical Observation Issue: If you provide data from 2020, it will be treated as the most recent data point, leading to incorrect rolling calculations. \nSolution: Calculate the most recent values for the best results.")

# Allow the user to select a ticker from the supported list.
tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "META"]
selected_ticker = st.selectbox("Select Ticker", tickers)

with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)
    with col1:
        open_price = st.number_input("Open Price", value=150.0)
        high_price = st.number_input("High Price", value=155.0)
        low_price = st.number_input("Low Price", value=149.0)
        close_price = st.number_input("Close Price", value=152.0)
        volume = st.number_input("Volume", value=1000000.0)
    with col2:
        rsi = st.number_input("RSI", value=50.0)
        macd = st.number_input("MACD", value=0.0)
        sma_50 = st.number_input("SMA 50", value=150.0)
        sma_200 = st.number_input("SMA 200", value=150.0)
        balance = st.number_input("Balance", value=10000.0)
        shares_held = st.number_input("Shares Held", value=0.0)

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    payload = {
        "ticker": selected_ticker,
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price,
        "Volume": volume,
        "rsi": rsi,
        "macd": macd,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "balance": balance,
        "shares_held": shares_held
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Action for {selected_ticker}: {result['action']}")
        else:
            st.error("Error: Prediction API returned a non-200 status code.")
    except Exception as e:
        st.error(f"API request failed: {e}")

""" st.header("Risk Metrics Overview")
st.write("This section can be expanded to display computed risk metrics (e.g., Sharpe Ratio, VaR, Maximum Drawdown) over historical performance.") """
