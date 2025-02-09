# 📈 AI-Driven Risk & Market Forecasting System 🚀

## 🔍 Project Overview

This project implements a **Reinforcement Learning (RL) agent for financial market simulation and risk prediction** using **Proximal Policy Optimization (PPO)**. The AI-driven trading system leverages real market data, technical indicators, and risk management metrics to optimize trading decisions while balancing profitability and risk.

✅ **Key Features:**

- **Reinforcement Learning-based Trading Agent** (PPO from `stable-baselines3`)
- **Real-time Market Data Processing** (Yahoo Finance API)
- **Technical Indicators for Market Analysis** (RSI, MACD, SMA50, SMA200)
- **Risk Management Metrics** (Sharpe Ratio, Value at Risk (VaR), Maximum Drawdown)
- **Scalable REST API** for real-time predictions (`FastAPI`, `Docker`)
- **Interactive Trading Dashboard** (`Streamlit`)

---

[![Watch the video](https://github.com/jpalvarado101/AI-Driven-Risk---Market-Forecasting-System/blob/main/maxresdefaultplay.jpg)](https://youtu.be/CHR2EzLQ6k8)

### [Watch this video on YouTube](https://youtu.be/CHR2EzLQ6k8)

## 🎯 Project Architecture

```
📂 AI-Driven Risk & Market Forecasting System
│── data/                  # Data collection & preprocessing
│   ├── data_preparation.py  # Fetches stock data & computes indicators
│   ├── processed_data.csv   # Processed historical data
│
│── env/                   # Custom Gym trading environment
│   ├── trading_env.py      # Trading environment for RL agent
│
│── models/                # Reinforcement Learning model
│   ├── rl_agent.py         # PPO agent training & evaluation
│
│── risk_management/       # Financial risk metrics
│   ├── risk_metrics.py     # Sharpe Ratio, VaR, Drawdown computation
│
│── deployment/            # API Deployment
│   ├── api.py              # FastAPI-based inference API
│   ├── Dockerfile          # Dockerization of the API
│
│── dashboard/             # Trading visualization dashboard
│   ├── dashboard.py        # Streamlit-based frontend
│
│── utils/                 # Helper functions
│   ├── helpers.py          # Reward plotting functions
│
│── train.py               # RL training entry point
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

---

## 📊 Data Collection & Processing

The `data/data_preparation.py` script fetches real-time stock market data using `yfinance` and computes various **technical indicators**:

- **RSI (Relative Strength Index)** – Measures momentum to detect overbought/oversold conditions.
- **MACD (Moving Average Convergence Divergence)** – Identifies trend direction.
- **SMA50 & SMA200 (Simple Moving Averages)** – Used for trend analysis.

**Run Data Processing:**

```bash
python data/data_preparation.py
```

---

## 🏦 Trading Environment

A **custom Gym environment** (`env/trading_env.py`) simulates a stock trading environment:

- **Observation Space**: Stock prices, indicators, account balance, shares held.
- **Action Space**: {0: Hold, 1: Buy, 2: Sell}.
- **Reward Function**: Change in net worth relative to initial balance.

---

## 🏆 Reinforcement Learning Agent

The RL agent uses **Proximal Policy Optimization (PPO)** from `stable-baselines3` to learn optimal trading strategies.

**Train the Agent:**

```bash
python train.py
```

- Saves trained model to `models/ppo_trading_agent.zip`.

**Load Trained Model:**

```python
from models.rl_agent import load_agent
agent = load_agent("models/ppo_trading_agent.zip")
```

---

## 📉 Risk Management Metrics

The `risk_management/risk_metrics.py` module computes essential **financial risk metrics**:

- **Sharpe Ratio** – Measures return relative to risk.
- **Value at Risk (VaR)** – Estimates maximum loss at a given confidence level.
- **Maximum Drawdown** – Tracks worst peak-to-trough decline.

Example usage:

```python
from risk_management.risk_metrics import compute_sharpe_ratio, compute_var, compute_max_drawdown
returns = [0.01, -0.02, 0.015, 0.03]
print("Sharpe Ratio:", compute_sharpe_ratio(returns))
print("VaR (95% confidence):", compute_var(returns))
```

---

## 🖥️ API Deployment

A **FastAPI-powered REST API** allows real-time trading predictions.

**Run the API Locally:**

```bash
uvicorn deployment.api:app --reload
```

**Test API Endpoint:**

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
 -d '{"Open": 150, "High": 155, "Low": 148, "Close": 152, "Volume": 1000000,
 "rsi": 50, "macd": 0.1, "sma_50": 150, "sma_200": 149, "balance": 10000, "shares_held": 0}'
```

**Docker Deployment:**

```bash
docker build -t financial-rl-api .
docker run -p 8000:8000 financial-rl-api
```

---

## 📊 Interactive Dashboard

A **Streamlit dashboard** allows users to:

- **Input market data & get trading predictions**
- **View risk metrics & trading performance**

**Launch Dashboard:**

```bash
streamlit run dashboard/dashboard.py
```

---

## 🚀 Getting Started

### **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2️⃣ Train the RL Agent**

```bash
python train.py
```

### **3️⃣ Deploy the API**

```bash
uvicorn deployment.api:app --reload
```

### **4️⃣ Run the Dashboard**

```bash
streamlit run dashboard/dashboard.py
```

---

## 🛠️ Technologies Used

- **Python** (3.8+)
- **Stable-Baselines3** (Reinforcement Learning)
- **Gym** (Custom Trading Environment)
- **Pandas & NumPy** (Market Data Processing)
- **FastAPI** (API Deployment)
- **Streamlit** (Dashboard UI)
- **Docker** (Containerization)
- **Yahoo Finance API** (Stock Data Retrieval)

---

## 🌟 Future Improvements

🔹 **Train on multiple stocks & ETFs** to generalize performance.  
🔹 **Add sentiment analysis (Twitter, news data)** for smarter decision-making.  
🔹 **Optimize model execution using TensorRT or JAX for faster inference.**  
🔹 **Integrate Bayesian Optimization for hyperparameter tuning.**


---

## 📜 License

This project is under the **Apache 2.0 License**.

---

## 📧 Contact

🚀 **Developer:** John Alvarado
📩 **Email:** contact@johnferreralvarado.com
🌐 **LinkedIn:** [\[My LinkedIn\] ](https://www.linkedin.com/in/johnfalvarado/)

---

## ⭐ Star this Repo if You Found It Useful! 🌟
