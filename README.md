# 📈 LSTM-Based Stock Forecasting & Backtesting Pipeline

**A modular research framework for building, fine-tuning, and rigorously backtesting deep learning models for financial time series forecasting.**

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/PyTorch-ML-orange" />
  <img src="https://img.shields.io/badge/W&B-Experiment_Tracking-yellow" />
</p>

---

## 🚀 Overview

This repository demonstrates a **complete, production-grade machine learning pipeline** for daily stock price prediction, alpha signal generation, and systematic trading backtesting.

Combining:
- ✅ **Modern LSTM and Attention-LSTM architectures**
- ✅ **Rich technical indicator engineering**
- ✅ **Strong baseline benchmarking (trees, regressors)**
- ✅ **Transparent backtesting with real trade simulation**
- ✅ **Full Weights & Biases (W&B) experiment tracking**

---

## 🎓 Academic Motivation

Designed to meet **postgraduate research standards** in quantitative finance and applied machine learning:

- Trains sequence models on engineered features that capture price momentum, trend, volatility, and cyclical structure.
- Supports **stock-specific fine-tuning** and **multi-asset pretraining**.
- Evaluates prediction quality with both **statistical loss (RMSE)** and **risk-focused metrics** (directional accuracy, Sharpe ratio, drawdown).
- Provides fully auditable trade signals, charts, and logs.

---

## 💼 Industry Relevance

Built to illustrate practical steps for real-world **systematic trading signal development**:

- Converts raw forecasts into actionable **Long/Short trading signals** via robust threshold sweeps.
- Simulates realistic equity curves, entry/exit logic, and PnL paths.
- Outputs easy-to-interpret **trade logs** (CSV) for third-party review.
- Provides clear benchmark baselines for measuring ML edge.

---

## ⚙️ Pipeline Modules

| Module | Purpose |
|---------------------|---------|
| **`pretrain_general.py`** | Pretrain a general-purpose LSTM on multiple tickers to learn shared dynamics. |
| **`finetune_single.py`** | Fine-tune pretrained weights on a specific stock over any custom date window. |
| **`feature_engineering.py`** | Adds robust technical signals: SMA, EMA, RSI, MACD, Bollinger Bands, OBV, and more. |
| **`train.py`** | Core training loop with scaling, batching, loss tracking, and W&B logging. |
| **`baseline_models.py`** | Runs classical regressors (Linear, Ridge, Lasso, SVR, Trees) for benchmark comparison. |
| **`evaluate.py`** | Computes test RMSE, directional accuracy, and mean predicted signal return. |
| **`backtest_strategy.py`** | Turns predictions into realistic trade positions, simulates PnL, exports trade logs. |
| **`threshold_sweep.py`** | Sweeps thresholds to find robust Long/Short signal breakpoints. |

---

## 📊 Key Metrics & Plots

- **RMSE** – prediction error for price forecasts.
- **Directional Accuracy** – % of up/down calls correct.
- **Win Rate** – % of trades that finish profitably (Long/Short).
- **Sharpe Ratio** – return per unit risk.
- **Max Drawdown** – worst peak-to-trough loss.
- **Equity Curve** – how the strategy capital grows over time.
- **Price Chart** – marks every entry/exit directly on the asset price.

All results log to **Weights & Biases** for reproducibility and tracking.

---

## 📈 Example: Realistic Backtest

A typical run:
1. Train on any stock for any date range.
2. Predict next-day price, convert to **implied return**.
3. Apply asymmetric thresholds:
   - **Long:** if predicted return > +X%.
   - **Short:** if predicted return < –X%.
4. Simulate trade PnL, visualize real entry/exit points on the price chart.
5. Export a **trade log CSV** for transparency.

---

## ✅ Quick Start

```bash
# Fine-tune on AAPL
python finetune_single.py

# Backtest with realistic trade simulation
python backtest_strategy.py

# Sweep thresholds to find robust signal breakpoints WIP
python threshold_sweep.py
