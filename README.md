# LSTM-Stock Forecasting Pipeline

This repository implements a fully causal, denoising-driven, modular pipeline for next-day stock return forecasting, with end-to-end support for training, evaluation, live prediction, and incremental fine-tuning.

---

## 🚀 Project Overview

**Goal:** Forecast next-day stock returns (distributional forecasting) using a CNN→LSTM→Attention hybrid architecture, advanced denoising, and return distribution modeling.

**Current State:**

* **Baseline LSTM** with EMA/Kalman denoising and lagged technical indicators.
* **Full train → holdout evaluation** pipeline with W\&B integration and plotting.
* **Daily live prediction** at market open, logging predictions and live directional accuracy.
* **Incremental fine-tuning** after market close for continuous adaptation.
* **Feature selection** via `CONFIG["feature_cols"]` and support for custom indicators.
* **Zero data leakage**: causal denoising, shifted indicators, placeholder alignment.

---

## 📁 Repository Structure

```
├── config.py             # Global configuration and hyperparameters
├── data/
│   ├── alphavantage.py   # Historical & latest-open fetchers
│   ├── split.py          # Sequence creation
│   └── scaler.py         # Data scaling and persistence
├── denoising/
│   └── denoise.py        # EMA / Kalman filters (causal)
├── features/
│   └── indicators.py     # Lagged technical & return-based features
├── models/
│   └── lstm.py           # LSTM regression model
├── training/
│   └── trainer.py        # Train loop, W&B, history
├── evaluation/
│   ├── holdout.py        # Holdout-only RMSE/DA
│   └── live.py           # Daily live predict & live DA
├── finetune/
│   └── finetuner.py      # Incremental fine-tuning
├── utils/
│   ├── metrics.py        # RMSE & directional accuracy
│   └── plots.py          # Matplotlib plot helpers
├── scripts/
│   ├── train.py          # Full retrain + holdout
│   ├── evaluate.py       # Holdout eval + plotting
│   ├── live_pipeline.py  # Live prediction entrypoint
│   ├── plot_live.py      # Plot live pred vs actual
│   └── finetune.py       # Daily fine-tuning
└── artifacts/            # Models, scalers, live_results.csv
```

---

## ⚙️ Configuration (`config.py`)

Centralizes all parameters:

```python
CONFIG = {
    # Data & sequences
    "start_date": "2019-01-01",
    "end_date": "2025-08-01",
    "sequence_length": 60,
    "holdout_size": 250,
    "default_symbol": "AAPL",

    # Model hyperparameters
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 50,

    # AlphaVantage
    "av_api_key": "YOUR_API_KEY",
    "denoising_method": "kalman",      # raw | ema | kalman

    # Features (flip features on/off here)
    "feature_cols": [
        "close", "SMA_20", "SMA_50", "RSI_14", "Volatility",
        "Return_1", "Return_5", "Volume_Change", "Return_Skew"
    ],

    # Artifacts
    "model_dir": "artifacts/models",
    "scaler_dir": "artifacts/scalers",
    "live_csv": "artifacts/live_results.csv",

    # W&B
    "wandb_project": "stock_forecasting_live",
}
```

---

## 📈 Training & Evaluation

### Full Retrain

```bash
python scripts/train.py
```

* Fetches history, denoises, computes indicators.
* Builds 60-day sequences, splits holdout.
* Scales data, trains LSTM for `epochs`, logs train/holdout metrics to W\&B.
* Plots training & holdout curves locally.

### Holdout-Only Evaluation

```bash
python scripts/evaluate.py
```

* Reuses saved model & scalers.
* Computes holdout RMSE & DA.
* Plots actual vs. predicted on holdout.

---

## 🌐 Live Prediction Pipeline

```bash
python scripts/live_pipeline.py
```

* Scheduled **Mon–Fri at 16:31 Istanbul** after U.S. open.
* Fetches history through yesterday, denoises, appends today’s open.
* Builds input window, logs last 60 real days.
* Loads model & scalers, predicts today’s close, appends to `live_results.csv`.
* Computes live directional accuracy over past dates, logs to W\&B.
* **Plot live**:

  ```bash
  python scripts/plot_live.py
  ```

---

## 🔄 Incremental Fine-Tuning

```bash
python scripts/finetune.py
```

* Runs **Mon–Fri at 23:00 Istanbul** after market close.
* Fetches last 2× holdout window, denoises, indicators.
* Splits recent holdout, scales with existing scalers.
* Fine-tunes model for a few epochs at low LR, saves updated model.

---

## 🛠️ Customization & Extensions

* **Feature Selection:** update `CONFIG["feature_cols"]` to add/remove signals.
* **Ticker Override:** add `--symbol` flag to scripts to target any stock.
* **Denoising:** switch between `raw`, `ema`, `kalman` in `config.py`.
* **Directional Optimization:** convert to classification or multi-task in `models/lstm.py` & `trainer.py`.

---

## 📝 Best Practices

* **No data leakage:** causal filters, lagged indicators, placeholder alignment.
* **W\&B Logging:** track train/holdout/live metrics and model versions.
* **Cron Scheduling:** ensure scripts run only on trading days, after open/close.

---

## 📚 References

* [PyTorch LSTM Docs](https://pytorch.org/docs/stable/nn.html#lstm)
* [AlphaVantage API](https://www.alphavantage.co/documentation/)
* [pykalman](https://pykalman.github.io)

---

Happy forecasting! 🚀
