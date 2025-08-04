"""
Plot all live predictions vs actual closes.
"""
import pandas as pd
from utils.plots import plot_pred_vs_actual
from config import CONFIG

if __name__ == "__main__":
    df = pd.read_csv(CONFIG["live_csv"], parse_dates=["date"])
    # fetch actual closes for the recorded dates
    from data.alphavantage import fetch_historical
    hist = fetch_historical(
        CONFIG["default_symbol"],
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d")
    )
    actuals = [hist.loc[pd.Timestamp(d), "close"] for d in df["date"]]
    plot_pred_vs_actual(df["date"], df["pred"].values, actuals, title="Live: Actual vs Predicted")
