import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- CONFIGURATION ----------------
# Example: S&P 500 tickers – replace or extend with your own list
# You can also load from a CSV: pd.read_csv("watchlist.csv")["ticker"].tolist()
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "PYPL",
    "BAC", "ADBE", "CRM", "NFLX", "INTC", "CSCO", "PFE", "KO",
    "PEP", "MRK", "ABT", "TMO", "COST", "AVGO", "TXN", "LLY",
    "NKE", "MCD", "DHR", "NEE", "PM", "UNP", "RTX"
]

# Lookback for data (must be > 200 trading days)
LOOKBACK_DAYS = 400

# Minimum average daily volume (optional filter)
MIN_AVG_VOLUME = 500_000

# ---------------- HELPERS ----------------

def fetch_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period=f"{LOOKBACK_DAYS}d", progress=False)
            if df.empty:
                continue
            # Handle new yfinance multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            data[t] = df
        except Exception:
            continue
    return data

def compute_sma(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    return df

def filter_above_sma(df: pd.DataFrame) -> bool:
    if len(df) < 200:
        return False
    last = df.iloc[-1]
    return (last["Close"] > last["SMA20"]) and (last["Close"] > last["SMA200"])

def extra_metrics(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    sma20 = last["SMA20"]
    sma200 = last["SMA200"]
    close = last["Close"]

    dist_20 = (close - sma20) / sma20 * 100
    dist_200 = (close - sma200) / sma200 * 100

    # 20-day slope (simple linear regression on SMA20)
    x = np.arange(20)
    y = df["SMA20"].iloc[-20:].values
    slope_20 = np.polyfit(x, y, 1)[0] if len(y) == 20 else np.nan

    # 200-day slope
    x2 = np.arange(200)
    y2 = df["SMA200"].iloc[-200:].values
    slope_200 = np.polyfit(x2, y2, 1)[0] if len(y2) == 200 else np.nan

    avg_vol = df["Volume"].iloc[-20:].mean()

    return {
        "close": close,
        "sma20": sma20,
        "sma200": sma200,
        "dist_from_20_pct": dist_20,
        "dist_from_200_pct": dist_200,
        "sma20_slope": slope_20,
        "sma200_slope": slope_200,
        "avg_volume_20d": avg_vol,
    }

# ---------------- MAIN ----------------

def run_bot(tickers: list[str], output_file: str = "sma_screen_results.csv"):
    print("Fetching data...")
    data = fetch_data(tickers)
    print(f"Downloaded data for {len(data)} tickers.")

    results = []

    for ticker, df in data.items():
        df = compute_sma(df)
        if not filter_above_sma(df):
            continue

        metrics = extra_metrics(df)
        if metrics["avg_volume_20d"] < MIN_AVG_VOLUME:
            continue

        row = {"ticker": ticker}
        row.update(metrics)
        results.append(row)

    if not results:
        print("No stocks matched your criteria.")
        return

    out_df = pd.DataFrame(results)
    # Sort by distance from 200-SMA (stronger uptrend first), or change as you like
    out_df = out_df.sort_values("dist_from_200_pct", ascending=False)

    out_df.to_csv(output_file, index=False)
    print(f"Saved {len(out_df)} stocks to {output_file}")
    print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    run_bot(TICKERS)
