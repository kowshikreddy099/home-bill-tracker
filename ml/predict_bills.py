"""
Predict next month's household bills with machine learning.

For each bill (rent, electricity, ...) it:
  1. builds features from history (last 1-3 months, 3-month average, trend, month of year)
  2. compares several algorithms with walk-forward validation (train on the past, test on the next month)
  3. picks the algorithm with the lowest error and predicts next month

Usage:  python predict_bills.py [bills.csv]
CSV format: first column "month" (YYYY-MM), then one column per bill in rupees.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent   # folder this script lives in
MIN_TRAIN = 12          # months of history needed before we start validating


def make_features(series: pd.Series) -> pd.DataFrame:
    """Turn a monthly series into a feature table. Row t describes month t using only older data."""
    df = pd.DataFrame({"y": series})
    df["lag1"] = series.shift(1)
    df["lag2"] = series.shift(2)
    df["lag3"] = series.shift(3)
    df["avg3"] = series.shift(1).rolling(3).mean()
    df["trend"] = np.arange(len(series))
    month = series.index.month
    df["m_sin"] = np.sin(2 * np.pi * month / 12)     # month of year as a circle,
    df["m_cos"] = np.cos(2 * np.pi * month / 12)     # so December is next to January
    return df


FEATURES = ["lag1", "lag2", "lag3", "avg3", "trend", "m_sin", "m_cos"]

MODELS = {
    "Naive (last month)": None,
    "Ridge regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random forest": lambda: RandomForestRegressor(n_estimators=200, random_state=0),
    "Gradient boosting": lambda: GradientBoostingRegressor(random_state=0),
}


def forecast_upper_bound(history: pd.Series) -> float:
    """Return a conservative, outlier-resistant upper bound for one-month forecasts.

    A single unusually large bill (for example, a repair included in an electricity
    bill) should not make a recurring-bill estimate jump by the same amount. The
    median and median absolute deviation (MAD) of the latest six observations are
    robust to one such value, unlike a mean or maximum.
    """
    recent = history.dropna().to_numpy(dtype=float)[-6:]
    if len(recent) == 0:
        return 0.0

    middle = float(np.median(recent))
    mad = float(np.median(np.abs(recent - middle)))
    # Leave room for normal month-to-month variation even when the recent bills
    # are identical. Three robust standard deviations is deliberately generous.
    spread = max(abs(middle) * 0.15, 1.4826 * mad)
    return max(0.0, middle + 3 * spread)


def safe_forecast(prediction: float, history: pd.Series) -> float:
    """Keep forecasts finite, non-negative, and within the recent robust range."""
    if not np.isfinite(prediction):
        prediction = float(history.dropna().iloc[-1])
    return float(np.clip(prediction, 0.0, forecast_upper_bound(history)))


def walk_forward_mae(df: pd.DataFrame, make_model) -> float:
    errors = []
    for t in range(MIN_TRAIN, len(df)):
        train, test = df.iloc[:t], df.iloc[[t]]
        if make_model is None:
            pred = test["lag1"].iloc[0]
        else:
            model = make_model()
            model.fit(train[FEATURES], train["y"])
            pred = model.predict(test[FEATURES])[0]
        # Score the same protected forecast that will be shown to the user.
        # Otherwise an unbounded model can win validation but produce a
        # disproportionately high final estimate.
        pred = safe_forecast(float(pred), train["y"])
        errors.append(abs(pred - test["y"].iloc[0]))
    return float(np.mean(errors))


def predict_next(series: pd.Series):
    series = series.dropna()
    if series.empty:
        raise ValueError("At least one bill amount is required to make a forecast.")

    # There are not enough complete lag rows to score a model until 16 months
    # are present. Repeating the latest bill is transparent and reliable here.
    if len(series) < MIN_TRAIN + 4:
        name = "Naive (last month)"
        return safe_forecast(float(series.iloc[-1]), series), name, {name: 0.0}

    # Add one empty future row so the same feature code builds next month's features.
    next_period = series.index[-1] + 1
    extended = pd.concat([series, pd.Series([np.nan], index=pd.PeriodIndex([next_period]))])
    feats = make_features(extended)
    hist, future = feats.iloc[:-1].dropna(), feats.iloc[[-1]]

    scores = {name: walk_forward_mae(hist, mk) for name, mk in MODELS.items()}
    best = min(scores, key=scores.get)

    if MODELS[best] is None:
        pred = float(future["lag1"].iloc[0])
    else:
        model = MODELS[best]()
        model.fit(hist[FEATURES], hist["y"])
        pred = float(model.predict(future[FEATURES])[0])
    return safe_forecast(pred, hist["y"]), best, scores


def main(path: str):
    data = pd.read_csv(path)
    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data = data.set_index("month").sort_index()
    next_month = data.index[-1] + 1

    results = []
    for bill in data.columns:
        pred, best, scores = predict_next(data[bill])
        results.append({
            "bill": bill,
            "this_month": data[bill].iloc[-1],
            "next_month_estimate": round(pred, 2),
            "best_model": best,
            "avg_error": round(scores[best], 2),
        })
        print(f"\n{bill}")
        for name, mae in sorted(scores.items(), key=lambda kv: kv[1]):
            print(f"  {name:<20} avg error Rs {mae:,.2f}" + ("   <- chosen" if name == best else ""))

    out = pd.DataFrame(results)
    total_now, total_next = out["this_month"].sum(), out["next_month_estimate"].sum()
    print(f"\n=== Estimate for {next_month} ===")
    print(out.to_string(index=False))
    print(f"\nThis month total : Rs {total_now:,.2f}")
    print(f"Next month total : Rs {total_next:,.2f}  ({total_next - total_now:+,.2f})")
    out.to_csv(HERE / "next_month_prediction.csv", index=False)
    print("\nSaved next_month_prediction.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "bills.csv"))
