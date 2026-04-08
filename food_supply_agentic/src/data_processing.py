import pandas as pd
import numpy as np
import os

def load_and_process(data_dir="data"):
    """
    Load and merge Walmart dataset files.
    If real CSVs not found, generates synthetic data for prototyping.
    """

    train_path    = os.path.join(data_dir, "train.csv")
    features_path = os.path.join(data_dir, "features.csv")
    stores_path   = os.path.join(data_dir, "stores.csv")

    # ── Real data path ──────────────────────────────────────────────
    if os.path.exists(train_path):
        print("[DataProcessor] Loading real Walmart dataset...")

        train    = pd.read_csv(train_path)
        features = pd.read_csv(features_path)
        stores   = pd.read_csv(stores_path)

        # Merge
        df = train.merge(features, on=["Store", "Date"], how="left")
        df = df.merge(stores, on="Store", how="left")

        # Keep useful columns
        cols = ["Store", "Date", "Weekly_Sales", "Temperature", "Fuel_Price"]
        df = df[[c for c in cols if c in df.columns]]

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

        # Drop rows with missing sales
        df = df.dropna(subset=["Weekly_Sales"])

    # ── Synthetic data fallback ──────────────────────────────────────
    else:
        print("[DataProcessor] Real data not found. Generating synthetic data...")
        df = _generate_synthetic_data()

    # Save processed file
    out_path = os.path.join(data_dir, "processed_data.csv")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DataProcessor] Saved processed data → {out_path}  ({len(df)} rows)")

    return df


def _generate_synthetic_data(n_stores=5, n_weeks=52):
    """Generate realistic synthetic Walmart-style data."""
    np.random.seed(42)
    records = []

    base_sales = {1: 25000, 2: 18000, 3: 30000, 4: 22000, 5: 15000}

    for store in range(1, n_stores + 1):
        base = base_sales.get(store, 20000)
        for week in range(n_weeks):
            date = pd.Timestamp("2012-01-01") + pd.Timedelta(weeks=week)

            # Seasonal trend + noise
            seasonal   = 1 + 0.15 * np.sin(2 * np.pi * week / 52)
            sales      = base * seasonal * np.random.uniform(0.85, 1.15)
            temperature = 60 + 20 * np.sin(2 * np.pi * week / 52) + np.random.normal(0, 5)
            fuel_price  = 3.5 + np.random.uniform(-0.3, 0.3)

            records.append({
                "Store":        store,
                "Date":         date,
                "Weekly_Sales": round(sales, 2),
                "Temperature":  round(temperature, 2),
                "Fuel_Price":   round(fuel_price, 3),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df
