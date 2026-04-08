import pandas as pd


def forecast_demand(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Predict next-week demand using a rolling average per store.

    Args:
        df     : Processed dataframe with Weekly_Sales column.
        window : Number of past weeks to average over.

    Returns:
        df with new column 'predicted_demand'.
    """
    print(f"[DemandForecaster] Running {window}-week rolling average forecast...")

    df = df.copy()

    # Rolling mean per store (shift by 1 so we don't use current week)
    df["predicted_demand"] = (
        df.groupby("Store")["Weekly_Sales"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Fill any remaining NaNs with store-level mean
    store_means = df.groupby("Store")["Weekly_Sales"].transform("mean")
    df["predicted_demand"] = df["predicted_demand"].fillna(store_means)

    df["predicted_demand"] = df["predicted_demand"].round(2)

    print(f"[DemandForecaster] Forecast complete. "
          f"Avg predicted demand: {df['predicted_demand'].mean():,.0f} units")

    return df
