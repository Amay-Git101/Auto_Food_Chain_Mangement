import pandas as pd
import numpy as np


def simulate_inventory(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Simulate inventory levels per store per week.

    Since the Walmart dataset has no real inventory column,
    we model it as predicted_demand × a store-specific stock factor.

    Stock factor ranges:
      - Some stores are well-stocked  (factor > 1.0)
      - Some stores are under-stocked (factor < 1.0)
      - Adds weekly noise to make it realistic

    Returns:
        df with new column 'inventory'.
    """
    print("[InventorySimulator] Simulating inventory levels...")

    np.random.seed(seed)
    df = df.copy()

    stores = df["Store"].unique()

    # Assign a base stock factor per store (some stores chronically under/over stocked)
    store_factors = {
        store: np.random.uniform(0.75, 1.30)
        for store in stores
    }

    # Build inventory column
    inventory_values = []
    for _, row in df.iterrows():
        factor     = store_factors[row["Store"]]
        weekly_noise = np.random.uniform(0.90, 1.10)
        inv        = row["predicted_demand"] * factor * weekly_noise
        inventory_values.append(round(inv, 2))

    df["inventory"] = inventory_values

    # Summary
    for store in sorted(stores):
        sf = store_factors[store]
        label = "OVER-STOCKED" if sf > 1.05 else ("UNDER-STOCKED" if sf < 0.95 else "BALANCED")
        print(f"  Store {store}: stock factor = {sf:.2f}  [{label}]")

    print("[InventorySimulator] Inventory simulation complete.")
    return df
