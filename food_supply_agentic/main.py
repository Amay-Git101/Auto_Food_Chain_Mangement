"""
main.py — Full Pipeline Entry Point

Runs the complete Agentic Inventory Rebalancing system:
  1. Load & process data
  2. Forecast demand
  3. Simulate inventory
  4. Run agents (Observe → Reason → Decide)
  5. Rebalance (Optimise transfers)
  6. Print output
"""

from src.data_processing   import load_and_process
from src.demand_forecasting import forecast_demand
from src.inventory_simulation import simulate_inventory
from src.decision_engine   import run_decision_engine, decisions_to_dataframe
from src.optimization      import rebalance, print_actions


def main():
    print("\n" + "="*65)
    print("  AUTONOMOUS INVENTORY REBALANCING — AGENTIC AI PROTOTYPE")
    print("="*65 + "\n")

    # Step 1 — Data
    df = load_and_process(data_dir="data")

    # Step 2 — Forecast
    df = forecast_demand(df, window=4)

    # Step 3 — Simulate Inventory
    df = simulate_inventory(df)

    # Step 4 — Run Agent Pipeline
    decisions = run_decision_engine(df)

    # Step 5 — Rebalance
    actions = rebalance(decisions)

    # Step 6 — Output
    print_actions(actions, limit=15)

    # Save decisions to CSV
    decisions_df = decisions_to_dataframe(decisions)
    decisions_df.to_csv("data/decisions.csv", index=False)
    print("[Main] Full decision log saved → data/decisions.csv")

    print("\n[Main] Pipeline complete. ✓\n")

    return df, decisions, actions


if __name__ == "__main__":
    df, decisions, actions = main()
