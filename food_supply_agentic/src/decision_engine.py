"""
decision_engine.py — Orchestrates all agents across the full dataset.

This is where the "agentic loop" runs:
  For every store × week:
    1. DemandAgent observes and classifies demand
    2. InventoryAgent observes and evaluates stock
    3. DecisionAgent reasons over both signals → produces decision
"""

import pandas as pd
from src.agents import DemandAgent, InventoryAgent, DecisionAgent, DecisionSignal


def run_decision_engine(df: pd.DataFrame) -> list[DecisionSignal]:
    """
    Run all three agents across the full dataframe.

    Returns:
        List of DecisionSignal objects — one per row.
    """
    print("\n[DecisionEngine] Initialising agents...")

    demand_agent    = DemandAgent()
    inventory_agent = InventoryAgent()
    decision_agent  = DecisionAgent()

    # Fit demand agent on historical averages
    demand_agent.fit(df)

    print("[DecisionEngine] Running agentic loop...\n")

    decisions: list[DecisionSignal] = []

    for _, row in df.iterrows():
        d_signal = demand_agent.analyze(row)
        i_signal = inventory_agent.analyze(row)
        decision = decision_agent.decide(d_signal, i_signal)
        decisions.append(decision)

    # Summary stats
    shortage_count = sum(1 for d in decisions if d.action == "SHORTAGE")
    surplus_count  = sum(1 for d in decisions if d.action == "SURPLUS")
    stable_count   = sum(1 for d in decisions if d.action == "STABLE")
    high_urgency   = sum(1 for d in decisions if d.urgency == "HIGH")

    print(f"[DecisionEngine] Results across {len(decisions)} store-weeks:")
    print(f"  SHORTAGE : {shortage_count}")
    print(f"  SURPLUS  : {surplus_count}")
    print(f"  STABLE   : {stable_count}")
    print(f"  HIGH urgency alerts : {high_urgency}\n")

    return decisions


def decisions_to_dataframe(decisions: list[DecisionSignal]) -> pd.DataFrame:
    """Convert list of DecisionSignals to a flat DataFrame for display/export."""
    return pd.DataFrame([{
        "Store":     d.store,
        "Date":      d.date,
        "Action":    d.action,
        "Urgency":   d.urgency,
        "Gap_Units": d.gap,
        "Reason":    d.reason,
    } for d in decisions])
