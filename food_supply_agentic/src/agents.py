"""
agents.py — Multi-Agent Core

Each agent follows the Observe → Reason → Report pattern.
Agents do NOT just return raw numbers — they return structured signals
with classification and reasoning attached.
"""

from dataclasses import dataclass
from typing import Literal


# ── Shared Signal Types ────────────────────────────────────────────────────────

@dataclass
class DemandSignal:
    store: int
    date: str
    raw_demand: float
    level: Literal["HIGH", "NORMAL", "LOW"]
    reason: str


@dataclass
class InventorySignal:
    store: int
    date: str
    raw_inventory: float
    days_of_cover: float          # how many weeks of demand can inventory cover
    status: Literal["CRITICAL", "ADEQUATE", "EXCESS"]
    reason: str


@dataclass
class DecisionSignal:
    store: int
    date: str
    action: Literal["SHORTAGE", "SURPLUS", "STABLE"]
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    gap: float                    # units short or surplus
    reason: str


# ── Agent 1: Demand Agent ──────────────────────────────────────────────────────

class DemandAgent:
    """
    Observes predicted demand and classifies it relative to
    the store's own historical average.
    """

    HIGH_THRESHOLD   = 1.10   # 10% above store avg → HIGH
    LOW_THRESHOLD    = 0.90   # 10% below store avg → LOW

    def __init__(self):
        self._store_averages: dict[int, float] = {}

    def fit(self, df):
        """Pre-compute per-store average demand."""
        self._store_averages = (
            df.groupby("Store")["predicted_demand"].mean().to_dict()
        )
        print(f"[DemandAgent] Fitted on {len(self._store_averages)} stores.")

    def analyze(self, row) -> DemandSignal:
        store  = row["Store"]
        demand = row["predicted_demand"]
        date   = str(row["Date"])[:10]

        avg    = self._store_averages.get(store, demand)
        ratio  = demand / avg if avg > 0 else 1.0

        if ratio >= self.HIGH_THRESHOLD:
            level  = "HIGH"
            reason = f"Demand {ratio:.1%} above store average ({avg:,.0f})"
        elif ratio <= self.LOW_THRESHOLD:
            level  = "LOW"
            reason = f"Demand {ratio:.1%} below store average ({avg:,.0f})"
        else:
            level  = "NORMAL"
            reason = f"Demand within normal range of store average ({avg:,.0f})"

        return DemandSignal(
            store=store, date=date,
            raw_demand=demand, level=level, reason=reason
        )


# ── Agent 2: Inventory Agent ───────────────────────────────────────────────────

class InventoryAgent:
    """
    Observes inventory levels and evaluates stock health
    in terms of weeks-of-cover (how long stock will last).
    """

    CRITICAL_COVER = 0.85   # less than 0.85 weeks → CRITICAL
    EXCESS_COVER   = 1.20   # more than 1.20 weeks → EXCESS

    def analyze(self, row) -> InventorySignal:
        store     = row["Store"]
        inventory = row["inventory"]
        demand    = row["predicted_demand"]
        date      = str(row["Date"])[:10]

        days_of_cover = (inventory / demand) if demand > 0 else 1.0

        if days_of_cover < self.CRITICAL_COVER:
            status = "CRITICAL"
            reason = f"Only {days_of_cover:.2f} weeks of stock remaining"
        elif days_of_cover > self.EXCESS_COVER:
            status = "EXCESS"
            reason = f"{days_of_cover:.2f} weeks of excess stock — risk of wastage"
        else:
            status = "ADEQUATE"
            reason = f"Stock covers {days_of_cover:.2f} weeks — healthy level"

        return InventorySignal(
            store=store, date=date,
            raw_inventory=inventory,
            days_of_cover=round(days_of_cover, 3),
            status=status, reason=reason
        )


# ── Agent 3: Decision Agent ────────────────────────────────────────────────────

class DecisionAgent:
    """
    Receives signals from DemandAgent and InventoryAgent,
    reasons about the combined situation, and produces
    a structured decision with urgency level.
    """

    # Decision matrix: (demand_level, inventory_status) → (action, urgency)
    DECISION_MATRIX = {
        ("HIGH",   "CRITICAL"): ("SHORTAGE", "HIGH"),
        ("HIGH",   "ADEQUATE"): ("SHORTAGE", "MEDIUM"),
        ("HIGH",   "EXCESS"):   ("STABLE",   "LOW"),
        ("NORMAL", "CRITICAL"): ("SHORTAGE", "MEDIUM"),
        ("NORMAL", "ADEQUATE"): ("STABLE",   "LOW"),
        ("NORMAL", "EXCESS"):   ("SURPLUS",  "LOW"),
        ("LOW",    "CRITICAL"): ("STABLE",   "LOW"),
        ("LOW",    "ADEQUATE"): ("SURPLUS",  "LOW"),
        ("LOW",    "EXCESS"):   ("SURPLUS",  "HIGH"),
    }

    def decide(self, demand_signal: DemandSignal,
               inventory_signal: InventorySignal) -> DecisionSignal:

        key    = (demand_signal.level, inventory_signal.status)
        action, urgency = self.DECISION_MATRIX.get(key, ("STABLE", "LOW"))

        gap = abs(inventory_signal.raw_inventory - demand_signal.raw_demand)

        reason = (
            f"Demand is {demand_signal.level} ({demand_signal.reason}). "
            f"Inventory is {inventory_signal.status} ({inventory_signal.reason}). "
            f"→ Action: {action} with {urgency} urgency."
        )

        return DecisionSignal(
            store=demand_signal.store,
            date=demand_signal.date,
            action=action,
            urgency=urgency,
            gap=round(gap, 2),
            reason=reason
        )
