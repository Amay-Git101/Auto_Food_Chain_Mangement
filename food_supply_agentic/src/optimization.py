"""
optimization.py — Inventory Rebalancing Engine

Takes DecisionSignals and computes concrete transfer actions:
  - Groups by week (same date = same rebalancing window)
  - Matches SHORTAGE stores with SURPLUS stores
  - Prioritises by urgency (HIGH first)
  - Greedy matching: fill highest-urgency shortage first
"""

from dataclasses import dataclass
from collections import defaultdict
from src.agents import DecisionSignal


URGENCY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


@dataclass
class TransferAction:
    date: str
    from_store: int
    to_store: int
    units: float
    urgency: str
    note: str


def rebalance(decisions: list[DecisionSignal]) -> list[TransferAction]:
    """
    Compute inventory transfer actions from a list of DecisionSignals.

    Groups decisions by date, then within each week:
      - Sorts shortages by urgency (HIGH → MEDIUM → LOW)
      - Matches with available surplus stores
      - Transfers as many units as possible

    Returns:
        List of TransferAction objects.
    """
    print("[Optimizer] Running rebalancing engine...")

    # Group by date
    by_date: dict[str, list[DecisionSignal]] = defaultdict(list)
    for d in decisions:
        by_date[d.date].append(d)

    all_actions: list[TransferAction] = []

    for date, week_decisions in sorted(by_date.items()):

        shortages = sorted(
            [d for d in week_decisions if d.action == "SHORTAGE"],
            key=lambda d: URGENCY_ORDER[d.urgency]
        )
        surpluses = [d for d in week_decisions if d.action == "SURPLUS"]

        if not shortages or not surpluses:
            continue

        # Work on mutable copies of gap values
        shortage_pool = [[s.store, s.gap, s.urgency] for s in shortages]
        surplus_pool  = [[s.store, s.gap] for s in surpluses]

        for shortage in shortage_pool:
            s_store, s_need, s_urgency = shortage

            for surplus in surplus_pool:
                sp_store, sp_available = surplus

                if s_need <= 0 or sp_available <= 0:
                    continue

                transfer = min(s_need, sp_available)

                all_actions.append(TransferAction(
                    date=date,
                    from_store=sp_store,
                    to_store=s_store,
                    units=round(transfer, 2),
                    urgency=s_urgency,
                    note=(
                        f"[{s_urgency} URGENCY] Transfer {transfer:,.0f} units "
                        f"from Store {sp_store} → Store {s_store} on {date}"
                    )
                ))

                # Update remaining gaps
                shortage[1] -= transfer
                surplus[1]  -= transfer

    print(f"[Optimizer] Generated {len(all_actions)} transfer actions.")
    return all_actions


def print_actions(actions: list[TransferAction], limit: int = 15):
    """Pretty-print transfer actions to console."""
    print(f"\n{'='*65}")
    print(f"  INVENTORY REBALANCING ACTIONS  (showing top {limit})")
    print(f"{'='*65}")

    if not actions:
        print("  No transfers needed.")
        return

    for i, a in enumerate(actions[:limit], 1):
        print(f"  {i:2}. {a.note}")

    if len(actions) > limit:
        print(f"\n  ... and {len(actions) - limit} more actions.")

    print(f"{'='*65}\n")
