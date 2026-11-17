"""Single-model metrics only; no cross-model statistics here."""
import numpy as np

from track2.config import MIN_ENERGY, CAPACITY
from track2.physics import settle_day


def price_daily(pred, truth):
    return np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2, axis=1))


def storage_daily(plan, truth):
    """Settle day by day from 5% SOC; an infeasible day executes nothing and keeps its SOC."""
    energy, rows = MIN_ENERGY, []
    for q, prices in zip(np.asarray(plan, dtype=float), np.asarray(truth, dtype=float)):
        result = settle_day(q, prices, energy)
        rows.append(dict(revenue=result['revenue'], valid=result['valid'],
                         reason=result['reason'] or '',
                         soc_start=energy / CAPACITY, soc_end=result['end_energy'] / CAPACITY))
        energy = result['end_energy']
    return rows
