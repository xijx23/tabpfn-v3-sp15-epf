import numpy as np

from track2.config import MIN_ENERGY, MAX_ENERGY, MAX_POWER, EFFICIENCY


def validate_schedule(q, initial_energy, expected_hours=24):
    """Independent net-power check, including all H+1 energy states."""
    q = np.asarray(q, dtype=float)
    energy = [float(initial_energy)]
    reason = None
    if q.ndim != 1 or len(q) != expected_hours or not np.isfinite(q).all():
        reason = 'Expected finite one-dimensional hourly powers'
    elif not np.isfinite(initial_energy) or not MIN_ENERGY <= initial_energy <= MAX_ENERGY:
        reason = 'Invalid initial energy'
    else:
        for t, power in enumerate(q):
            energy.append(energy[-1] - (power * EFFICIENCY if power <= 0 else power / EFFICIENCY))
            if abs(power) > MAX_POWER + 1e-10 and reason is None:
                reason = f'Power limit at hour {t + 1}'
            if not MIN_ENERGY - 1e-10 <= energy[-1] <= MAX_ENERGY + 1e-10 and reason is None:
                reason = f'Energy limit at hour {t + 1}'
    states = np.asarray(energy)
    violation = float(max(0., np.max(MIN_ENERGY - states), np.max(states - MAX_ENERGY)))
    return dict(valid=reason is None, energy=states, reason=reason, max_energy_violation=violation)


def quantize_schedule(q, initial_energy):
    """Project sequentially into the feasible six-decimal lattice, not round then clip."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError('Non-finite or non-vector schedule')
    if not MIN_ENERGY <= initial_energy <= MAX_ENERGY:
        raise ValueError('Invalid initial energy')
    energy, result = float(initial_energy), []
    for power in q:
        lower = max(-MAX_POWER, -(MAX_ENERGY - energy) / EFFICIENCY)
        upper = min(MAX_POWER, (energy - MIN_ENERGY) * EFFICIENCY)
        lo, hi = np.ceil(lower * 1e6), np.floor(upper * 1e6)
        value = float(np.clip(np.rint(power * 1e6), lo, hi) / 1e6)
        next_energy = energy - (value * EFFICIENCY if value <= 0 else value / EFFICIENCY)
        # Floating subtraction can cross a bound by one ulp even when the
        # lattice endpoint is mathematically feasible. Move one output tick
        # inward rather than clamp the battery state and hide energy error.
        if next_energy < MIN_ENERGY:
            value = round(value - 1e-6, 6)
        elif next_energy > MAX_ENERGY:
            value = round(value + 1e-6, 6)
        result.append(value)
        energy -= value * EFFICIENCY if value <= 0 else value / EFFICIENCY
    result = np.asarray(result)
    if not validate_schedule(result, initial_energy, len(result))['valid']:
        raise ValueError('Quantized plan failed independent validation')
    return result


def settle_day(q, actual_prices, initial_energy):
    prices = np.asarray(actual_prices, dtype=float)
    if prices.shape != (24,) or not np.isfinite(prices).all():
        raise ValueError('Settlement requires 24 finite prices')
    check = validate_schedule(q, initial_energy)
    executed = np.asarray(q, dtype=float) if check['valid'] else np.zeros(24)
    end = float(check['energy'][-1]) if check['valid'] else float(initial_energy)
    return dict(executed=executed, revenue=float(prices @ executed),
                end_energy=end, **check)
