import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from track2.config import MIN_ENERGY, MAX_ENERGY, MAX_POWER, EFFICIENCY
from track2.physics import validate_schedule, quantize_schedule


def solve_schedule(prices, initial_energy, scenarios=None, risk_weight=0., time_limit=30.):
    """H-hour MILP. Scenario risk applies to the first 24 fixed decisions only."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or not len(prices) or not np.isfinite(prices).all():
        raise ValueError('Prices must be a finite nonempty vector')
    if not MIN_ENERGY <= initial_energy <= MAX_ENERGY or not np.isfinite(initial_energy):
        raise ValueError('Invalid initial energy')
    if risk_weight < 0 or not np.isfinite(risk_weight):
        raise ValueError('Invalid risk weight')
    h = len(prices)
    scenarios = None if scenarios is None else np.asarray(scenarios, dtype=float)
    if scenarios is not None and (h < 24 or scenarios.ndim != 2 or scenarios.shape[1] != 24
                                   or not len(scenarios) or not np.isfinite(scenarios).all()):
        raise ValueError('Scenarios require shape (n, 24) and a horizon of at least 24')
    if risk_weight and scenarios is None:
        raise ValueError('Risk optimization requires scenarios')
    n = len(scenarios) if scenarios is not None and risk_weight else 0
    # c[H], d[H], z[H], E[H+1], then unrestricted VaR and nonnegative excess losses.
    base = 4 * h + 1
    size = base + (n + 1 if n else 0)
    objective = np.zeros(size)
    mean = prices.copy()
    if scenarios is not None:
        mean[:24] = scenarios.mean(axis=0)
    objective[:h], objective[h:2*h] = mean, -mean
    lower, upper = np.zeros(size), np.full(size, np.inf)
    upper[:2*h], upper[2*h:3*h] = MAX_POWER, 1
    lower[3*h:base], upper[3*h:base] = MIN_ENERGY, MAX_ENERGY
    lower[3*h] = upper[3*h] = initial_energy
    integrality = np.zeros(size)
    integrality[2*h:3*h] = 1
    matrix = lil_matrix((3*h + n, size))
    lb, ub = np.full(3*h + n, -np.inf), np.zeros(3*h + n)
    for t in range(h):
        matrix[t, [t, h+t, 3*h+t, 3*h+t+1]] = [-EFFICIENCY, 1 / EFFICIENCY, -1, 1]
        lb[t] = 0
        matrix[h+t, [t, 2*h+t]] = [1, -MAX_POWER]
        matrix[2*h+t, [h+t, 2*h+t]] = [1, MAX_POWER]
        ub[2*h+t] = MAX_POWER
    if n:
        lower[base] = -np.inf
        objective[base] = risk_weight
        objective[base+1:] = risk_weight / (.1 * n)
        for s, path in enumerate(scenarios):
            matrix[3*h+s, :24] = path
            matrix[3*h+s, h:h+24] = -path
            matrix[3*h+s, base] = -1
            matrix[3*h+s, base+1+s] = -1
    started = time.monotonic()
    result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix.tocsc(), lb, ub),
                  options=dict(time_limit=time_limit, mip_rel_gap=1e-6))
    reason = None
    q = np.zeros(h)
    if result.status not in (0, 1) or result.x is None:
        reason = f'No usable incumbent: {result.message}'
    else:
        x = result.x
        raw = x[h:2*h] - x[:h]
        # Never hide a genuinely infeasible incumbent through projection.
        check = validate_schedule(raw, initial_energy, h)
        if (not np.isfinite(x).all() or check['max_energy_violation'] > 1e-5
                or np.max(np.abs(raw)) > MAX_POWER + 1e-5
                or np.max(np.minimum(x[:h], x[h:2*h])) > 1e-5):
            reason = 'Incumbent failed physical checks'
        else:
            q = quantize_schedule(raw, initial_energy)
    check = validate_schedule(q, initial_energy, h)
    def finite(value):
        return float(value) if value is not None and np.isfinite(value) else None

    return dict(power=q, energy=check['energy'], status=int(result.status),
                message=str(result.message), fallback=reason,
                objective=finite(-result.fun if result.fun is not None else None),
                objective_upper_bound=finite(-result.mip_dual_bound if getattr(result, 'mip_dual_bound', None) is not None else None),
                expected_revenue=float(mean @ q),
                mip_gap=finite(getattr(result, 'mip_gap', None)),
                seconds=time.monotonic() - started)
