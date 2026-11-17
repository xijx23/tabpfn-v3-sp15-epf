"""Paper Sec. 6.2 BESS strategy, adapted to the competition battery.

Paper: 1 MWh battery, one buy hour h_b and one sell hour h_s > h_b chosen to maximise the predicted
profit eta*p_hs - p_hb/eta - C; trade only if it is positive (unlimited bid, UB). The quantile-based (QB)
variant limits the buy at q^{1-tau}_{hb} and the sell at q^{tau}_{hs}, tau = (1 - alpha) / 2.

Adaptation: 7 MW / 12.6 MWh usable needs several hours per leg, so the single buy/sell pair becomes one
daily loop: all charging before a split hour, all discharging after it, filled greedily from the cheapest
and dearest forecast hours. Energy left from earlier days is sold first. Fixed-power submissions cannot be
conditional limit orders, so QB is applied as a go/no-go filter: the day trades only if the profit of the
UB plan, valued at the pessimistic quantiles, is still positive.
"""
import numpy as np

from paper_tabpfn.config import ROUND_TRIP_COST
from track2.config import MIN_ENERGY, MAX_ENERGY, MAX_POWER, EFFICIENCY
from track2.physics import quantize_schedule

USABLE = MAX_ENERGY - MIN_ENERGY
CHARGE_CAP = MAX_POWER * EFFICIENCY      # stored MWh gained per hour at full charge
DISCHARGE_CAP = MAX_POWER / EFFICIENCY   # stored MWh drawn per hour at full discharge


def _loop(prices, stored, split, cost_per_mwh):
    """Best single loop with charging in hours < split and discharging in hours >= split."""
    charge, discharge = np.zeros(24), np.zeros(24)
    sell = sorted(range(split, 24), key=lambda h: -prices[h])
    buy = sorted(range(split), key=lambda h: prices[h])
    profit, room, i, j = 0., USABLE - stored, 0, 0
    sell_left = {h: DISCHARGE_CAP for h in sell}
    # Energy already stored is free: sell it in the dearest hours while the price is positive.
    for h in sell:
        if stored <= 0 or prices[h] <= 0:
            break
        amount = min(stored, sell_left[h])
        discharge[h] += amount
        sell_left[h] -= amount
        stored -= amount
        profit += amount * EFFICIENCY * prices[h]
    buy_left = {h: CHARGE_CAP for h in buy}
    while i < len(sell) and j < len(buy) and room > 1e-9:
        hs, hb = sell[i], buy[j]
        margin = EFFICIENCY * prices[hs] - prices[hb] / EFFICIENCY - cost_per_mwh
        if margin <= 0:
            break
        amount = min(sell_left[hs], buy_left[hb], room)
        discharge[hs] += amount
        charge[hb] += amount
        sell_left[hs] -= amount
        buy_left[hb] -= amount
        room -= amount
        profit += amount * margin
        i += sell_left[hs] <= 1e-9
        j += buy_left[hb] <= 1e-9
    power = discharge * EFFICIENCY - charge / EFFICIENCY
    return profit, power


def ub_plan(prices, energy):
    """Unlimited-bid plan (MW, + discharge) from a 24-hour point forecast and the start energy (MWh)."""
    prices = np.asarray(prices, dtype=float)
    stored = energy - MIN_ENERGY
    cost = ROUND_TRIP_COST / USABLE
    profit, power = max((_loop(prices, stored, split, cost) for split in range(25)), key=lambda r: r[0])
    if profit <= 0:
        return 0., np.zeros(24)
    return profit, quantize_schedule(power, energy)


def dp_plan(prices, energy, step=USABLE / 252):
    """Multi-loop extension of UB: exact daily optimum on a stored-energy grid by dynamic programming.

    Same rule as the paper (act on the point forecast, trade only if predicted profit > 0), but any
    number of charge/discharge loops. `prices` may cover more than 24 hours (e.g. 48); only the first
    24 hours are returned, so energy can be held past midnight for the next morning.
    Energy left at the end of the horizon carries no value.
    """
    prices = np.asarray(prices, dtype=float)
    hours = len(prices)
    grid = MIN_ENERGY + step * np.arange(round(USABLE / step) + 1)

    def reward(start, end, price):
        delta = end - start[:, None]
        power = np.where(delta > 0, -delta / EFFICIENCY, -delta * EFFICIENCY)
        ok = (delta <= CHARGE_CAP + 1e-9) & (-delta <= DISCHARGE_CAP + 1e-9)
        return np.where(ok, price * power, -np.inf), power

    value, choice = np.zeros(len(grid)), []
    for hour in range(hours - 1, 0, -1):
        total = reward(grid, grid, prices[hour])[0] + value
        choice.append(total.argmax(axis=1))
        value = total.max(axis=1)
    first, first_power = reward(np.array([energy]), grid, prices[0])
    total = first[0] + value
    if total.max() <= 1e-9:
        return 0., np.zeros(24)
    state = int(total.argmax())
    power = [first_power[0, state]]
    for pick in reversed(choice[-23:]):
        nxt = int(pick[state])
        power.append(-(grid[nxt] - grid[state]) / EFFICIENCY if grid[nxt] > grid[state]
                     else (grid[state] - grid[nxt]) * EFFICIENCY)
        state = nxt
    return float(total.max()), quantize_schedule(np.array(power), energy)


def pessimistic_profit(power, low, high):
    """Plan value if every sale clears at the low quantile and every purchase at the high one."""
    power = np.asarray(power)
    return float(np.where(power > 0, power * low, power * high).sum())


def qb_plan(prices, low, high, energy):
    """QB filter: keep the UB plan only if it stays profitable at quantiles tau / 1 - tau."""
    profit, power = ub_plan(prices, energy)
    if profit <= 0 or pessimistic_profit(power, low, high) <= ROUND_TRIP_COST:
        return np.zeros(24)
    return power
