"""Daily design matrix of paper Eq. (5), one row per operating day, 24 hourly targets.

f(p_{d-1}, p_{d-2}, p_{d-3}, p_{d-7}, L_d, L_{d-1}, L_{d-7}, R_d, R_{d-1}, R_{d-7}, fuel_{d-lag}, DoW_d)
Each p/L/R term is the full 24-hour vector of that day.
"""
import pandas as pd

from data.src.common import drop_repeated_local_hour
from paper_tabpfn.config import TZ, LOAD_COL, RES_COLS, FUEL_COLS, FUEL_LAG, PRICE_LAGS, EXOG_LAGS


def daily_profiles(wide):
    """Local-date x hour matrices. Fall-back day keeps the first 01:00; 23-hour days stay incomplete."""
    wide = drop_repeated_local_hour(wide.sort_index())
    local = wide.index.tz_convert(TZ)
    frame = wide.assign(date=local.strftime('%Y-%m-%d'), hour=local.hour)
    # Paper Sec. 3: missing renewable forecasts are replaced with zeros.
    frame['res'] = frame[RES_COLS].fillna(0).sum(axis=1).where(frame[RES_COLS].notna().any(axis=1))
    dates = pd.date_range(local.min().date(), local.max().date(), freq='D').strftime('%Y-%m-%d')

    def pivot(column):
        return frame.pivot_table(index='date', columns='hour', values=column, aggfunc='first') \
            .reindex(index=dates, columns=range(24))

    frame['solar'] = frame[[c for c in RES_COLS if c.endswith('Solar')]].fillna(0).sum(axis=1)
    frame['wind'] = frame[[c for c in RES_COLS if c.endswith('Wind')]].fillna(0).sum(axis=1)
    fuels = frame.groupby('date')[list(FUEL_COLS)].mean().reindex(dates).ffill().rename(columns=FUEL_COLS)
    return dict(price=pivot('sp15_dam_lmp'), load=pivot(LOAD_COL), res=pivot('res'), fuel=fuels,
                solar=pivot('solar'), wind=pivot('wind'))


def design_matrix(profiles):
    """X indexed by date (features known before the D-1 10:00 cutoff) and Y = 24 SP15 DAM prices."""
    parts = []
    for lag in PRICE_LAGS:
        parts.append(profiles['price'].shift(lag).add_prefix(f'p_lag{lag}_h'))
    for name in ('load', 'res'):
        for lag in EXOG_LAGS:
            parts.append(profiles[name].shift(lag).add_prefix(f'{name}_lag{lag}_h'))
    parts.append(profiles['fuel'].shift(FUEL_LAG).add_suffix(f'_lag{FUEL_LAG}'))
    x = pd.concat(parts, axis=1)
    x['dow'] = pd.to_datetime(x.index).day_of_week
    x.columns = [str(c) for c in x.columns]
    return x, profiles['price']


def compact_matrix(profiles, hour):
    """Per-hour design matrix (~20 columns) so TabPFN does not subsample away the relevant hour."""
    price, load, res = profiles['price'], profiles['load'], profiles['res']
    net = load - res
    x = pd.DataFrame(index=price.index)
    for lag in (1, 2, 7):
        x[f'p_lag{lag}'] = price[hour].shift(lag)
    lag1 = price.shift(1)
    x['p_lag1_mean'], x['p_lag1_min'], x['p_lag1_max'] = lag1.mean(axis=1), lag1.min(axis=1), lag1.max(axis=1)
    x['load'], x['res'], x['net'] = load[hour], res[hour], net[hour]
    x['solar'], x['wind'] = profiles['solar'][hour], profiles['wind'][hour]
    x['net_mean'], x['net_min'], x['net_max'] = net.mean(axis=1), net.min(axis=1), net.max(axis=1)
    x['net_change_1d'] = net[hour] - net[hour].shift(1)
    x['res_lag1'] = res[hour].shift(1)
    for name, values in profiles['fuel'].shift(FUEL_LAG).items():
        x[name] = values
    x['dow'] = pd.to_datetime(x.index).day_of_week
    return x


def context_before(x, y, date, days):
    """Training rows strictly before date, complete target and day-ahead load forecast, most recent `days`."""
    mask = (x.index < date) & y.notna().all(axis=1) & x.filter(like='load_lag0_').notna().all(axis=1)
    return x.loc[mask].tail(days), y.loc[mask].tail(days)
