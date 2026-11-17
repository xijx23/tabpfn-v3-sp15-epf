"""Run TabPFN-2 and the paper's BESS strategies on the testbench intervals.

Every target day d gets its own context: the most recent CONTEXT_DAYS complete days before d (paper's
daily rolling in-context setup; no weights are trained). Forecasts are cached in reports/forecasts.csv,
so overlapping intervals and reruns reuse them.
"""
import argparse

import numpy as np
import pandas as pd

from paper_tabpfn.config import WIDE_PATH, OUTPUT, CONTEXT_DAYS, QUANTILES, ALPHAS
from paper_tabpfn.features import daily_profiles, design_matrix, context_before
from paper_tabpfn.storage import ub_plan, qb_plan, dp_plan
from testbench import truth
from testbench.config import INTERVALS, interval_dates
from testbench.evaluate import evaluate
from testbench.io import write_matrix, read_matrix
from track1.config import REPORT_DIR as TRACK1_REPORTS
from track2.config import MIN_ENERGY
from track2.physics import settle_day
from track2.scenarios import price_profile, continuation_prices

VERSIONS = ('v2', 'v2.5', 'v2.6', 'v3', 'v3.5')
TRACK1_DIR = TRACK1_REPORTS / 'testbench'
KEYS = ['mean', 'median', *map(str, QUANTILES)]


def suffix(version):
    """'' for the original TabPFN-2 run (unchanged file and model names), '_v3' etc. otherwise."""
    return '' if version == 'v2' else '_' + version.replace('.', '')


def cache_path(version):
    return OUTPUT / f'forecasts{suffix(version)}.csv'


def load_cache(version='v2'):
    if cache_path(version).exists():
        return pd.read_csv(cache_path(version), dtype={'date': str, 'kind': str})
    return pd.DataFrame(columns=['date', 'kind', *map(str, range(24))])


def forecast_dates(dates, version='v2'):
    """Forecast missing dates, append them to the cache and return the full cache."""
    cache = load_cache(version)
    todo = sorted(set(dates) - set(cache.date))
    if todo:
        from tabpfn.constants import ModelVersion
        from paper_tabpfn.model import make_regressor, forecast_day
        x, y = design_matrix(daily_profiles(pd.read_parquet(WIDE_PATH)))
        regressor = make_regressor(ModelVersion(version))
        for date in todo:
            if x.loc[date].filter(regex='^(p_lag1|load_lag0)_').isna().any():
                raise ValueError(f'{date}: missing D-1 prices or D-day load forecast')
            cx, cy = context_before(x, y, date, CONTEXT_DAYS)
            result = forecast_day(regressor, cx, cy, x.loc[[date]])
            rows = pd.DataFrame([[date, str(k), *result[k]] for k in result], columns=cache.columns)
            cache = pd.concat([cache, rows], ignore_index=True)
            OUTPUT.mkdir(parents=True, exist_ok=True)
            cache.to_csv(cache_path(version), index=False)
            print(f'{date}: context {len(cx)} days', flush=True)
    return cache


def matrix(cache, dates, kind):
    block = cache[cache.kind == kind].set_index('date').loc[dates]
    return block[[str(h) for h in range(24)]].to_numpy(dtype=float)


def tree_forecast(interval, n_days):
    """track1's frozen ensemble_analog forecast (read-only; run track1.testbench_run first)."""
    return read_matrix(TRACK1_DIR / interval / 'price_ensemble_analog.csv', n_days)


def dispatch_dp48(forecast, dates, actual, profile):
    """Multi-loop DP over D plus a D+1 continuation (4-week same weekday/hour mean); last day 24 h."""
    energy, plans = MIN_ENERGY, []
    for i, date in enumerate(dates):
        prices = forecast[i] if i == len(dates) - 1 else np.r_[forecast[i], continuation_prices(profile, date)]
        power = dp_plan(prices, energy)[1]
        plans.append(power)
        energy = settle_day(power, actual[i], energy)['end_energy']
    return np.array(plans)


def dispatch(cache, dates, actual, alpha=None):
    """Plan day by day, advancing the SOC with realised prices as the testbench does."""
    median = matrix(cache, dates, 'median')
    if alpha is not None:
        tau = round((1 - alpha) / 2, 2)
        low, high = matrix(cache, dates, str(tau)), matrix(cache, dates, str(round(1 - tau, 2)))
    energy, plans = MIN_ENERGY, []
    for i in range(len(dates)):
        if alpha is None:
            power = ub_plan(median[i], energy)[1]
        else:
            power = qb_plan(median[i], low[i], high[i], energy)
        plans.append(power)
        energy = settle_day(power, actual[i], energy)['end_energy']
    return np.array(plans)


def run_interval(cache, interval, profile, tag=''):
    dates = interval_dates(interval)
    actual, _ = truth.load(interval)
    out = OUTPUT / f'{interval}{tag}'
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    median = matrix(cache, dates, 'median')
    # Equal-weight forecast averaging with the tree ensemble; weight fixed a priori, not tuned.
    combo = (median + tree_forecast(interval, len(dates))) / 2
    for values, kind, name in [(median, 'median', 't1_tabpfn'), (matrix(cache, dates, 'mean'), 'mean', 't1_tabpfn_mean'),
                               (combo, 'combo', 't1_tabpfn_combo')]:
        name = name.replace('tabpfn', f'tabpfn{tag}')
        path = out / f'price_{kind}.csv'
        write_matrix(path, values)
        summaries[name] = evaluate(interval, name, price=path)[1]['price']['mean_rmse']
    for values, name in [(median, 't2_tabpfn_dp48'), (combo, 't2_tabpfn_combo_dp48')]:
        path = out / f'dispatch_{name[3:]}.csv'
        name = name.replace('tabpfn', f'tabpfn{tag}')
        write_matrix(path, dispatch_dp48(values, dates, actual, profile))
        storage = evaluate(interval, name, dispatch=path)[1]['storage']
        summaries[name] = (storage['total_revenue'], storage['violation_days'])
    for alpha in (None, *ALPHAS):
        rule = 'ub' if alpha is None else f'qb{int(alpha * 100)}'
        path = out / f'dispatch_{rule}.csv'
        write_matrix(path, dispatch(cache, dates, actual, alpha))
        name = f't2_tabpfn{tag}_{rule}'
        storage = evaluate(interval, name, dispatch=path)[1]['storage']
        summaries[name] = (storage['total_revenue'], storage['violation_days'])
    return summaries


def main():
    parser = argparse.ArgumentParser(description='TabPFN (arXiv:2609.00089) on the testbench intervals')
    parser.add_argument('--interval', choices=list(INTERVALS), action='append')
    parser.add_argument('--version', choices=VERSIONS, default='v2', help='TabPFN weights (v2.5+ need a license key)')
    args = parser.parse_args()
    intervals = args.interval or list(INTERVALS)
    cache = forecast_dates({d for i in intervals for d in interval_dates(i)}, args.version)
    profile = price_profile(pd.read_parquet(WIDE_PATH, columns=['sp15_dam_lmp']))
    for interval in intervals:
        print(interval)
        for name, value in run_interval(cache, interval, profile, suffix(args.version)).items():
            print(f'  {name}: {value}')


if __name__ == '__main__':
    main()
