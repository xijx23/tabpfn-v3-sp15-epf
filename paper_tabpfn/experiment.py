"""Variant comparison on validation days outside every testbench interval.

Validation days also appear in track1's daily-retrained rolling backtest, so TabPFN can be compared
with, and combined with, the tree ensemble there. Each variant caches to reports/experiments/<name>.csv.
"""
import argparse

import numpy as np
import pandas as pd

from paper_tabpfn.config import WIDE_PATH, OUTPUT, CONTEXT_DAYS
from paper_tabpfn.features import daily_profiles, design_matrix, compact_matrix
from track1.config import REPORT_DIR

VALIDATION = [*pd.date_range('2024-11-23', '2024-12-06'), *pd.date_range('2026-09-26', '2026-10-23')]
VALIDATION = [d.strftime('%Y-%m-%d') for d in VALIDATION]
TREE = REPORT_DIR / 'aligned_daily_weighted_analog/predictions.csv'
# name -> (features, context days, target relative to yesterday's same hour, TabPFN weights)
VARIANTS = {'paper': ('paper', CONTEXT_DAYS, False, 'v2'),
            'compact': ('compact', CONTEXT_DAYS, False, 'v2'),
            'compact_ctx365': ('compact', 365, False, 'v2'),
            'compact_diff': ('compact', CONTEXT_DAYS, True, 'v2'),
            'paper_v3': ('paper', CONTEXT_DAYS, False, 'v3')}


def hour_inputs(profiles, paper_x, features, hour):
    if features == 'paper':
        return paper_x
    return compact_matrix(profiles, hour)


def run_variant(name, dates):
    path = OUTPUT / 'experiments' / f'{name}.csv'
    done = pd.read_csv(path, dtype={'date': str}) if path.exists() else pd.DataFrame()
    todo = [d for d in dates if done.empty or d not in set(done.date)]
    if not todo:
        return done
    from tabpfn.constants import ModelVersion
    from paper_tabpfn.model import make_regressor, forecast_hour
    features, days, diff, version = VARIANTS[name]
    profiles = daily_profiles(pd.read_parquet(WIDE_PATH))
    paper_x, y = design_matrix(profiles)
    regressor = make_regressor(ModelVersion(version))
    inputs = {h: hour_inputs(profiles, paper_x, features, h) for h in range(24)}
    complete = y.notna().all(axis=1) & profiles['load'].notna().all(axis=1)
    rows = []
    for date in todo:
        for hour in range(24):
            x = inputs[hour]
            mask = complete & (x.index < date)
            cx, target = x.loc[mask].tail(days), y.loc[mask, hour].tail(days)
            base = y[hour].shift(1)
            if diff:
                target = (target - base.loc[target.index]).dropna()
            out = forecast_hour(regressor, cx.loc[target.index], target, x.loc[[date]])
            shift = base.loc[date] if diff else 0.
            rows.append(dict(date=date, hour=hour, mean=out['mean'] + shift, median=out['median'] + shift,
                             actual=y.loc[date, hour]))
        print(f'{name} {date}', flush=True)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        rows = []
        path.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(path, index=False)
    return done


def daily_rmse(frame, column):
    return frame.assign(se=(frame[column] - frame.actual) ** 2).groupby('date').se.mean().pow(.5)


def report(names, dates):
    tree = pd.read_csv(TREE, usecols=['operating_date', 'hour', 'model', 'prediction'])
    tree = tree[tree.model == 'ensemble'].rename(columns={'operating_date': 'date', 'prediction': 'tree'})
    rows = {}
    for name in names:
        frame = run_variant(name, dates)
        frame = frame[frame.date.isin(dates)].merge(tree[['date', 'hour', 'tree']], on=['date', 'hour'])
        frame['combo'] = (frame['mean'] + frame.tree) / 2
        rows[name] = {col: daily_rmse(frame, col).mean() for col in ('median', 'mean', 'combo')}
        rows[name]['bias'] = (frame['mean'] - frame.actual).mean()
        rows['tree_ensemble'] = dict(mean=daily_rmse(frame, 'tree').mean(), bias=(frame.tree - frame.actual).mean())
    print(pd.DataFrame(rows).T.round(3).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANTS), action='append', required=True)
    parser.add_argument('--every', type=int, default=1, help='use every k-th validation day')
    args = parser.parse_args()
    report(args.variant, VALIDATION[::args.every])


if __name__ == '__main__':
    main()
