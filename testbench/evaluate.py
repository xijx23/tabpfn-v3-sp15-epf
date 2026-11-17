"""Phase 1: score one member's submission on its own (no cross-model statistics)."""
import argparse
import json
import re
import shutil
from datetime import datetime, timezone

import pandas as pd

from testbench.config import INTERVALS, RESULT_DIR, RELATIVE_DIRNAME, interval_dates
from testbench.io import read_matrix, sha256
from testbench.metrics import price_daily, storage_daily
from testbench import truth


def evaluate(interval, name, price=None, dispatch=None, result_dir=RESULT_DIR):
    if not (price or dispatch):
        raise ValueError('Provide --price and/or --dispatch')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name == RELATIVE_DIRNAME:
        raise ValueError(f'Invalid name {name!r}: use letters, digits, _ . -')
    dates = interval_dates(interval)
    actual, truth_sha = truth.load(interval)
    # Validate everything before touching the result directory.
    pred = read_matrix(price, len(dates)) if price else None
    plan = read_matrix(dispatch, len(dates)) if dispatch else None

    out = result_dir / interval / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    summary = dict(interval=interval, name=name, days=len(dates), truth_sha256=truth_sha,
                   evaluated_at_utc=datetime.now(timezone.utc).isoformat())
    if pred is not None:
        shutil.copyfile(price, out / 'price.csv')
        daily = pd.DataFrame(dict(date=dates, rmse=price_daily(pred, actual)))
        daily.to_csv(out / 'price_daily.csv', index=False)
        summary['price'] = dict(mean_rmse=float(daily['rmse'].mean()),
                                submission_sha256=sha256(price))
    if plan is not None:
        shutil.copyfile(dispatch, out / 'dispatch.csv')
        daily = pd.DataFrame(storage_daily(plan, actual))
        daily.insert(0, 'date', dates)
        daily.to_csv(out / 'storage_daily.csv', index=False)
        summary['storage'] = dict(total_revenue=float(daily['revenue'].sum()),
                                  mean_daily_revenue=float(daily['revenue'].mean()),
                                  violation_days=int((~daily['valid']).sum()),
                                  submission_sha256=sha256(dispatch))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return out, summary


def main():
    parser = argparse.ArgumentParser(description='Phase 1: per-model daily metrics')
    parser.add_argument('--interval', choices=list(INTERVALS), required=True)
    parser.add_argument('--name', required=True, help='member_model, e.g. alice_xgb')
    parser.add_argument('--price', help='header-less N x 24 price forecast CSV')
    parser.add_argument('--dispatch', help='header-less N x 24 storage plan CSV (MW, +discharge)')
    args = parser.parse_args()
    out, summary = evaluate(args.interval, args.name, args.price, args.dispatch)
    with pd.option_context('display.width', 160, 'display.max_columns', 10):
        if 'price' in summary:
            print(pd.read_csv(out / 'price_daily.csv').round(3).to_string(index=False))
            print(f"mean daily RMSE = {summary['price']['mean_rmse']:.4f}\n")
        if 'storage' in summary:
            print(pd.read_csv(out / 'storage_daily.csv', keep_default_na=False).round(4).to_string(index=False))
            s = summary['storage']
            print(f"total revenue = {s['total_revenue']:.2f}, violation days = {s['violation_days']}")
    print(f'results -> {out}')


if __name__ == '__main__':
    main()
