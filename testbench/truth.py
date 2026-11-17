"""Build and freeze the N x 24 SP15 DAM LMP truth matrix for each test interval.

Column j is local (America/Los_Angeles) hour j:00-(j+1):00. On the fall-back day the
repeated 01:00 keeps only its first occurrence (PDT); the second (PST) is dropped.
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from testbench.config import (INTERVALS, PRICE_COLUMN, NODE, WIDE_PATH, RAW_LMP_DIR,
                              TRUTH_DIR, TZ, interval_dates)
from testbench.io import write_matrix, sha256


def day_matrix(series, dates):
    """series: UTC-indexed hourly prices. Returns len(dates) x 24 with first-occurrence hours."""
    local = series.dropna().tz_convert(TZ)
    frame = pd.DataFrame({'value': local.to_numpy(),
                          'date': local.index.strftime('%Y-%m-%d'),
                          'hour': local.index.hour})
    frame = frame[frame['date'].isin(dates)]
    frame = frame.drop_duplicates(['date', 'hour'], keep='first')  # index is sorted by UTC
    rows = []
    for date in dates:
        day = frame[frame['date'] == date].set_index('hour')['value']
        if sorted(day.index) != list(range(24)):
            raise ValueError(f'{date}: expected local hours 0-23, got {len(day)} values')
        rows.append(day.sort_index().to_numpy())
    return np.vstack(rows)


def raw_check(matrix, dates):
    """Cross-check against raw OASIS OPR_DT/OPR_HR: column k-1 <-> OPR_HR k, OPR_HR 25 dropped."""
    months = sorted({d[:7] for d in dates})
    raw = pd.concat([pd.read_parquet(RAW_LMP_DIR / f'{m}.parquet') for m in months])
    raw = raw[(raw['NODE'] == NODE) & (raw['LMP_TYPE'] == 'LMP') & (raw['OPR_HR'] <= 24)]
    raw = raw.drop_duplicates(['OPR_DT', 'OPR_HR'], keep='last')
    for i, date in enumerate(dates):
        day = raw[raw['OPR_DT'] == date].set_index('OPR_HR')['MW'].sort_index()
        if list(day.index) != list(range(1, 25)):
            raise ValueError(f'{date}: raw OPR_HR 1-24 incomplete')
        # Fall-back day: OPR_HR 2 is the first 01:00; OPR_HR k>=3 is local hour k-1. Same mapping.
        if not np.allclose(day.to_numpy(), matrix[i], atol=1e-6):
            raise ValueError(f'{date}: wide-table truth disagrees with raw OPR_HR')


def build(interval, force=False):
    dates = interval_dates(interval)
    csv_path = TRUTH_DIR / f'{interval}.csv'
    if csv_path.exists() and not force:
        raise FileExistsError(f'{csv_path} already frozen; pass --force to rebuild')
    series = pd.read_parquet(WIDE_PATH, columns=[PRICE_COLUMN])[PRICE_COLUMN]
    matrix = day_matrix(series, dates)
    raw_check(matrix, dates)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    write_matrix(csv_path, matrix)
    meta = dict(interval=interval, dates=dates, node=NODE, market='DAM', lmp_type='LMP',
                sha256=sha256(csv_path), source=str(WIDE_PATH.relative_to(WIDE_PATH.parents[2])),
                source_mtime_utc=datetime.fromtimestamp(WIDE_PATH.stat().st_mtime, timezone.utc).isoformat(),
                built_at_utc=datetime.now(timezone.utc).isoformat())
    (TRUTH_DIR / f'{interval}.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return csv_path, meta


def load(interval):
    """Return (matrix, sha256) of the frozen truth."""
    from testbench.io import read_matrix
    csv_path = TRUTH_DIR / f'{interval}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} missing; run python -m testbench.truth --interval {interval}')
    meta = json.loads((TRUTH_DIR / f'{interval}.json').read_text(encoding='utf-8'))
    digest = sha256(csv_path)
    if digest != meta['sha256']:
        raise ValueError(f'{csv_path} hash differs from its JSON record')
    return read_matrix(csv_path, len(meta['dates'])), digest


def main():
    parser = argparse.ArgumentParser(description='Freeze SP15 DAM truth matrices')
    parser.add_argument('--interval', choices=list(INTERVALS), action='append')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    for interval in args.interval or list(INTERVALS):
        path, meta = build(interval, args.force)
        print(f'{interval}: {path} sha256={meta["sha256"][:12]}')


if __name__ == '__main__':
    main()
