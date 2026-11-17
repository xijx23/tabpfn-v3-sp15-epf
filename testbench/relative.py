"""Phase 2: relative scores among all team submissions present for an interval."""
import argparse
import json

import numpy as np
import pandas as pd

from testbench.config import INTERVALS, RESULT_DIR, RELATIVE_DIRNAME, S_VALUES

TRACKS = {'price': ('price_daily.csv', 'rmse', False),
          'storage': ('storage_daily.csv', 'revenue', True)}


def relative_scores(matrix, s, higher_better):
    """matrix: models x days. Per day, compare each model with the cross-model mean/std."""
    matrix = np.asarray(matrix, dtype=float)
    cross_mean = matrix.mean(axis=0)
    cross_std = matrix.std(axis=0)  # population std, ddof=0
    sign = 1. if higher_better else -1.
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(cross_std > 0, sign * (matrix - cross_mean) / cross_std, 0.)
    return np.clip(60. + s * z, 0., 100.), cross_mean, cross_std


def collect(interval, track, result_dir=RESULT_DIR):
    filename, column, _ = TRACKS[track]
    frames, hashes = {}, {}
    for folder in sorted((result_dir / interval).glob('*')):
        if folder.name == RELATIVE_DIRNAME or not (folder / filename).exists():
            continue
        frames[folder.name] = pd.read_csv(folder / filename).set_index('date')[column]
        hashes[folder.name] = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))['truth_sha256']
    if len(set(hashes.values())) > 1:
        raise ValueError(f'{track}: results use different truth files {hashes}; re-run phase 1')
    return pd.DataFrame(frames)  # days x models


def run(interval, result_dir=RESULT_DIR):
    out = result_dir / interval / RELATIVE_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for track, (_, column, higher_better) in TRACKS.items():
        table = collect(interval, track, result_dir)
        if table.empty:
            continue
        daily = pd.DataFrame(index=table.index)
        for s in S_VALUES:
            scores, cross_mean, cross_std = relative_scores(table.T.to_numpy(), s, higher_better)
            for model, row in zip(table.columns, scores):
                daily[f'{model}_s{s}'] = row
                summary.append(dict(track=track, model=model, s=s, mean_relative_score=row.mean(),
                                    participants=';'.join(table.columns)))
        daily[f'cross_mean_{column}'] = cross_mean
        daily[f'cross_std_{column}'] = cross_std
        daily.to_csv(out / f'{track}_relative_daily.csv')
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'relative_summary.csv', index=False)
    return out, summary


def main():
    parser = argparse.ArgumentParser(description='Phase 2: cross-model relative scores')
    parser.add_argument('--interval', choices=list(INTERVALS), required=True)
    args = parser.parse_args()
    out, summary = run(args.interval)
    if summary.empty:
        print('no phase-1 results found')
    else:
        table = summary.pivot_table(index=['track', 'model'], columns='s', values='mean_relative_score')
        print(table.rename(columns=lambda s: f's={s}').round(2).to_string())
    print(f'results -> {out}')


if __name__ == '__main__':
    main()
