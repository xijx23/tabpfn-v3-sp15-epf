import hashlib

import numpy as np
import pandas as pd


def read_matrix(path, n_days):
    """Read a header-less N x 24 numeric CSV; reject the whole file on any format error."""
    try:
        frame = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError:
        raise ValueError(f'{path}: empty file')
    if frame.shape != (n_days, 24):
        raise ValueError(f'{path}: expected {n_days} rows x 24 columns, got {frame.shape[0]} x {frame.shape[1]}')
    values = frame.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    bad = np.argwhere(~np.isfinite(values))
    if len(bad):
        row, col = bad[0]
        raise ValueError(f'{path}: non-numeric or non-finite value at row {row + 1}, column {col + 1} '
                         f'({len(bad)} bad cells)')
    return values


def write_matrix(path, values):
    pd.DataFrame(values).to_csv(path, header=False, index=False)


def sha256(path):
    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()
