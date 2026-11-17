"""TabPFN-2 zero-shot forecaster: 24 hourly in-context regressions."""
import warnings

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from paper_tabpfn.config import QUANTILES, SEED

warnings.filterwarnings('ignore', message='Feature subsampling is active')


def make_regressor(version=ModelVersion.V2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return TabPFNRegressor.create_default_for_version(version, device=device, random_state=SEED)


def forecast_hour(regressor, context_x, context_y, day_x):
    """Return {'mean', 'median', *QUANTILES} scalars for one target hour."""
    regressor.fit(context_x.to_numpy(), context_y.to_numpy())
    out = regressor.predict(day_x.to_numpy().reshape(1, -1), output_type='full', quantiles=list(QUANTILES))
    return {'mean': out['mean'][0], 'median': out['median'][0],
            **{q: values[0] for q, values in zip(QUANTILES, out['quantiles'])}}


def forecast_day(regressor, context_x, context_y, day_x):
    """Paper setup: same Eq. (5) matrix for every hour. Returns {key: (24,) array}."""
    hours = [forecast_hour(regressor, context_x, context_y[hour], day_x) for hour in range(24)]
    return {key: np.array([h[key] for h in hours]) for key in hours[0]}
