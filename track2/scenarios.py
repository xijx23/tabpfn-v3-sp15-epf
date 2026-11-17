import numpy as np
import pandas as pd

from data.src.common import drop_repeated_local_hour
from track2.config import TZ
from track2.optimizer import solve_schedule


def price_profile(wide):
    wide = drop_repeated_local_hour(wide.sort_index())
    local = wide.index.tz_convert(TZ)
    rows = pd.DataFrame({'date': local.strftime('%Y-%m-%d'), 'hour': local.hour,
                         'price': wide.sp15_dam_lmp.to_numpy()})
    counts = rows.groupby('date').size()
    rows = rows.loc[rows.date.isin(counts[counts == 24].index)]
    return rows.pivot(index='date', columns='hour', values='price').reindex(columns=range(24))


def continuation_prices(profile, date):
    target = pd.Timestamp(date) + pd.Timedelta(days=1)
    dates = [(target - pd.Timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 5)]
    values = profile.reindex(dates).mean(axis=0).to_numpy()
    return values if np.isfinite(values).all() else None


def residual_scenarios(predictions, date, model, point):
    start = (pd.Timestamp(date) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    past = predictions.loc[(predictions.model == model) & (predictions.operating_date >= start)
                           & (predictions.operating_date < date)].copy()
    if past.empty:
        return None
    if (past.trained_through >= past.operating_date).any():
        raise ValueError('Residuals contain in-sample predictions')
    if past.duplicated(['operating_date', 'hour']).any():
        raise ValueError('Duplicate residual observations')
    past['residual'] = past.actual - past.prediction
    paths = past.pivot(index='operating_date', columns='hour', values='residual').reindex(columns=range(24))
    paths = paths.loc[np.isfinite(paths).all(axis=1)]
    if len(paths) < 60:
        return None
    return np.asarray(point) + (paths - paths.mean(axis=0)).to_numpy()


def plan_day(point, initial_energy, profile, date, horizon=48, final_day=False,
             scenarios=None, risk_weight=0.):
    point = np.asarray(point, dtype=float)
    if point.shape != (24,) or not np.isfinite(point).all():
        raise ValueError('Expected 24 finite day-ahead forecasts')
    if horizon not in (24, 48):
        raise ValueError('Horizon must be 24 or 48')
    continuation = continuation_prices(profile, date) if horizon == 48 and not final_day else None
    prices = point if continuation is None else np.r_[point, continuation]
    effective_risk = risk_weight if scenarios is not None else 0.
    result = solve_schedule(prices, initial_energy, scenarios, effective_risk)
    result.update(horizon=len(prices), requested_risk=risk_weight, effective_risk=effective_risk,
                  scenario_count=0 if scenarios is None else len(scenarios),
                  risk_disabled_reason='Fewer than 60 complete residual days' if risk_weight and scenarios is None else None)
    result['power'] = result['power'][:24]
    result['energy'] = result['energy'][:25]
    return result
