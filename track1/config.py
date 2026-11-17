from data.src.common import ROOT, TZ

WIDE_PATH = ROOT / 'data/processed/hourly_wide.parquet'
REPORT_DIR = ROOT / 'track1/reports'
OUTPUT_DIR = ROOT / 'track1/output'
MODEL_PATH = OUTPUT_DIR / 'model.joblib'
TARGET = 'target_sp15_dam_lmp'
TRAIN_START = '2023-07-28'
MIN_TRAIN_ROWS = 90 * 24
THREADS = 4
SEED = 2026
MODEL_NAMES = ('naive_day', 'naive_week', 'hgb', 'hgb_asinh', 'xgboost', 'xgb_asinh', 'ensemble')
# Declared before results are observed. End dates are inclusive.
VALIDATION_WINDOWS = [('2025-11-23', '2025-12-06'), ('2026-09-17', '2026-09-30')]
HOLDOUT_WINDOW = ('2026-10-01', '2026-10-31')
HGB_PARAMS = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
                  min_samples_leaf=40, l2_regularization=1., early_stopping=False,
                  random_state=SEED)
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=0,
                  max_leaves=31, grow_policy='lossguide',
                  min_child_weight=40, subsample=1., colsample_bytree=1.,
                  reg_lambda=1., objective='reg:squarederror', tree_method='hist',
                  n_jobs=THREADS, random_state=SEED)
# Similar-day features. Prior choices, not tuned on any evaluation window.
ANALOG_PARAMS = dict(k=10, lookback_days=730, min_candidates=60, day_type_penalty=.5,
                     weights={'net_load': 1., 'load_caiso': .5, 'solar_total': .5,
                              'wind_total': .5, 'sp15_dam_lmp_lag1d': 1.})
# Feature-based models re-trained with analog features; reported as '<name>_analog'.
ANALOG_GROUP = ('hgb', 'hgb_asinh', 'xgboost', 'xgb_asinh', 'ensemble')
# Change-vs-yesterday target + capacity-normalised renewables + extended holidays (from the
# partner model hh110g); reported as '<name>_rch'.
RCH_GROUP = ('hgb', 'hgb_asinh', 'xgboost', 'xgb_asinh', 'ensemble')
