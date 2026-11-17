from data.src.common import ROOT, TZ

# California operating dates, both ends inclusive.
INTERVALS = {'w7_2026': ('2026-11-02', '2026-11-08'),
             'd14_2026': ('2026-10-26', '2026-11-08'),
             'd14_2025': ('2025-11-23', '2025-12-06')}
S_VALUES = (10, 20)
PRICE_COLUMN = 'sp15_dam_lmp'
NODE = 'TH_SP15_GEN-APND'
WIDE_PATH = ROOT / 'data/processed/hourly_wide.parquet'
RAW_LMP_DIR = ROOT / 'data/raw/dam_lmp'
TRUTH_DIR = ROOT / 'testbench/truth'
RESULT_DIR = ROOT / 'testbench/results'
RELATIVE_DIRNAME = '_relative'


def interval_dates(interval):
    import pandas as pd
    start, end = INTERVALS[interval]
    return [d.strftime('%Y-%m-%d') for d in pd.date_range(start, end, freq='D')]
