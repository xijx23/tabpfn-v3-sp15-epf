from data.src.common import ROOT, TZ

CAPACITY = 14.
MIN_ENERGY = .7
MAX_ENERGY = 13.3
MAX_POWER = 7.
EFFICIENCY = .85
REPORT_DIR = ROOT / 'track2/reports'
OUTPUT_DIR = ROOT / 'track2/output'
MODELS = ('ensemble', 'hgb_asinh', 'xgb_asinh')
# No window may overlap a testbench interval (2025-11-23..12-06, 2026-10-26..11-08).
WINDOWS = [('2024-11-23', '2024-12-06'), ('2026-08-29', '2026-09-11'),
           ('2026-09-12', '2026-09-25'), ('2026-09-26', '2026-10-09'),
           ('2026-10-10', '2026-10-23')]
