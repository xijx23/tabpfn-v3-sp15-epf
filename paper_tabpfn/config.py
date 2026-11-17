from data.src.common import ROOT, TZ

WIDE_PATH = ROOT / 'data/processed/hourly_wide.parquet'
OUTPUT = ROOT / 'paper_tabpfn/reports'

# Paper Sec. 4.3: TabPFN context = 1449 most recent days (we have less history, so all of it is used).
CONTEXT_DAYS = 1449
PRICE_LAGS = (1, 2, 3, 7)          # p_{d-1}, p_{d-2}, p_{d-3}, p_{d-7}
EXOG_LAGS = (0, 1, 7)              # L_d, L_{d-1}, L_{d-7}; same for R
SEED = 2026

# CAISO stand-ins for the paper's inputs.
LOAD_COL = 'load_fc_2da__CA ISO-TAC__2'
RES_COLS = [f'renew_dam__{hub}__{kind}' for hub in ('NP15', 'SP15', 'ZP26') for kind in ('Solar', 'Wind')]
# EUA/TTF/Brent/API2 -> SoCal gas index and carbon price. The project cutoff (D-1 10:00) allows lag 1.
FUEL_COLS = {'gas__FRSCE1': 'gas', 'carbon_price': 'carbon'}
FUEL_LAG = 1

# Quantile levels returned by the forecaster (median = point forecast, paper Eq. 10).
QUANTILES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95)
# Quantile-based strategy: alpha -> tau = (1 - alpha) / 2 (paper Sec. 6.2).
ALPHAS = (0.9, 0.7, 0.5)
# Round-trip operating cost C of the paper. The competition charges nothing, so 0.
ROUND_TRIP_COST = 0.
