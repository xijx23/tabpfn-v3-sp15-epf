from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
TZ = 'America/Los_Angeles'
NODES = ['TH_SP15_GEN-APND', 'TH_NP15_GEN-APND', 'TH_ZP26_GEN-APND']


def local_date(value):
    return pd.Timestamp(value).tz_localize(TZ)


def drop_repeated_local_hour(frame):
    """Fall-back day: keep the first local 01:00 (PDT) and drop the repeated one (PST)."""
    local = frame.index.tz_convert(TZ)
    keys = pd.Series(local.strftime('%Y-%m-%d %H'))
    return frame.loc[~keys.duplicated(keep='first').to_numpy()]


def chunks(start, end):
    """Local calendar months; requests must remain within 31 elapsed days."""
    current = local_date(start)
    stop = local_date(end)
    if current >= stop:
        raise ValueError('start must precede exclusive end')
    while current < stop:
        following = min(current + pd.offsets.MonthBegin(1), stop)
        if following - current > pd.Timedelta(days=31):
            following = current + pd.DateOffset(days=30)
        yield current, following
        current = following


def api_time(value):
    return value.tz_convert('UTC').strftime('%Y%m%dT%H:%M-0000')
