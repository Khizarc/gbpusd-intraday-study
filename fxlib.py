"""Shared loading, session and statistics helpers for the GBP/USD studies.

Deliberately dependency-light: pandas, numpy and the standard library only.
Every estimator here is written out rather than imported so the assumptions
behind a number are visible at the point it is produced.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PARQUET = HERE / "gbpusd_m1.parquet"

PIP = 10_000
NY = "America/New_York"

# Session boundaries live in the study that uses them, not here: each test
# defines the window it is actually measuring, right next to the code that
# measures it. All of them are on the New York clock, and none hold past the
# New York afternoon.


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_m1(path: Path = PARQUET, require_clean: bool = True) -> pd.DataFrame:
    """Load the M1 parquet, checking how it was built.

    load_data.py stamps its build report into the parquet metadata. Reading it
    back turns "did I remember to rebuild with the right flags?" from a thing
    you have to remember into a thing that fails loudly.
    """
    df = pd.read_parquet(path)

    try:
        from load_data import read_build_metadata
        meta = read_build_metadata(path)
    except Exception:
        meta = {}

    if require_clean and meta:
        problems = []
        if not meta.get("duplicates_dropped", False):
            problems.append("duplicate timestamps were kept")
        if not meta.get("exclusion_applied", False):
            problems.append("the degraded 2023 H1 window was kept")
        if problems:
            raise ValueError(
                f"{path.name} was built with: {'; '.join(problems)}. "
                "Rebuild with `python load_data.py` (cleaning is the default), "
                "or pass require_clean=False to analyse the raw feed on purpose."
            )
    elif require_clean and not meta:
        print(f"note: {path.name} carries no build metadata -- it predates the "
              "current load_data.py. Rebuild to get provenance checking.\n")

    return df


def weekday_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Index on the NY clock and drop the thin Sunday-evening reopen."""
    s = df.set_index("timestamp_ny").sort_index()
    return s[s.index.dayofweek != 6]


def session_day(index: pd.DatetimeIndex, roll_evening: bool = False) -> np.ndarray:
    """Naive session-date label. With roll_evening, bars at/after 19:00 roll on.

    Kept naive on purpose: adding a day to a tz-aware midnight lands on 23:00
    or 01:00 across a DST boundary, which would corrupt the grouping.
    """
    base = pd.to_datetime(index.date)
    if not roll_evening:
        return base.values
    return (base + pd.to_timedelta((index.hour >= 19).astype(int), unit="D")).values


# --------------------------------------------------------------------------
# returns
# --------------------------------------------------------------------------

def log_returns(s: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close log returns with contiguity and segment bookkeeping.

    A return is only defined across two adjacent minutes that are actually
    adjacent in time. Anything spanning a weekend, a holiday or a feed gap is
    left NaN: it is a repricing, not a one-minute move, and letting it into a
    variance estimate contaminates every horizon at once.

    `segment` numbers the maximal runs of consecutive minutes, which is the
    unit any multi-minute window has to stay inside.
    """
    out = pd.DataFrame(index=s.index)
    step = s.index.to_series().diff()
    contiguous = step == pd.Timedelta(minutes=1)

    out["logp"] = np.log(s["close"].to_numpy())
    out["ret"] = out["logp"].diff().where(contiguous)
    out["segment"] = (~contiguous).cumsum()
    out["hour"] = s.index.hour
    out["minute_of_day"] = s.index.hour * 60 + s.index.minute
    out["day"] = session_day(s.index)
    return out


def minute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """1-minute log returns on the NY clock, gaps dropped, OHLC carried along.

    The lighter cousin of log_returns: no segment bookkeeping, because callers
    that only ever look inside one session never need it. Gap-spanning rows are
    dropped outright rather than left NaN.
    """
    s = df.set_index("timestamp_ny").sort_index()
    contiguous = s.index.to_series().diff() == pd.Timedelta(minutes=1)

    out = pd.DataFrame(index=s.index)
    out["ret"] = np.log(s["close"]).diff().where(contiguous)
    out["open"], out["high"] = s["open"], s["high"]
    out["low"], out["close"] = s["low"], s["close"]
    return out.dropna(subset=["ret"])


def tod_profile(r: pd.DataFrame, bucket: str = "15min") -> pd.DataFrame:
    """Volatility multiplier per bucket of the New York clock.

    Built from mean squared return rather than mean absolute return: it is the
    variance that scales, and a stop is a distance, so the reported multiplier
    is the square root -- directly the factor to scale a pip distance by.

    Sunday is excluded. Its evening is thin enough to drag the 19:00-23:59
    buckets down and make Monday look calmer than it trades.
    """
    weekday = r[r.index.dayofweek != 6]
    key = weekday.index.floor(bucket).time
    ret2 = weekday["ret"] ** 2

    grand = ret2.mean()
    per_bucket = ret2.groupby(key).mean()

    prof = pd.DataFrame({
        "mult": np.sqrt(per_bucket / grand),
        "rms_pips": np.sqrt(per_bucket) * PIP,
        "n": ret2.groupby(key).size(),
    })
    prof.index.name = "ny_time"
    # The multiplier is relative to the average minute, so undoing the shape
    # needs the average minute's own size as well as the ratio.
    prof.attrs["grand_rms"] = float(np.sqrt(grand))
    prof.attrs["bucket"] = bucket
    return prof


def deseasonalize(r: pd.DataFrame, prof: pd.DataFrame) -> pd.Series:
    """Rescale returns to a common volatility unit using a time-of-day profile.

    Everything that compares a move against "how big is a normal move" needs
    this first. A 15-pip minute at 09:30 is unremarkable; the same 15 pips at
    01:00 is a violent event, and any test that does not divide out the shape
    will simply rediscover the shape and call it a signal.
    """
    key = pd.Index(r.index.floor(prof.attrs["bucket"]).time)
    unit = prof["mult"].reindex(key).to_numpy() * prof.attrs["grand_rms"]
    return pd.Series(r["ret"].to_numpy() / unit, index=r.index, name="ret_std")


def horizon_returns(r: pd.DataFrame, q: int) -> pd.DataFrame:
    """q-minute overlapping log returns that lie wholly inside one segment.

    Labelled by the START of the window, because that is the decision point:
    "I am looking at 09:15 -- what do the next q minutes do?"
    """
    logp = r["logp"].to_numpy()
    segment = r["segment"].to_numpy()
    n = len(r)
    if n <= q:
        return pd.DataFrame(columns=["ret_q", "hour", "day"])

    valid = segment[q:] == segment[:-q]
    ret_q = logp[q:] - logp[:-q]

    start = r.iloc[:-q]
    return pd.DataFrame({
        "ret_q": ret_q[valid],
        "hour": start["hour"].to_numpy()[valid],
        "day": start["day"].to_numpy()[valid],
    })


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def norm_sf(z: float) -> float:
    """Upper-tail standard normal probability, without pulling in scipy."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_p(z: float) -> float:
    return 2.0 * norm_sf(abs(z))


def stars(p: float) -> str:
    """Conventional significance marks. Read them with the multiple-testing
    caveat each caller prints -- a table of 24 hours will show ~1 false star."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int) -> dict:
    """OLS with Newey-West (Bartlett) heteroskedasticity- and
    autocorrelation-consistent standard errors.

    Overlapping-window regressors and clustered volatility both make plain OLS
    standard errors far too small here, which is the usual way an intraday
    "signal" turns out to be nothing.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)          # Bartlett kernel
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)

    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)

    return {
        "beta": beta, "se": se, "t": tstat,
        "p": np.array([two_sided_p(t) if np.isfinite(t) else np.nan for t in tstat]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": n, "k": k, "resid": resid,
    }


def day_block_bootstrap(day_labels: np.ndarray, statistic, n_boot: int = 1000,
                        seed: int = 42, alpha: float = 0.05) -> tuple:
    """Percentile CI from resampling whole trading days with replacement.

    Intraday observations inside one day are heavily dependent and overlapping
    windows are dependent by construction, so the resampling unit has to be the
    day. `statistic(idx)` receives positional indices into the original arrays
    and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    days, inverse = np.unique(day_labels, return_inverse=True)

    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(days)))
    ends = np.append(starts[1:], len(order))
    per_day = [order[a:b] for a, b in zip(starts, ends)]

    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        draws[b] = statistic(np.concatenate([per_day[i] for i in pick]))

    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


def split_is_oos(years: np.ndarray, oos_from: int = 2024) -> tuple:
    """Boolean masks for an in-sample / out-of-sample split by calendar year.

    A single fixed split, chosen once and stated up front. Every number in
    these studies is reported on both halves so a result that only exists in
    the half it was found in is visible as such.
    """
    years = np.asarray(years)
    return years < oos_from, years >= oos_from
