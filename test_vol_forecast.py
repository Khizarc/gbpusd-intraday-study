"""TEST 2 -- How big is today going to be, and how much of that is knowable at 3am?

Position size, stop distance and target distance are all the same decision in
disguise: a guess about how far GBP/USD is going to travel between the London
open and the New York afternoon. This test asks how good that guess can be,
using only information that exists before the window opens.

Two separate things get measured, because they are usually confused:

  SHAPE   Volatility has a fixed time-of-day shape -- the London/New York
          overlap is worth several times the Asian lunch hour, every single
          day. That shape is deterministic, not a forecast, and a stop quoted
          in fixed pips silently means something different at 02:00 than at
          09:00. Section 2A measures it as a multiplier per 15-minute bucket.

  LEVEL   On top of the shape, whole days are quiet or busy together --
          volatility clusters. Section 2C asks how much of TODAY's level is
          predictable from yesterday's, from the last week's, and from the
          Asian session that has just finished, in a HAR-style regression
          (Corsi 2009: a cascade of daily, weekly and monthly lags).

Discipline applied throughout:

  no lookahead   Every regressor closes before 03:00 New York on the day it
                 predicts. The Asian term is the whole point of the test -- it
                 is the only predictor that carries information from today.
  honest errors  Realized volatility is strongly autocorrelated, so OLS
                 standard errors are meaningless here; all t-statistics are
                 Newey-West.
  honest fit     In-sample R-squared on a persistent series is close to free.
                 The number that decides whether this is real is the
                 out-of-sample R-squared on 2024-2025, from coefficients
                 fitted only on 2020-2023.
  the 2023 hole  The degraded Feb-Jul 2023 window is cut from the data, so a
                 22-day lookback taken across it would silently splice January
                 onto August. Those rows are dropped rather than spliced.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxlib import (PIP, load_m1, minute_returns, newey_west_ols,
                   session_day, tod_profile)

# Windows on the New York clock. Nothing is held past DAY_CLOSE.
ASIA_START, ASIA_END = 19, 3        # 19:00 previous day -> 02:59
DAY_START, DAY_CLOSE = 3, 16        # 03:00 -> 15:59, the daytrading window

ASIA_MINUTES = 8 * 60
DAY_MINUTES = (DAY_CLOSE - DAY_START) * 60
MIN_COVERAGE = 0.70                 # a window this empty cannot be measured

BUCKET = "15min"
NW_LAGS = 10                        # Newey-West bandwidth on daily data
OOS_FROM = 2024
MAX_LOOKBACK_SPAN_DAYS = 40         # 22 trading days should not span more


# --------------------------------------------------------------------------
# 2A / 2B  the time-of-day shape
# --------------------------------------------------------------------------

def deseasonalize_check(r: pd.DataFrame, prof: pd.DataFrame) -> pd.Series:
    """Apply a shape to returns and report the RMS left in each bucket.

    Applied to the returns the shape was fitted on this is a tautology: the
    answer is 1.000 everywhere by construction. It is only informative when
    `prof` was fitted on different years than `r` covers -- which is how 2B
    calls it. A holdout bucket that comes back at 1.20 means the shape
    understates that time of day by 20% in years it never saw.
    """
    weekday = r[r.index.dayofweek != 6]
    bucket = pd.Index(weekday.index.floor(BUCKET).time)

    unit = prof["mult"].reindex(bucket).to_numpy() * prof.attrs["grand_rms"]
    scaled = weekday["ret"].to_numpy() / unit
    return pd.Series(scaled, index=bucket).pow(2).groupby(level=0).mean().pow(0.5)


# --------------------------------------------------------------------------
# 2C  daily realized volatility and its predictors
# --------------------------------------------------------------------------

def window_stats(r: pd.DataFrame, start: int, end: int, roll_evening: bool,
                 expected: int, label: str) -> pd.DataFrame:
    """Realized vol and high-low range for one intraday window, per session day.

    Realized vol is the root sum of squared 1-minute returns -- the standard
    estimator, and the one the HAR literature is written in. The range is
    carried alongside because that is what a trader actually quotes a stop in.
    """
    hour = r.index.hour
    inside = (hour >= start) & (hour < end) if start < end else (hour >= start) | (hour < end)
    w = r[inside]

    day = session_day(w.index, roll_evening=roll_evening)
    g = w.groupby(day)

    out = pd.DataFrame({
        f"rv_{label}": np.sqrt(g["ret"].apply(lambda x: (x ** 2).sum())) * PIP,
        f"range_{label}": (g["high"].max() - g["low"].min()) * PIP,
        f"bars_{label}": g.size(),
    })
    out.index.name = "day"
    return out[out[f"bars_{label}"] >= MIN_COVERAGE * expected]


def build_panel(r: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day: today's outcome plus lags that all precede it."""
    asia = window_stats(r, ASIA_START, ASIA_END, True, ASIA_MINUTES, "asia")
    day = window_stats(r, DAY_START, DAY_CLOSE, False, DAY_MINUTES, "day")

    panel = day.join(asia, how="inner").sort_index()
    panel = panel[pd.DatetimeIndex(panel.index).dayofweek <= 4]

    # Logs throughout: realized vol is right-skewed and roughly lognormal, so
    # the regression is both better specified and easier to read in logs.
    panel["y"] = np.log(panel["rv_day"])
    panel["x_asia"] = np.log(panel["rv_asia"])

    lag1 = panel["y"].shift(1)
    lag5 = panel["y"].shift(1).rolling(5).mean()
    lag22 = panel["y"].shift(1).rolling(22).mean()
    panel["x_lag1"], panel["x_week"], panel["x_month"] = lag1, lag5, lag22

    # Drop any row whose 22-day window silently spans the excised 2023 hole.
    idx = pd.DatetimeIndex(panel.index)
    span = pd.Series(idx, index=panel.index).diff(22).dt.days
    panel["lookback_days"] = span
    panel = panel[span <= MAX_LOOKBACK_SPAN_DAYS]

    panel["year"] = pd.DatetimeIndex(panel.index).year
    return panel.dropna(subset=["y", "x_asia", "x_lag1", "x_week", "x_month"])


MODELS = {
    "constant only": [],
    "yesterday": ["x_lag1"],
    "HAR (day+week+month)": ["x_lag1", "x_week", "x_month"],
    "Asia only": ["x_asia"],
    "HAR + Asia": ["x_lag1", "x_week", "x_month", "x_asia"],
}


def design(panel: pd.DataFrame, cols: list) -> np.ndarray:
    X = np.ones((len(panel), 1))
    if cols:
        X = np.hstack([X, panel[cols].to_numpy()])
    return X


def fit_and_score(panel: pd.DataFrame, cols: list, is_mask: np.ndarray,
                  oos_mask: np.ndarray) -> dict:
    """Fit on the training years only, then score the untouched holdout.

    Out-of-sample R-squared is measured against the IN-SAMPLE mean, not the
    holdout's own mean. Using the holdout mean would hand the benchmark
    information the model was never given, and quietly flatter every model.
    """
    y = panel["y"].to_numpy()
    X = design(panel, cols)

    fit = newey_west_ols(y[is_mask], X[is_mask], NW_LAGS)
    pred = X[oos_mask] @ fit["beta"]
    resid = y[oos_mask] - pred

    benchmark = y[is_mask].mean()
    ss_res = float(resid @ resid)
    ss_tot = float(((y[oos_mask] - benchmark) ** 2).sum())

    fit["r2_oos"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    fit["rmse_oos_logs"] = float(np.sqrt(np.mean(resid ** 2)))
    fit["cols"] = cols
    return fit


# --------------------------------------------------------------------------
# 2D  the sizing table
# --------------------------------------------------------------------------

def sizing_table(panel: pd.DataFrame, mask: np.ndarray, label: str) -> pd.DataFrame:
    """Asian-range decile -> what the London/NY window actually did.

    The regression says how much is predictable; this says what to do with it.
    Deciles are cut on the sample being shown, so the holdout table is built
    without reference to the training years.
    """
    sub = panel[mask].copy()
    sub["decile"] = pd.qcut(sub["range_asia"], 10, labels=False, duplicates="drop") + 1

    out = sub.groupby("decile").agg(
        asia_range=("range_asia", "median"),
        day_range=("range_day", "median"),
        day_range_p25=("range_day", lambda x: x.quantile(0.25)),
        day_range_p75=("range_day", lambda x: x.quantile(0.75)),
        n=("range_day", "size"),
    )
    out["day_over_asia"] = (out["day_range"] / out["asia_range"]).round(2)
    out.attrs["label"] = label
    return out.round(1)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def show(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_fit(name: str, fit: dict, cols: list) -> None:
    names = ["const", *cols]
    print(f"  {name}")
    print(f"    {'term':<10s} {'beta':>9s} {'NW se':>9s} {'t':>8s} {'p':>8s}")
    for i, term in enumerate(names):
        print(f"    {term:<10s} {fit['beta'][i]:>9.4f} {fit['se'][i]:>9.4f} "
              f"{fit['t'][i]:>8.2f} {fit['p'][i]:>8.4f}")
    print(f"    in-sample R2 {fit['r2']:.4f}   OUT-OF-SAMPLE R2 {fit['r2_oos']:.4f}"
          f"   (n_train {fit['n']:,})")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-profile", action="store_true",
                    help="Print all 96 time-of-day buckets, not just the summary.")
    args = ap.parse_args()

    df = load_m1()
    r = minute_returns(df)

    show("TEST 2  INTRADAY VOLATILITY -- its fixed shape, and its forecastable level")
    print(f"{len(r):,} usable 1-minute returns, New York clock")
    print(f"daytrading window {DAY_START:02d}:00-{DAY_CLOSE - 1:02d}:59, "
          f"Asian reference window {ASIA_START:02d}:00-{ASIA_END - 1:02d}:59")

    prof = tod_profile(r)

    show("2A  TIME-OF-DAY SHAPE  (multiplier on the average minute, Mon-Fri)")
    print("mult = how many times the average minute's volatility this bucket is.")
    print("A stop of X pips at the average minute is the same stop as mult*X here.")
    print()
    if args.full_profile:
        print(prof.round(3).to_string())
    else:
        print("  busiest ten buckets:")
        print(prof.nlargest(10, "mult").round(3).to_string())
        print()
        print("  quietest ten buckets:")
        print(prof.nsmallest(10, "mult").round(3).to_string())
        print()
        print("  (--full-profile prints all 96)")
    print()
    print(f"peak/trough ratio {prof['mult'].max() / prof['mult'].min():.1f}x -- "
          "the same nominal stop is a very different stop across the day")

    show("2B  IS THE SHAPE STABLE ENOUGH TO TRADE ON?  (out-of-sample)")
    print(f"Fit the multipliers on {OOS_FROM - 4}-{OOS_FROM - 1} only, then apply them")
    print(f"to {OOS_FROM}-2025. If the shape is stable, the RMS of the rescaled")
    print("holdout returns is 1.000 in every bucket. Deviations are how wrong a")
    print("stop scaled by last year's profile would be this year.")
    print()
    years = r.index.year
    prof_train = tod_profile(r[years < OOS_FROM])
    flat = deseasonalize_check(r[years >= OOS_FROM], prof_train)

    print(f"  holdout RMS after rescaling: min {flat.min():.3f}  "
          f"max {flat.max():.3f}  median {flat.median():.3f}")
    print(f"  the level shift ({flat.median():.3f}) is {OOS_FROM}-2025 simply being "
          "a calmer regime overall;")
    print("  what matters for the SHAPE is the spread around that level:")
    rel = flat / flat.median()
    print(f"    relative to the holdout's own level: min {rel.min():.3f}  "
          f"max {rel.max():.3f}")
    print(f"    worst bucket {rel.sub(1).abs().idxmax()} is off by "
          f"{100 * abs(rel - 1).max():.1f}%")
    print(f"    buckets within 10% of their fitted shape: "
          f"{100 * (abs(rel - 1) < 0.10).mean():.0f}% of 96")

    panel = build_panel(r)
    is_mask = (panel["year"] < OOS_FROM).to_numpy()
    oos_mask = ~is_mask

    show("2C  IS TODAY'S LEVEL PREDICTABLE?  HAR regression on log realized vol")
    print(f"target: log realized vol over {DAY_START:02d}:00-{DAY_CLOSE - 1:02d}:59, "
          f"in pips")
    print(f"panel {len(panel):,} trading days  ->  train {int(is_mask.sum()):,} "
          f"(<{OOS_FROM})   holdout {int(oos_mask.sum()):,} (>={OOS_FROM})")
    print("every regressor is complete before 03:00 on the day it predicts.")
    print("standard errors Newey-West, {} lags.".format(NW_LAGS))
    print()

    results = {}
    for name, cols in MODELS.items():
        fit = fit_and_score(panel, cols, is_mask, oos_mask)
        results[name] = fit
        print_fit(name, fit, cols)

    print("  MODEL COMPARISON on the untouched holdout")
    print(f"    {'model':<24s} {'OOS R2':>9s} {'IS R2':>9s}")
    for name, fit in results.items():
        print(f"    {name:<24s} {fit['r2_oos']:>9.4f} {fit['r2']:>9.4f}")
    gain = results["HAR + Asia"]["r2_oos"] - results["HAR (day+week+month)"]["r2_oos"]
    print()
    print(f"    value added by watching the Asian session: {gain:+.4f} OOS R2")
    print("    (the honest read of that number is in the summary at the bottom)")

    show("2D  SIZING TABLE  Asian range decile -> London/NY range, in pips")
    for label, mask in (("TRAIN 2020-2023", is_mask), ("HOLDOUT 2024-2025", oos_mask)):
        print(f"  {label}")
        print(sizing_table(panel, mask, label).to_string().replace("\n", "\n  "))
        print()

    corr_is = np.corrcoef(panel.loc[is_mask, "range_asia"],
                          panel.loc[is_mask, "range_day"])[0, 1]
    corr_oos = np.corrcoef(panel.loc[oos_mask, "range_asia"],
                           panel.loc[oos_mask, "range_day"])[0, 1]
    print(f"  Asian range vs London/NY range, Pearson r: "
          f"train {corr_is:.3f}   holdout {corr_oos:.3f}")

    print()
    print("-" * 78)
    print("Reading it: 2A is a certainty and should be applied to every stop and")
    print("target you quote. 2C is a genuine but partial forecast -- compare the")
    print("HAR + Asia holdout R2 against 'yesterday' alone to see how much of it is")
    print("just persistence you already had. 2D is the version you can trade: it")
    print("says what range to plan for, and the p25/p75 columns say how wrong that")
    print("plan is routinely allowed to be.")


if __name__ == "__main__":
    main()
