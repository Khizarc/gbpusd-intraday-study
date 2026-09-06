"""TEST 5 -- What does a GBP/USD day actually look like from the inside?

Tests 1-4 said what the price process is. This asks what a DAY is: when the
high and the low get put in, how much of the range is left by lunchtime,
whether a move travels in a straight line or grinds back and forth, and
whether any of that is knowable while the day is still running.

This is the descriptive layer the other tests keep leaning on. A daytrader's
real questions are shaped like this, and none of them are about direction:

  "Is it too late to be looking for an entry?"     -> 5A, 5B
  "Is today a trend day or a chop day?"            -> 5C
  "Could I have known that by 9am?"                -> 5D

The measurements, all over 03:00-15:59 New York, Monday to Friday:

  range         high - low over the session
  path          the sum of every absolute minute move -- how far price
                actually travelled, as opposed to how far it got
  efficiency    |net move| / path. 1.0 is a straight line from open to close;
                0.05 is a day that went nowhere the long way round. This is
                Kaufman's efficiency ratio, and it is the single number that
                separates a trend day from a chop day.
  CLV           where in the range the session closed, 0 at the low and 1 at
                the high. Days that close on their extreme are days that
                trended into the close.

5D is the only part making a claim rather than a description, so it is the
only part carrying confidence intervals. They are day-block bootstraps, and
the split it uses -- London morning versus New York afternoon -- is fixed in
advance rather than chosen after looking.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxlib import PIP, day_block_bootstrap, load_m1, minute_returns, session_day

DAY_START, DAY_CLOSE = 3, 16        # 03:00 -> 15:59 New York
SPLIT_HOUR = 9                      # London morning | New York afternoon
MIN_BARS = 400

# A day whose net move is at least this share of its range is a trend day.
# Chosen once, up front; 5C reports the whole distribution so the choice of
# cut can be second-guessed against the data rather than taken on faith.
TREND_CUT = 0.50
CHOP_CUT = 0.20
N_BOOT = 2000


# --------------------------------------------------------------------------
# per-day measurements
# --------------------------------------------------------------------------

def session_frame(r: pd.DataFrame) -> pd.DataFrame:
    """The daytrading window only, weekdays only, labelled by session day."""
    hour = r.index.hour
    w = r[(hour >= DAY_START) & (hour < DAY_CLOSE)].copy()
    w["day"] = session_day(w.index)
    w["hour"] = hour[(hour >= DAY_START) & (hour < DAY_CLOSE)]
    w = w[pd.DatetimeIndex(w["day"]).dayofweek <= 4]

    counts = w.groupby("day")["ret"].transform("size")
    return w[counts >= MIN_BARS]


def day_metrics(w: pd.DataFrame) -> pd.DataFrame:
    """One row per session: geometry, path, efficiency, and extreme timing."""
    g = w.groupby("day", sort=True)

    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "path": g["ret"].apply(lambda x: x.abs().sum()) * PIP,
        "bars": g.size(),
    })
    out["range"] = (out["high"] - out["low"]) * PIP
    out["net"] = (out["close"] - out["open"]) * PIP

    # First occurrence of each extreme, as minutes since the session open.
    mod = w.index.hour * 60 + w.index.minute - DAY_START * 60
    tmp = w.assign(mod=mod)
    out["t_high"] = tmp.loc[tmp.groupby("day")["high"].idxmax()].set_index("day")["mod"]
    out["t_low"] = tmp.loc[tmp.groupby("day")["low"].idxmin()].set_index("day")["mod"]
    out["t_range_complete"] = out[["t_high", "t_low"]].max(axis=1)

    out["efficiency"] = out["net"].abs() / out["path"].replace(0, np.nan)
    out["range_eff"] = out["net"].abs() / out["range"].replace(0, np.nan)
    out["clv"] = ((out["close"] - out["low"])
                  / (out["high"] - out["low"]).replace(0, np.nan))
    out["direction"] = np.where(out["net"] >= 0, "up", "down")
    out["year"] = pd.DatetimeIndex(out.index).year
    out.index.name = "day"
    return out


def range_built_curve(w: pd.DataFrame) -> pd.DataFrame:
    """Share of the session's final range already established by each hour.

    Running high-minus-low within the day, divided by the day's finished
    range, then averaged across days. This is the "am I early or late?" curve.
    """
    g = w.groupby("day", sort=False)
    running = (g["high"].cummax() - g["low"].cummin()) * PIP
    final = g["range_final"].transform("first") if "range_final" in w else None

    frame = pd.DataFrame({"day": w["day"].to_numpy(), "hour": w["hour"].to_numpy(),
                          "running": running.to_numpy()})
    finals = frame.groupby("day")["running"].transform("max")
    frame["frac"] = frame["running"] / finals.replace(0, np.nan)

    out = frame.groupby("hour")["frac"].agg(["mean", "median"]).round(3)
    out.columns = ["mean_share_of_range", "median_share_of_range"]
    return out


def _path_and_net(close: np.ndarray, step: int) -> tuple:
    """Path length and net move in pips, sampling the close every `step` minutes."""
    sub = close[::step]
    if sub.size < 3:
        return np.nan, np.nan
    lp = np.log(sub)
    return float(np.abs(np.diff(lp)).sum() * PIP), float(abs(lp[-1] - lp[0]) * PIP)


def efficiency_by_scale(w: pd.DataFrame, steps=(1, 5, 15, 30, 60)) -> pd.DataFrame:
    """The efficiency ratio at several sampling intervals.

    The ratio is NOT scale-free and it is routinely quoted as though it were.
    Path length grows roughly linearly as you sample more finely, while the net
    move does not move at all, so |net| / path falls towards zero purely by
    sampling faster. Quoting "efficiency = 0.04" without the interval it was
    measured at says nothing. Reported across intervals so the reader can see
    how much of the number is the market and how much is the ruler.
    """
    rows = []
    for step in steps:
        effs, paths = [], []
        for _, chunk in w.groupby("day", sort=False):
            path, net = _path_and_net(chunk["close"].to_numpy(), step)
            if np.isfinite(path) and path > 0:
                effs.append(net / path)
                paths.append(path)
        rows.append({
            "sample_min": step,
            "median_path_pips": round(float(np.median(paths)), 1),
            "median_efficiency": round(float(np.median(effs)), 3),
            "mean_efficiency": round(float(np.mean(effs)), 3),
        })
    return pd.DataFrame(rows).set_index("sample_min")


def hourly_efficiency(w: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Efficiency ratio within each single hour, pooled over days.

    Sampled at `step` minutes rather than 1, so the number is about the hour's
    character instead of about the sampling rate. Every hour uses the same
    ruler, so the comparison down the column is meaningful even though the
    absolute level is not.
    """
    rows = {}
    for (day, hour), chunk in w.groupby(["day", "hour"], sort=False):
        path, net = _path_and_net(chunk["close"].to_numpy(), step)
        if np.isfinite(path) and path > 0:
            rows.setdefault(hour, []).append(net / path)

    out = pd.DataFrame({
        "mean_efficiency": {h: np.mean(v) for h, v in rows.items()},
        "median_efficiency": {h: np.median(v) for h, v in rows.items()},
        "n": {h: len(v) for h, v in rows.items()},
    }).sort_index().round(3)
    out.index.name = "hour"
    return out


# --------------------------------------------------------------------------
# 5D  is any of it knowable early?
# --------------------------------------------------------------------------

def split_metrics(w: pd.DataFrame) -> pd.DataFrame:
    """Same measurements, computed separately either side of SPLIT_HOUR."""
    parts = {}
    for label, mask in (("early", w["hour"] < SPLIT_HOUR),
                        ("late", w["hour"] >= SPLIT_HOUR)):
        g = w[mask].groupby("day", sort=True)
        net = (g["close"].last() - g["open"].first()) * PIP
        path = g["ret"].apply(lambda x: x.abs().sum()) * PIP
        parts[f"net_{label}"] = net
        parts[f"path_{label}"] = path
        parts[f"range_{label}"] = (g["high"].max() - g["low"].min()) * PIP
        parts[f"eff_{label}"] = net.abs() / path.replace(0, np.nan)
    out = pd.DataFrame(parts).dropna()
    out["same_direction"] = np.sign(out["net_early"]) == np.sign(out["net_late"])
    out["year"] = pd.DatetimeIndex(out.index).year
    return out


def boot_ci(frame: pd.DataFrame, statistic, seed: int = 42) -> tuple:
    return day_block_bootstrap(frame.index.to_numpy(), statistic,
                               n_boot=N_BOOT, seed=seed)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def show(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_m1()
    r = minute_returns(df)
    w = session_frame(r)
    m = day_metrics(w)

    show("TEST 5  THE ANATOMY OF A DAY  "
         f"({DAY_START:02d}:00-{DAY_CLOSE - 1:02d}:59 New York, Mon-Fri)")
    print(f"{len(m):,} sessions")
    print(f"median range {m['range'].median():.1f} pips   "
          f"median path travelled {m['path'].median():.1f} pips   "
          f"median |net| {m['net'].abs().median():.1f} pips")
    print(f"price travels {m['path'].median() / m['range'].median():.1f}x its own "
          "range to produce it -- measured minute by minute, which is a")
    print("sampling-dependent statement; 5B puts a scale on it.")

    show("5A  WHEN THE HIGH AND THE LOW GET PUT IN")
    print("Hour of the first occurrence of each session extreme.")
    print()
    hi = pd.Series(m["t_high"] // 60 + DAY_START).value_counts().sort_index()
    lo = pd.Series(m["t_low"] // 60 + DAY_START).value_counts().sort_index()
    tbl = pd.DataFrame({"session_high": hi, "session_low": lo}).fillna(0).astype(int)
    tbl["high_pct"] = (100 * tbl["session_high"] / len(m)).round(1)
    tbl["low_pct"] = (100 * tbl["session_low"] / len(m)).round(1)
    tbl.index.name = "ny_hour"
    print(tbl.to_string())
    print()
    for h in (5, 8, 10, 12, 14):
        done = float((m["t_range_complete"] <= (h - DAY_START) * 60).mean())
        print(f"  by {h:02d}:00 the day's full range is already set on "
              f"{100 * done:4.1f}% of days")

    show("5B  HOW MUCH OF THE RANGE IS LEFT")
    print("Average share of the session's finished range already established by")
    print("the end of each hour. This is the cost of waiting for confirmation.")
    print()
    print(range_built_curve(w).to_string())
    print()
    print("  EFFICIENCY vs THE RULER USED TO MEASURE IT")
    print("  |net| / path at several sampling intervals. The fall as you sample")
    print("  faster is arithmetic, not information -- which is why the trend/chop")
    print("  split in 5C uses |net| / range, the scale-free version, instead.")
    print()
    print(efficiency_by_scale(w).to_string().replace("\n", "\n  "))
    print()
    print("  efficiency WITHIN each hour, sampled at 5 min (same ruler on every")
    print("  row, so the column compares even though its level means little):")
    print(hourly_efficiency(w).to_string().replace("\n", "\n  "))

    show("5C  TREND DAYS AND CHOP DAYS")
    print("range_eff = |net move| / range. A day closing on its high has 1.0;")
    print("a day that ends where it started has 0.0.")
    print()
    q = m["range_eff"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3)
    print("  range_eff percentiles: "
          + "  ".join(f"p{int(100 * k)} {v}" for k, v in q.items()))
    print("  (|net| / path is deliberately NOT used as the cut -- see 5B)")
    print()

    trend = m["range_eff"] >= TREND_CUT
    chop = m["range_eff"] <= CHOP_CUT
    label = np.where(trend, "trend", np.where(chop, "chop", "mixed"))
    m["kind"] = label

    kinds = m.groupby("kind").agg(
        days=("range", "size"),
        share_pct=("range", lambda x: round(100 * len(x) / len(m), 1)),
        median_range=("range", "median"),
        median_path=("path", "median"),
        median_t_range_complete=("t_range_complete", "median"),
    ).round(2)
    print(kinds.to_string())
    print()
    print(f"  a trend day (>= {TREND_CUT:.0%} of range captured) happens "
          f"{100 * trend.mean():.0f}% of the time;")
    print(f"  a chop day (<= {CHOP_CUT:.0%}) happens {100 * chop.mean():.0f}% "
          "of the time.")
    print()
    print("  CLV -- where the session closes within its range:")
    clv = m["clv"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3)
    print("    " + "  ".join(f"p{int(100 * k)} {v}" for k, v in clv.items()))
    print(f"    closes in the top or bottom 20% of the range: "
          f"{100 * ((m['clv'] > 0.8) | (m['clv'] < 0.2)).mean():.0f}% of days")

    s = split_metrics(w)
    show(f"5D  COULD YOU HAVE KNOWN BY {SPLIT_HOUR:02d}:00?")
    print(f"Split each session at {SPLIT_HOUR:02d}:00 -- London morning, then the")
    print("New York afternoon -- and ask whether the morning tells you anything")
    print("about the afternoon. 95% CIs are day-block bootstraps.")
    print(f"  {len(s):,} sessions with both halves measurable")
    print()

    eff_e = s["eff_early"].to_numpy()
    eff_l = s["eff_late"].to_numpy()
    rho = float(np.corrcoef(eff_e, eff_l)[0, 1])
    lo, hi = boot_ci(s, lambda idx: float(np.corrcoef(eff_e[idx], eff_l[idx])[0, 1]),
                     args.seed)
    print(f"  does a trending morning mean a trending afternoon?")
    print(f"    corr(early efficiency, late efficiency) = {rho:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"    -> {'yes, weakly' if lo > 0 else 'no'}")
    print()

    same = s["same_direction"].to_numpy()
    p = float(same.mean())
    lo, hi = boot_ci(s, lambda idx: float(same[idx].mean()), args.seed)
    print(f"  does the afternoon continue the morning's direction?")
    print(f"    P(same direction) = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"vs 0.500 by chance")
    print(f"    -> {'directional edge' if lo > 0.5 or hi < 0.5 else 'a coin flip'}")
    print()

    # The version that would actually be traded: only act when the morning was
    # decisive. Conditioning on a strong morning is the natural next thing to
    # try, and it is also the natural way to find a pattern that is not there.
    strong = s[s["eff_early"] >= s["eff_early"].quantile(0.75)]
    p_s = float(strong["same_direction"].mean())
    lo_s, hi_s = boot_ci(strong,
                         lambda idx: float(strong["same_direction"]
                                           .to_numpy()[idx].mean()), args.seed)
    print(f"  ...and when the morning was DECISIVE (top-quartile efficiency)?")
    print(f"    P(same direction) = {p_s:.3f}  95% CI [{lo_s:.3f}, {hi_s:.3f}]  "
          f"(n={len(strong):,})")
    print(f"    -> {'directional edge' if lo_s > 0.5 or hi_s < 0.5 else 'a coin flip'}")
    print()

    rng = np.corrcoef(s["range_early"], s["range_late"])[0, 1]
    print(f"  for contrast, the SIZE question: "
          f"corr(early range, late range) = {rng:+.3f}")
    print("  -- which is test 2's result showing up again: magnitude carries")
    print("     across the day, direction does not.")

    show("5E  STABILITY ACROSS THE SPLIT")
    per_year = m.groupby("year").agg(
        days=("range", "size"),
        median_range=("range", "median"),
        median_range_eff=("range_eff", "median"),
        trend_day_pct=("kind", lambda x: round(100 * (x == "trend").mean(), 1)),
    ).round(2)
    print(per_year.to_string())

    print()
    print("-" * 78)
    print("Reading it: 5A and 5B are the ones to internalise -- they say how much")
    print("of the day is already gone by the time a setup is obvious. 5C says how")
    print("often a day is worth trend-trading at all. 5D is the warning: the SIZE")
    print("of the afternoon is forecastable and its DIRECTION is not, which is the")
    print("same answer tests 1 and 3 gave, arrived at from a different direction.")


if __name__ == "__main__":
    main()
