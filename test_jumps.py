"""TEST 4 -- Is GBP/USD's movement smooth, or does it arrive in jumps?

Tests 1-3 said direction is close to unpredictable while volatility is highly
structured. This asks what the price process actually IS, because "volatility"
covers two completely different things that a stop order does not treat alike:

  DIFFUSION  price grinding continuously through every level in between. A
             stop at 15 pips gets filled at 15 pips.
  JUMPS      price relocating in a single minute with nothing traded in
             between. A stop at 15 pips gets filled wherever the other side
             reappears, and the gap is the trader's, not the broker's.

If a meaningful share of the day's variance is jumps, then every risk number
computed from realized volatility is optimistic, and every backtest that
assumes stops fill at their level -- including test 3 -- is flattered.

Two estimators, doing different jobs:

  4A/4B  Barndorff-Nielsen & Shephard. Realized variance counts everything;
         bipower variation, built from products of ADJACENT absolute returns,
         is immune to isolated jumps because a jump can only ever contaminate
         two of its terms. The gap between them is the jump component, and
         their ratio statistic says whether a given day's gap is larger than
         sampling noise.

  4C/4D  Lee & Mykland. Same idea aimed at a single minute rather than a day:
         score every return against a LOCAL bipower estimate of what a normal
         return looked like just before it, and take the extremes. With ~1.9m
         minutes the null distribution of the maximum is Gumbel, which is what
         sets the threshold -- so the family-wise error rate over the entire
         sample is the stated alpha, not per-minute.

The one thing that would break both: intraday seasonality. A 12-pip minute at
09:30 is ordinary and the same minute at 01:00 is an event, so a detector run
on raw returns simply rediscovers the time-of-day curve from test 2 and calls
every New York open a jump. Returns are divided by that curve first, and 4C
reports both so the size of that mistake is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from fxlib import (PIP, day_block_bootstrap, deseasonalize, load_m1,
                   minute_returns, session_day, tod_profile)

DAY_START, DAY_CLOSE = 3, 16       # the daytrading window, as in test 2
DAY_MINUTES = (DAY_CLOSE - DAY_START) * 60
MIN_BARS = 400                     # a day thinner than this cannot be tested

LM_WINDOW = 270                    # Lee-Mykland local window, their 1-min value
LM_ALPHA = 0.01                    # family-wise, over the whole sample
FORWARD = (5, 15, 30, 60)          # minutes to follow a jump for

# The 17:00 New York rollover is the FX day boundary: liquidity leaves almost
# entirely for an hour or two. A single tick there moves price several pips
# against a time-of-day curve that says minutes here should be tiny, so the
# detector fires constantly -- on illiquidity, not on information. Those hours
# are reported, because "do not hold through them" is a real finding, but they
# are kept out of every number about what a jump means.
ROLLOVER_HOURS = (17, 18)

MU1 = math.sqrt(2.0 / math.pi)                        # E|Z|
MU43 = 2.0 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(0.5)   # E|Z|^(4/3)


# --------------------------------------------------------------------------
# 4A/4B  daily decomposition
# --------------------------------------------------------------------------

def bns_day(ret: np.ndarray) -> dict:
    """Split one day's realized variance into continuous and jump parts.

        RV = sum r^2                       everything
        BV = mu1^-2 * sum |r_i||r_i-1|     jump-robust
        JV = max(RV - BV, 0)               what the jumps contributed

    The BNS ratio statistic tests whether JV/RV is bigger than sampling noise,
    standardised by tripower quarticity so that a volatile day does not look
    jumpy merely for being volatile. It is one-sided: only RV above BV is
    evidence of jumps, the other direction is noise.
    """
    n = ret.size
    if n < 50:
        return {}

    rv = float(np.sum(ret ** 2))
    a = np.abs(ret)

    bv = float(MU1 ** -2 * (n / (n - 1)) * np.sum(a[1:] * a[:-1]))
    tq = float(n * MU43 ** -3 * (n / (n - 2))
               * np.sum((a[2:] * a[1:-1] * a[:-2]) ** (4 / 3)))

    rj = (rv - bv) / rv if rv > 0 else np.nan
    denom = ((math.pi ** 2 / 4) + math.pi - 5) * (1.0 / n) * max(1.0, tq / bv ** 2)
    z = rj / math.sqrt(denom) if denom > 0 and np.isfinite(rj) else np.nan

    return {
        "rv_pips": math.sqrt(rv) * PIP,
        "bv_pips": math.sqrt(max(bv, 0.0)) * PIP,
        "jump_share": max(rj, 0.0) if np.isfinite(rj) else np.nan,
        "z": z,
        "n": n,
    }


def daily_decomposition(r: pd.DataFrame) -> pd.DataFrame:
    hour = r.index.hour
    w = r[(hour >= DAY_START) & (hour < DAY_CLOSE)]
    rows = []
    for day, chunk in w.groupby(session_day(w.index)):
        if len(chunk) < MIN_BARS or pd.Timestamp(day).dayofweek > 4:
            continue
        stats = bns_day(chunk["ret"].to_numpy())
        if stats:
            rows.append({"day": day, **stats})
    out = pd.DataFrame(rows)
    out["year"] = pd.DatetimeIndex(out["day"]).year
    # One-sided 1% critical value: jumps push RV above BV, never below.
    out["has_jump"] = out["z"] > 2.326
    return out


# --------------------------------------------------------------------------
# 4C  minute-level jump detection
# --------------------------------------------------------------------------

def lee_mykland(ret: pd.Series, window: int = LM_WINDOW,
                alpha: float = LM_ALPHA) -> pd.DataFrame:
    """Flag individual minutes whose return dwarfs the local volatility.

        sigma_i^2 = (1/(K-2)) * sum_{j=i-K+2}^{i-1} |r_j| |r_j-1|
        L_i       = r_i / sigma_i

    sigma is built from returns STRICTLY BEFORE i, so a jump never inflates
    its own benchmark. The threshold comes from the Gumbel limit of max|L|:
    with n minutes, C_n and S_n below centre and scale that maximum, and the
    stated alpha is the chance of ONE false positive across the whole sample,
    not one per minute. At n ~ 1.9m that puts the bar near 7.4 local sigmas.
    """
    r = ret.to_numpy()
    n = r.size
    a = np.abs(r)

    # Local bipower, ending at i-1. rolling(...).shift(1) is what enforces the
    # "strictly before" -- without it a jump would help set its own threshold.
    prod = pd.Series(a[1:] * a[:-1], index=ret.index[1:])
    sigma2 = prod.rolling(window - 2).sum().shift(1) / (window - 2)
    sigma = np.sqrt(sigma2.reindex(ret.index).to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(sigma > 0, r / sigma, np.nan)

    valid = np.isfinite(L)
    m = int(valid.sum())
    if m < 100:
        return pd.DataFrame()

    log_m = math.log(m)
    root = math.sqrt(2.0 * log_m)
    c_n = root / MU1 - (math.log(math.pi) + math.log(log_m)) / (2 * MU1 * root)
    s_n = 1.0 / (MU1 * root)
    beta = -math.log(-math.log(1.0 - alpha))
    threshold = c_n + s_n * beta

    out = pd.DataFrame({"L": L}, index=ret.index)
    out["is_jump"] = valid & (np.abs(L) > threshold)
    out.attrs["threshold"] = threshold
    out.attrs["n_tested"] = m
    return out


# --------------------------------------------------------------------------
# 4D  what happens after a jump
# --------------------------------------------------------------------------

def post_jump(raw: pd.DataFrame, jumps: pd.DataFrame, horizons=FORWARD,
              hours: tuple | None = None) -> pd.DataFrame:
    """Continuation or reversal in the minutes after a detected jump.

    Signed so that positive means the jump CONTINUED. The control is every
    non-jump minute in the same hour of the day, which strips out the fact
    that jumps land in busy hours where any forward window is larger.

    `hours` restricts which jumps count, so that liquid-session jumps and
    rollover artefacts are never averaged into one meaningless number.
    """
    close = raw["close"].to_numpy()
    logp = np.log(close)
    hour = raw.index.hour.to_numpy()

    is_jump = jumps["is_jump"].to_numpy()
    if hours is not None:
        is_jump = is_jump & np.isin(hour, hours)
    sign = np.sign(jumps["L"].to_numpy())

    # Forward returns are only valid inside an unbroken run of minutes.
    step = raw.index.to_series().diff().to_numpy()
    contiguous = step == np.timedelta64(60, "s")

    rows = []
    for h in horizons:
        ahead = np.full(len(raw), np.nan)
        ok = np.zeros(len(raw), bool)
        if h < len(raw):
            span_ok = np.convolve(contiguous[1:].astype(int),
                                  np.ones(h, int), mode="valid") == h
            ok[:len(span_ok)] = span_ok
            ahead[:len(logp) - h] = logp[h:] - logp[:-h]

        signed = ahead * sign * PIP
        jm = is_jump & ok & np.isfinite(signed)
        if jm.sum() < 20:
            continue

        # Hour-matched control: same hours, same weights, no jump.
        weights = pd.Series(hour[jm]).value_counts(normalize=True)
        ctrl = (~is_jump) & ok & np.isfinite(ahead)
        ctrl_by_hour = pd.Series(np.abs(ahead[ctrl]) * PIP).groupby(
            pd.Series(hour[ctrl])).mean()
        common = weights.index.intersection(ctrl_by_hour.index)
        w = weights[common] / weights[common].sum()

        vals = signed[jm]
        rows.append({
            "minutes_after": h,
            "n_jumps": int(jm.sum()),
            "mean_pips": float(vals.mean()),
            "median_pips": float(np.median(vals)),
            "continued_share": float((vals > 0).mean()),
            "typical_move_pips": float((ctrl_by_hour[common] * w).sum()),
        })
    return pd.DataFrame(rows).set_index("minutes_after")


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
    ap.add_argument("--window", type=int, default=LM_WINDOW,
                    help=f"Lee-Mykland local window in minutes (default {LM_WINDOW}).")
    ap.add_argument("--alpha", type=float, default=LM_ALPHA,
                    help=f"Family-wise false-positive rate (default {LM_ALPHA}).")
    args = ap.parse_args()

    df = load_m1()
    raw = minute_returns(df)
    prof = tod_profile(raw)
    std = deseasonalize(raw, prof)

    show("TEST 4  JUMPS -- how much of the move arrives all at once?")
    print(f"{len(raw):,} usable 1-minute returns, New York clock")
    print(f"daily decomposition over {DAY_START:02d}:00-{DAY_CLOSE - 1:02d}:59")

    daily = daily_decomposition(raw)

    show("4A  DAILY DECOMPOSITION  (Barndorff-Nielsen & Shephard)")
    print("jump_share = (RV - BV) / RV, the fraction of the day's variance that")
    print("bipower variation refuses to count. has_jump tests it at 1%, one-sided.")
    print()
    print(f"  trading days tested            {len(daily):,}")
    print(f"  days with a significant jump   {int(daily['has_jump'].sum()):,} "
          f"({100 * daily['has_jump'].mean():.1f}%)")
    print(f"  mean jump share of variance    {daily['jump_share'].mean():.3f}")
    print(f"  median jump share              {daily['jump_share'].median():.3f}")
    print(f"  mean share on jump days        "
          f"{daily.loc[daily['has_jump'], 'jump_share'].mean():.3f}")
    print()
    print("  by year:")
    per_year = daily.groupby("year").agg(
        days=("day", "size"),
        jump_days_pct=("has_jump", lambda x: round(100 * x.mean(), 1)),
        mean_jump_share=("jump_share", "mean"),
        median_rv_pips=("rv_pips", "median"),
    ).round(3)
    print(per_year.to_string().replace("\n", "\n  "))

    show("4B  DOES A JUMP DAY LOOK DIFFERENT?")
    grp = daily.groupby("has_jump").agg(
        days=("day", "size"),
        median_rv_pips=("rv_pips", "median"),
        median_bv_pips=("bv_pips", "median"),
        mean_jump_share=("jump_share", "mean"),
    ).round(2)
    grp.index = ["no jump", "jump day"]
    print(grp.to_string())
    print()
    print("If the two rows have similar bipower variation but different realized")
    print("variance, the jump really is extra movement rather than just a busier")
    print("day being easier to detect a jump on.")

    jumps = lee_mykland(std, args.window, args.alpha)
    raw_jumps = lee_mykland(raw["ret"], args.window, args.alpha)

    show("4C  MINUTE-LEVEL JUMPS  (Lee & Mykland)")
    print(f"local window {args.window} min, family-wise alpha {args.alpha}")
    print(f"threshold {jumps.attrs['threshold']:.2f} local sigmas over "
          f"{jumps.attrs['n_tested']:,} tested minutes")
    print()
    n_std, n_raw = int(jumps["is_jump"].sum()), int(raw_jumps["is_jump"].sum())
    print(f"  jumps on DESEASONALIZED returns  {n_std:,}   "
          f"({n_std / len(daily):.2f} per trading day)")
    print(f"  jumps on RAW returns             {n_raw:,}")
    print(f"  the raw detector finds {n_raw - n_std:+,} -- that difference is the")
    print("  time-of-day curve being mistaken for news, in whichever direction")
    print("  the busy hours happen to pull it")
    print()

    hit = jumps[jumps["is_jump"]]
    by_hour = hit.groupby(hit.index.hour).size().reindex(range(24), fill_value=0)
    print("  WHEN JUMPS HAPPEN -- the news clock, read off the price alone:")
    tbl = pd.DataFrame({"jumps": by_hour})
    tbl["share_pct"] = (100 * tbl["jumps"] / tbl["jumps"].sum()).round(1)
    tbl["note"] = ["<-- rollover, see below" if h in ROLLOVER_HOURS else ""
                   for h in tbl.index]
    print(tbl.to_string().replace("\n", "\n  "))
    print()

    roll = hit[hit.index.hour.isin(ROLLOVER_HOURS)]
    print(f"  {len(roll):,} of {len(hit):,} detections ({100 * len(roll) / len(hit):.0f}%) "
          f"land in the {ROLLOVER_HOURS[0]}:00-{ROLLOVER_HOURS[-1]}:59 rollover.")
    print("  That is not news. Liquidity leaves at the 17:00 New York day boundary,")
    print("  so one tick moves price several pips while the time-of-day curve says")
    print("  minutes there should be tiny -- and the detector fires on the ratio.")
    print("  The finding is real and it is about risk, not opportunity: a position")
    print("  held through the rollover is exposed to gaps at the worst spread of")
    print("  the day. Everything below excludes these hours.")
    print()

    liquid = hit[~hit.index.hour.isin(ROLLOVER_HOURS)]
    by_min = liquid.groupby([liquid.index.hour, liquid.index.minute]).size()
    print("  the twelve busiest single minutes, ROLLOVER EXCLUDED:")
    for (h, m), count in by_min.nlargest(12).items():
        print(f"    {h:02d}:{m:02d}   {count:>4d}")
    print("  (no economic calendar was used -- these fall out of the prices)")
    print()

    size = (raw["ret"][liquid.index] * PIP).abs()
    print(f"  jump size in pips: median {size.median():.1f}  "
          f"p90 {size.quantile(0.9):.1f}  max {size.max():.1f}")
    dow = liquid.groupby(liquid.index.dayofweek).size()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("  by weekday: " + "  ".join(f"{names[d]} {c}" for d, c in dow.items()))

    show("4D  AFTER A JUMP -- follow it or fade it?")
    print("Signed so positive = the jump continued. typical_move_pips is the")
    print("average ABSOLUTE move over the same horizon in the same hours, i.e.")
    print("the yardstick any of these numbers has to be read against.")

    trading_hours = tuple(range(DAY_START, DAY_CLOSE))
    for label, hours in (("DAYTRADING WINDOW 03:00-15:59", trading_hours),
                         ("ROLLOVER 17:00-18:59 (artefact, for contrast)",
                          ROLLOVER_HOURS)):
        print()
        print(f"  {label}")
        out = post_jump(raw, jumps, hours=hours)
        print(out.round(3).to_string().replace("\n", "\n  ") if not out.empty
              else "  too few to score")

    logp = np.log(raw["close"].to_numpy())
    hour_arr = raw.index.hour.to_numpy()
    in_window = np.isin(hour_arr, trading_hours)
    hit_idx = np.flatnonzero(jumps["is_jump"].to_numpy() & in_window)
    keep = hit_idx[hit_idx + 30 < len(logp)]

    if keep.size:
        sign = np.sign(jumps["L"].to_numpy())[keep]
        signed = (logp[keep + 30] - logp[keep]) * sign * PIP
        lo, hi = day_block_bootstrap(
            session_day(raw.index[keep]),
            lambda idx: float(np.mean(signed[idx])), n_boot=2000)
        print()
        print(f"  Daytrading-window jumps only, 30-minute continuation:")
        print(f"    mean {signed.mean():+.2f} pips, 95% day-block CI "
              f"[{lo:+.2f}, {hi:+.2f}]  (n={keep.size:,})")
        verdict = ("jumps CONTINUE" if lo > 0 else
                   "jumps REVERSE" if hi < 0 else
                   "no tradable drift after a jump")
        print(f"    verdict: {verdict}")

    print()
    print("-" * 78)
    print("Reading it: 4A sizes the risk that realized volatility understates --")
    print("that share of variance arrives in minutes where a stop does not fill at")
    print("its level. 4C says exactly which minutes, and it agrees with the release")
    print("calendar without having seen one, which is the check that the detector")
    print("is finding news rather than noise. 4D says whether any of it is")
    print("tradable after the fact -- and if 4D is flat, the honest use of 4C is")
    print("to be SMALLER into those minutes, not to trade them.")


if __name__ == "__main__":
    main()
