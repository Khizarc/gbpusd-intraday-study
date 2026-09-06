"""TEST 1 -- Does GBP/USD trend or mean-revert intraday, and in which hours?

The question a daytrader actually needs answered before choosing a tactic:
over the 2-to-60 minute horizons a day trade lives on, does a move persist
(trade breakouts) or fade (trade reversion)? And does the answer change with
the session clock?

Method: the Lo-MacKinlay (1988) variance ratio.

    VR(q) = Var[q-minute return] / (q * Var[1-minute return])

Under a random walk VR(q) = 1 at every q. VR > 1 means variance grows faster
than time -- returns are positively autocorrelated, moves persist, momentum.
VR < 1 means variance grows slower than time -- negative autocorrelation,
moves fade, mean reversion.

Three things this script is careful about, because each one on its own is
enough to fake a result:

  gaps       A q-minute window is only used if all q minutes are consecutive
             in the feed. Windows spanning a weekend, a holiday or an outage
             would otherwise dump a repricing into the "q-minute move" bucket
             and push VR up at every horizon.

  dependence Overlapping windows are mechanically autocorrelated and intraday
             volatility clusters, so the textbook standard error is far too
             small. The heteroskedasticity-robust z of Lo-MacKinlay is
             reported for reference, but the confidence interval that decides
             the call is a bootstrap that resamples whole trading days.

  noise      At the 1-minute scale, bid-ask bounce and the feed's 26k frozen
             (zero-range) bars both push VR down mechanically, mimicking mean
             reversion that is not tradable. The volatility signature plot at
             the end separates that microstructure artefact from real
             reversion: if short-horizon VR is low purely because of noise,
             realized volatility will be visibly inflated at short sampling
             intervals and flatten out as the interval grows.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxlib import (PIP, horizon_returns, load_m1, log_returns,
                   split_is_oos, stars, two_sided_p, weekday_frame)

HORIZONS = (2, 3, 5, 10, 15, 30, 60)
HOUR_HORIZONS = (5, 15, 30)
SIGNATURE_INTERVALS = (1, 2, 3, 5, 10, 15, 30, 60)
N_BOOT = 500  # default; --n-boot overrides


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

def variance_ratio(r1: np.ndarray, rq: np.ndarray, q: int) -> float:
    """VR(q) from pooled 1-minute and q-minute returns.

    Both series are demeaned by the 1-minute mean scaled to the horizon, which
    is what makes the ratio a statement about variance rather than about drift.
    Intraday drift is ~0 here, so this barely moves the number -- it is done
    because leaving it out would be wrong at longer horizons.
    """
    if r1.size < 2 or rq.size < 2:
        return np.nan
    mu = r1.mean()
    var_1 = np.mean((r1 - mu) ** 2)
    var_q = np.mean((rq - q * mu) ** 2) / q
    return var_q / var_1 if var_1 > 0 else np.nan


def lo_mackinlay_z(r1: np.ndarray, segment: np.ndarray, vr: float, q: int) -> float:
    """Heteroskedasticity-robust z for VR(q) - 1, asymptotically N(0,1).

        theta = sum_{j=1}^{q-1} [2(q-j)/q]^2 * delta_j
        delta_j = E[(r_t - mu)^2 (r_{t-j} - mu)^2] / Var[r]^2
        z*      = sqrt(N) * (VR - 1) / sqrt(theta)

    delta_j is formed as a mean over the lag-j pairs that exist rather than as
    Lo-MacKinlay's raw sum, so that the gaps carved out of the series do not
    quietly shrink theta and inflate every z. Under a homoskedastic random walk
    delta_j -> 1 and theta collapses to the familiar 2(q-1)(2q-1)/(3q).
    """
    if not np.isfinite(vr) or r1.size < q + 2:
        return np.nan

    e2 = (r1 - r1.mean()) ** 2
    var_1 = e2.mean()
    if var_1 <= 0:
        return np.nan

    theta = 0.0
    for j in range(1, q):
        pairs = segment[j:] == segment[:-j]     # same contiguous run
        if not pairs.any():
            return np.nan
        delta_j = np.mean(e2[j:][pairs] * e2[:-j][pairs]) / var_1 ** 2
        theta += (2.0 * (q - j) / q) ** 2 * delta_j

    return np.sqrt(r1.size) * (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan


def bootstrap_vr(r1: pd.DataFrame, rq: pd.DataFrame, q: int, seed: int,
                 n_boot: int = N_BOOT) -> tuple:
    """Day-block percentile CI for VR(q).

    The 1-minute and q-minute samples are resampled by the SAME day draw, so a
    bootstrap replicate is a coherent set of trading days rather than two
    independently shuffled series.
    """
    days = np.union1d(r1["day"].to_numpy(), rq["day"].to_numpy())
    idx1 = _group_positions(r1["day"].to_numpy(), days)
    idxq = _group_positions(rq["day"].to_numpy(), days)

    v1 = r1["ret"].to_numpy()
    vq = rq["ret_q"].to_numpy()

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        a = np.concatenate([idx1[i] for i in pick])
        c = np.concatenate([idxq[i] for i in pick])
        draws[b] = variance_ratio(v1[a], vq[c], q)

    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return (np.nan, np.nan)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _group_positions(labels: np.ndarray, universe: np.ndarray) -> list:
    """Positional indices for each value of `universe`, in order."""
    codes = np.searchsorted(universe, labels)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(universe)))
    ends = np.append(starts[1:], len(order))
    return [order[a:b] for a, b in zip(starts, ends)]


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def vr_table(r: pd.DataFrame, horizons=HORIZONS, boot: bool = True,
             seed: int = 42, n_boot: int = N_BOOT) -> pd.DataFrame:
    base = r.dropna(subset=["ret"])
    r1 = base["ret"].to_numpy()
    seg = base["segment"].to_numpy()

    rows = []
    for q in horizons:
        rq = horizon_returns(r, q)
        vr = variance_ratio(r1, rq["ret_q"].to_numpy(), q)
        z = lo_mackinlay_z(r1, seg, vr, q)
        lo, hi = (bootstrap_vr(base, rq, q, seed, n_boot) if boot
                  else (np.nan, np.nan))
        p = two_sided_p(z) if np.isfinite(z) else np.nan
        rows.append({
            "q_min": q, "VR": vr, "z_LM": z, "p_LM": p,
            "boot_lo": lo, "boot_hi": hi,
            "n_1min": len(r1), "n_qmin": len(rq),
            "verdict": _verdict(vr, lo, hi, z),
            "sig": stars(p) if np.isfinite(p) else "",
        })
    return pd.DataFrame(rows).set_index("q_min")


def _verdict(vr: float, lo: float, hi: float, z: float) -> str:
    """The call. Made on the bootstrap CI when there is one, otherwise on the
    analytic z -- never on the bare sign of VR-1, which is always "significant"
    to four decimal places and never means anything."""
    if not np.isfinite(vr):
        return "n/a"
    if np.isfinite(lo) and np.isfinite(hi):
        if hi < 1.0:
            return "mean-reverting"
        return "trending" if lo > 1.0 else "random walk"
    if not np.isfinite(z) or abs(z) < 1.96:
        return "random walk"
    return "trending" if z > 0 else "mean-reverting"


def vr_by_hour(r: pd.DataFrame, horizons=HOUR_HORIZONS) -> pd.DataFrame:
    """VR per New York hour, windows labelled by the hour they START in.

    A window is allowed to run past the end of its hour. That is deliberate:
    the question is "given that it is now 09:15, what do the next 30 minutes
    do?", and truncating windows at the hour boundary would answer a different
    question with a sample biased toward the start of each hour.
    """
    base = r.dropna(subset=["ret"])
    out = {}
    for q in horizons:
        rq = horizon_returns(r, q)
        per_hour = {}
        for hour, chunk in rq.groupby("hour"):
            r1h = base.loc[base["hour"] == hour, "ret"].to_numpy()
            per_hour[hour] = variance_ratio(r1h, chunk["ret_q"].to_numpy(), q)
        out[f"VR_{q}m"] = pd.Series(per_hour)

    tbl = pd.DataFrame(out).reindex(range(24))
    tbl["n_1min"] = base.groupby("hour").size().reindex(range(24))
    return tbl


def volatility_signature(s: pd.DataFrame, r: pd.DataFrame,
                         intervals=SIGNATURE_INTERVALS) -> pd.DataFrame:
    """Realized volatility as a function of sampling interval.

    The classic microstructure diagnostic. If RV measured at 1 minute sits well
    above RV measured at 15 or 30, the excess is bid-ask bounce and stale
    quotes, not information -- and any VR < 1 at short q is that artefact
    rather than a fade a trader could capture.

    Reported in pips per hour so the numbers are readable as trading sizes:
    RV per sampled return is scaled by sqrt(60 / interval).
    """
    rows = []
    for step in intervals:
        rq = horizon_returns(r, step)
        v = rq["ret_q"].to_numpy()
        if v.size < 2:
            continue
        rv_per_step = np.sqrt(np.mean(v ** 2))
        rows.append({
            "interval_min": step,
            "rv_pips_per_hour": rv_per_step * np.sqrt(60.0 / step) * PIP,
            "n_returns": v.size,
        })
    out = pd.DataFrame(rows).set_index("interval_min")
    anchor = out["rv_pips_per_hour"].iloc[-1]
    out["vs_slowest"] = (out["rv_pips_per_hour"] / anchor).round(3)
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def show(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt(tbl: pd.DataFrame) -> str:
    return tbl.round(4).to_string()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="Skip the day-block bootstrap CIs (much faster).")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help=f"Bootstrap replicates for the CIs (default {N_BOOT}).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_m1()
    s = weekday_frame(df)
    r = log_returns(s)

    usable = r["ret"].notna()
    show("TEST 1  VARIANCE RATIO -- trend vs mean reversion at daytrading horizons")
    print(f"M1 bars {len(s):,} on the New York clock, Mon-Fri, Sunday reopen dropped")
    print(f"1-minute returns usable {int(usable.sum()):,} "
          f"({100 * usable.mean():.1f}% -- the rest span a gap and are excluded)")
    print(f"contiguous segments {r['segment'].nunique():,}  "
          f"(a segment is an unbroken run of minutes; windows never cross one)")
    zero = float((r.loc[usable, "ret"] == 0).mean())
    print(f"exactly-zero 1-minute returns {100 * zero:.1f}%  "
          "-- frozen bars; they drag VR down at short q on their own")

    show("1A  POOLED OVER ALL HOURS")
    print("VR < 1 fade, VR > 1 follow. The verdict column reads the bootstrap CI,")
    print("not the analytic z -- with overlapping windows the z is optimistic.")
    print()
    pooled = vr_table(r, boot=not args.no_bootstrap, seed=args.seed,
                      n_boot=args.n_boot)
    print(fmt(pooled))

    show("1B  IN-SAMPLE / OUT-OF-SAMPLE  (train 2020-2023, holdout 2024-2025)")
    print("A horizon whose sign flips between the halves is noise, whatever its z.")
    print()
    years = pd.DatetimeIndex(r.index).year.to_numpy()
    is_mask, oos_mask = split_is_oos(years)
    for label, mask in (("2020-2023", is_mask), ("2024-2025", oos_mask)):
        sub = vr_table(r[mask], boot=False)
        print(f"  {label}")
        print(fmt(sub[["VR", "z_LM", "n_qmin", "verdict"]]).replace("\n", "\n  "))
        print()

    show("1C  BY NEW YORK HOUR  (window labelled by the hour it starts in)")
    print("This is the table that changes what you do: hours where VR sits below 1")
    print("are hours to fade extensions, hours above 1 are hours to follow breaks.")
    print("24 hours x 3 horizons is 72 comparisons -- expect a few extremes by luck,")
    print("so trust the block of adjacent hours that agree, not a lone cell.")
    print()
    by_hour = vr_by_hour(r)
    print(by_hour.round(3).to_string())

    cols = [c for c in by_hour.columns if c.startswith("VR_")]
    mean_vr = by_hour[cols].mean(axis=1)
    print()
    print("most mean-reverting hours (mean VR across horizons):")
    print("  " + ", ".join(f"{h:02d}:00 {v:.3f}" for h, v in mean_vr.nsmallest(4).items()))
    print("most trending hours:")
    print("  " + ", ".join(f"{h:02d}:00 {v:.3f}" for h, v in mean_vr.nlargest(4).items()))

    show("1D  VOLATILITY SIGNATURE -- how much of the short-horizon result is noise")
    print("Realized vol at each sampling interval, in pips per hour. A falling")
    print("column means 1-minute prices carry bounce that is not tradable range.")
    print()
    print(volatility_signature(s, r).round(3).to_string())

    print()
    print("-" * 78)
    print("Reading it: compare 1A's short-q VR against 1D. If VR(2) is far below 1")
    print("AND 1D shows RV inflated at 1-2 minutes, the short-horizon 'reversion' is")
    print("the spread breathing, and a fade there pays the spread to capture it. The")
    print("horizons worth trading are the ones where 1D has already flattened out.")


if __name__ == "__main__":
    main()
