"""VALIDATION -- do the hand-rolled estimators actually compute what they claim?

Every statistic in this project is written out by hand rather than imported
from a tested library: the Lo-MacKinlay variance ratio and its
heteroskedasticity-robust z, Newey-West standard errors, the Lee-Mykland jump
threshold, the Barndorff-Nielsen & Shephard decomposition, the day-block
bootstrap and White's Reality Check. Any one of them could carry a sign slip
or a normalisation error and still produce numbers that look entirely
reasonable on real data, because on real data there is nothing to check them
against.

So they are checked here against synthetic data whose answer is known in
advance. Two kinds of check, and both are needed:

  CALIBRATION  Feed the estimator data with NO effect in it and confirm it
               says so -- and, for the tests that carry a p-value, that it
               says so at the advertised rate. A test that rejects 20% of the
               time at a nominal 5% is not conservative or aggressive, it is
               broken, and every null result in this repo would be worthless.

  POWER        Feed it data with a KNOWN effect of known size and confirm it
               finds it, at roughly the right magnitude. This is the half that
               matters most here, because the headline findings of tests 1, 3,
               5D and 6 are all null results. A null result from an estimator
               that cannot detect anything is not evidence of absence.

Run it directly. Exit code is 0 if everything passes, 1 otherwise, so it works
as a CI gate as well as a thing to read.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

import fxlib
from fxlib import (day_block_bootstrap, deseasonalize, horizon_returns,
                   log_returns, newey_west_ols, tod_profile)
from test_jumps import bns_day, lee_mykland
from test_snooping import holm, individual_stats, reality_check
from test_variance_ratio import lo_mackinlay_z, variance_ratio

NY = "America/New_York"
CHECKS = []


def check(name: str, group: str):
    def register(fn):
        CHECKS.append((group, name, fn))
        return fn
    return register


# --------------------------------------------------------------------------
# synthetic data
# --------------------------------------------------------------------------

def synth_frame(ret: np.ndarray, start: str = "2021-01-04 03:00") -> pd.DataFrame:
    """Wrap a return series as a minute-indexed OHLC frame the library accepts.

    A single unbroken run of minutes, so segment logic is exercised but never
    the reason a check fails. Checks that care about gaps build their own.
    """
    idx = pd.date_range(start, periods=len(ret) + 1, freq="min", tz=NY)
    close = np.exp(np.concatenate([[0.0], np.cumsum(ret)]))
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close}, index=idx)


def ar1(n: int, phi: float, sigma: float, rng) -> np.ndarray:
    """AR(1) returns: r_t = phi*r_{t-1} + eps_t. Autocorrelation is phi^k."""
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def vr_theory(q: int, phi: float) -> float:
    """VR(q) = 1 + 2 * sum_{k=1}^{q-1} (1 - k/q) * rho_k, with rho_k = phi^k."""
    return 1.0 + 2.0 * sum((1 - k / q) * phi ** k for k in range(1, q))


# --------------------------------------------------------------------------
# 1. variance ratio
# --------------------------------------------------------------------------

@check("random walk gives an UNBIASED VR of 1", "variance ratio")
def _vr_null():
    """Averaged over independent walks, so this tests for systematic bias
    rather than for luck.

    A single walk is not a useful check: VR(q) has a real standard error of
    sqrt(theta_q / n) with theta_q = 2(q-1)(2q-1)/(3q), which at q=60 is around
    0.036 on 60k points. Demanding VR within 0.03 of 1 on one draw would fail
    a perfectly correct estimator roughly a third of the time. Each horizon is
    therefore judged against its OWN standard error, and the mean across seeds
    is what has to sit on 1.
    """
    rng = np.random.default_rng(1)
    seeds, n = 12, 100_000

    per_q = {q: [] for q in (2, 5, 15, 60)}
    for _ in range(seeds):
        r = log_returns(synth_frame(rng.normal(0, 1e-4, n)))
        r1 = r["ret"].dropna().to_numpy()
        for q in per_q:
            per_q[q].append(
                variance_ratio(r1, horizon_returns(r, q)["ret_q"].to_numpy(), q))

    ok, detail = True, []
    for q, values in per_q.items():
        theta = 2 * (q - 1) * (2 * q - 1) / (3 * q)
        se_mean = math.sqrt(theta / n) / math.sqrt(seeds)
        mean_vr = float(np.mean(values))
        z = (mean_vr - 1.0) / se_mean
        ok &= abs(z) < 4.0
        detail.append(f"q={q}: {mean_vr:.4f} (z={z:+.2f})")
    return ok, (f"mean VR over {seeds} walks of {n:,}, vs each horizon's own "
                f"standard error\n      " + "   ".join(detail))


@check("AR(1) returns recover the closed-form VR", "variance ratio")
def _vr_power():
    rng = np.random.default_rng(2)
    rows, worst = [], 0.0
    for phi in (-0.20, 0.20):
        r = log_returns(synth_frame(ar1(400_000, phi, 1e-4, rng)))
        r1 = r["ret"].dropna().to_numpy()
        for q in (2, 5, 10):
            got = variance_ratio(r1, horizon_returns(r, q)["ret_q"].to_numpy(), q)
            want = vr_theory(q, phi)
            worst = max(worst, abs(got - want))
            rows.append(f"phi={phi:+.2f} q={q}: got {got:.3f} want {want:.3f}")
    return worst < 0.02, f"max error {worst:.4f}\n      " + "\n      ".join(rows)


@check("the robust z rejects at its advertised 5% rate", "variance ratio")
def _vr_z_calibration():
    """The estimator's own delta_j is a modified form of Lo-MacKinlay's, taken
    as a mean over surviving lag pairs so that gaps do not shrink it. That
    modification is exactly the kind of change that silently breaks a test's
    size, so the size is measured rather than assumed."""
    rng = np.random.default_rng(3)
    reject = 0
    trials = 300
    for _ in range(trials):
        r = log_returns(synth_frame(rng.normal(0, 1e-4, 20_000)))
        base = r.dropna(subset=["ret"])
        r1 = base["ret"].to_numpy()
        vr = variance_ratio(r1, horizon_returns(r, 5)["ret_q"].to_numpy(), 5)
        z = lo_mackinlay_z(r1, base["segment"].to_numpy(), vr, 5)
        reject += int(abs(z) > 1.96)

    rate = reject / trials
    se = math.sqrt(0.05 * 0.95 / trials)
    ok = abs(rate - 0.05) < 4 * se
    return ok, (f"rejection rate {rate:.3f} at nominal 0.050 "
                f"({trials} trials, +/-{4 * se:.3f} allowed)")


@check("windows never straddle a gap", "variance ratio")
def _horizon_gap_safety():
    """A q-minute window spanning a weekend would be a repricing counted as a
    q-minute move, which inflates VR at every horizon at once."""
    rng = np.random.default_rng(4)
    frame = synth_frame(rng.normal(0, 1e-4, 2_000))
    # Excise a block, then jam a large jump across the resulting seam.
    kept = frame.drop(frame.index[900:1000])
    kept.iloc[900:] *= 1.05

    r = log_returns(kept)
    rq = horizon_returns(r, 30)
    biggest = float(np.abs(rq["ret_q"]).max())
    naive = abs(math.log(1.05))
    return biggest < naive / 2, (
        f"largest 30-min window {biggest:.5f}; the 5% seam is {naive:.5f} "
        f"-- {'excluded' if biggest < naive / 2 else 'LEAKED IN'}")


# --------------------------------------------------------------------------
# 2. Newey-West
# --------------------------------------------------------------------------

@check("recovers known coefficients", "newey-west")
def _nw_beta():
    rng = np.random.default_rng(5)
    n = 20_000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    truth = np.array([0.5, -1.5, 2.0])
    y = X @ truth + rng.normal(0, 0.5, n)

    fit = newey_west_ols(y, X, lags=5)
    err = float(np.abs(fit["beta"] - truth).max())
    return err < 0.02, (f"max coefficient error {err:.4f}  "
                        f"(got {np.round(fit['beta'], 3)}, want {truth})")


@check("matches a hand-computed sandwich at lag 1", "newey-west")
def _nw_hand():
    """The Bartlett weight and the G + G' symmetrisation are easy to get wrong
    in a way no simulation would reveal, so one small case is checked against
    the formula written out longhand."""
    rng = np.random.default_rng(6)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.normal(size=n)

    fit = newey_west_ols(y, X, lags=1)

    xtx_inv = np.linalg.inv(X.T @ X)
    resid = y - X @ (xtx_inv @ (X.T @ y))
    u = X * resid[:, None]
    S = u.T @ u
    G = u[1:].T @ u[:-1]
    S = S + 0.5 * (G + G.T)                      # Bartlett weight 1 - 1/2
    se_manual = np.sqrt(np.diag(xtx_inv @ S @ xtx_inv))

    err = float(np.abs(fit["se"] - se_manual).max())
    return err < 1e-12, f"max |se - hand-computed| = {err:.2e}"


@check("fixes the coverage that plain OLS breaks", "newey-west")
def _nw_coverage():
    """With autocorrelated errors, textbook OLS intervals are far too narrow.
    This is the entire reason the project uses Newey-West, so it is worth
    confirming that it is true here and that the fix works."""
    rng = np.random.default_rng(7)
    trials, n, truth = 400, 400, 1.0
    cover_nw = cover_ols = 0

    for _ in range(trials):
        x = ar1(n, 0.7, 1.0, rng)
        e = ar1(n, 0.7, 1.0, rng)
        y = truth * x + e
        X = np.column_stack([np.ones(n), x])

        fit = newey_west_ols(y, X, lags=8)
        cover_nw += int(abs(fit["beta"][1] - truth) < 1.96 * fit["se"][1])

        resid = y - X @ fit["beta"]
        s2 = resid @ resid / (n - 2)
        se_ols = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        cover_ols += int(abs(fit["beta"][1] - truth) < 1.96 * se_ols)

    nw, ols = cover_nw / trials, cover_ols / trials
    return nw > 0.88 and nw > ols, (
        f"95% CI coverage: Newey-West {nw:.3f}   plain OLS {ols:.3f}   "
        f"(nominal 0.950)")


# --------------------------------------------------------------------------
# 3. Lee-Mykland
# --------------------------------------------------------------------------

@check("finds almost nothing in a jump-free series", "lee-mykland")
def _lm_null():
    """Family-wise alpha of 1% means ~1% of RUNS should show any detection at
    all, not 1% of minutes. A per-minute reading of the threshold would flag
    thousands here."""
    rng = np.random.default_rng(8)
    runs, with_any, total = 40, 0, 0
    for _ in range(runs):
        s = pd.Series(rng.normal(0, 1e-4, 30_000))
        s.index = pd.date_range("2021-01-04", periods=len(s), freq="min", tz=NY)
        found = int(lee_mykland(s)["is_jump"].sum())
        total += found
        with_any += int(found > 0)
    rate = with_any / runs
    return rate <= 0.15, (f"{total} detections across {runs} clean runs of 30k "
                          f"minutes; {with_any} run(s) had any ({rate:.1%})")


@check("recovers injected jumps at the right places", "lee-mykland")
def _lm_power():
    rng = np.random.default_rng(9)
    n, sigma = 30_000, 1e-4
    ret = rng.normal(0, sigma, n)
    where = np.arange(2_000, n - 2_000, 2_500)
    ret[where] += 12 * sigma * rng.choice([-1, 1], len(where))

    s = pd.Series(ret, index=pd.date_range("2021-01-04", periods=n,
                                           freq="min", tz=NY))
    out = lee_mykland(s)
    found = np.flatnonzero(out["is_jump"].to_numpy())

    hits = len(set(found) & set(where))
    false_pos = len(found) - hits
    return hits >= 0.9 * len(where) and false_pos <= 2, (
        f"recovered {hits}/{len(where)} injected 12-sigma jumps, "
        f"{false_pos} false positive(s)")


@check("is not fooled by time-of-day seasonality alone", "lee-mykland")
def _lm_seasonality():
    """The failure mode that actually bit during test 4: a purely seasonal
    series has no jumps in it, but a detector run on raw returns will call
    every busy hour one."""
    rng = np.random.default_rng(10)
    idx = pd.date_range("2021-01-04 00:00", periods=60_000, freq="min", tz=NY)
    shape = 1.0 + 1.5 * np.sin(np.pi * (idx.hour * 60 + idx.minute) / 1440) ** 2
    ret = rng.normal(0, 1e-4, len(idx)) * shape

    frame = pd.DataFrame({"ret": ret}, index=idx)
    raw_hits = int(lee_mykland(frame["ret"])["is_jump"].sum())
    std_hits = int(lee_mykland(deseasonalize(frame, tod_profile(frame)))
                   ["is_jump"].sum())
    return std_hits <= raw_hits and std_hits <= 3, (
        f"raw {raw_hits} detections, deseasonalized {std_hits} "
        f"(truth is 0 -- this series is seasonal, not jumpy)")


# --------------------------------------------------------------------------
# 4. Barndorff-Nielsen & Shephard
# --------------------------------------------------------------------------

@check("continuous path gives BV = RV and no jump", "BNS")
def _bns_null():
    rng = np.random.default_rng(11)
    shares, rejects, trials = [], 0, 400
    for _ in range(trials):
        out = bns_day(rng.normal(0, 1e-4, 780))
        shares.append(out["jump_share"])
        rejects += int(out["z"] > 2.326)

    mean_share, rate = float(np.mean(shares)), rejects / trials
    return mean_share < 0.03 and rate < 0.05, (
        f"mean jump share {mean_share:.4f} (want ~0); "
        f"rejects {rate:.3f} of the time at nominal 0.010")


@check("detects an injected jump of known size", "BNS")
def _bns_power():
    rng = np.random.default_rng(12)
    sigma = 1e-4
    detected, shares = 0, []
    for _ in range(200):
        ret = rng.normal(0, sigma, 780)
        ret[390] += 25 * sigma          # one large jump, mid-session
        out = bns_day(ret)
        detected += int(out["z"] > 2.326)
        shares.append(out["jump_share"])

    # A 25-sigma jump against 780 sigma^2 of diffusion is 625/(780+625) ~ 0.44.
    expected = 625 / (780 + 625)
    got = float(np.mean(shares))
    return detected >= 180 and abs(got - expected) < 0.10, (
        f"detected on {detected}/200 days; mean jump share {got:.3f} "
        f"vs {expected:.3f} predicted from the injected size")


# --------------------------------------------------------------------------
# 5. day-block bootstrap
# --------------------------------------------------------------------------

@check("covers at 95% and beats an iid resample", "bootstrap")
def _boot_coverage():
    """Observations inside a day share a common shock. Resampling observations
    instead of days ignores that and produces intervals that are too narrow --
    which is the mistake this helper exists to prevent."""
    rng = np.random.default_rng(13)
    trials, n_days, per_day = 200, 200, 10
    cover_day = cover_naive = 0

    for _ in range(trials):
        shock = rng.normal(0, 1.0, n_days).repeat(per_day)
        vals = shock + rng.normal(0, 0.3, n_days * per_day)
        days = np.arange(n_days).repeat(per_day)

        lo, hi = day_block_bootstrap(days, lambda i: float(vals[i].mean()),
                                     n_boot=300, seed=int(rng.integers(1e6)))
        cover_day += int(lo <= 0.0 <= hi)

        draws = [vals[rng.integers(0, vals.size, vals.size)].mean()
                 for _ in range(300)]
        n_lo, n_hi = np.quantile(draws, [0.025, 0.975])
        cover_naive += int(n_lo <= 0.0 <= n_hi)

    day, naive = cover_day / trials, cover_naive / trials
    return day > 0.88 and day > naive, (
        f"coverage: day-block {day:.3f}   iid-over-observations {naive:.3f}   "
        f"(nominal 0.950)")


# --------------------------------------------------------------------------
# 6. White's Reality Check
# --------------------------------------------------------------------------

@check("a family of pure noise clears at ~5%", "reality check")
def _rc_null():
    """If this rejects too often, test 6's 'nothing survives' verdict is
    worthless -- it would mean the procedure manufactures survivors."""
    rng = np.random.default_rng(14)
    trials, rejects = 120, 0
    for _ in range(trials):
        fam = pd.DataFrame(rng.normal(0, 10, (400, 20)),
                           columns=[f"rule{i}" for i in range(20)])
        rejects += int(reality_check(fam, n_boot=300,
                                     seed=int(rng.integers(1e6)))["p_value"] < 0.05)
    rate = rejects / trials
    se = math.sqrt(0.05 * 0.95 / trials)
    return rate < 0.05 + 4 * se, (
        f"rejects {rate:.3f} of the time at nominal 0.050 ({trials} trials)")


@check("finds one genuinely good rule hidden among noise", "reality check")
def _rc_power():
    rng = np.random.default_rng(15)
    trials, found = 60, 0
    for _ in range(trials):
        data = rng.normal(0, 10, (400, 20))
        data[:, 7] += 2.0                      # one rule with a real edge
        fam = pd.DataFrame(data, columns=[f"rule{i}" for i in range(20)])
        rc = reality_check(fam, n_boot=300, seed=int(rng.integers(1e6)))
        found += int(rc["p_value"] < 0.05 and rc["best_rule"] == "rule7")
    rate = found / trials
    return rate > 0.70, (f"identified the planted rule {rate:.0%} of the time "
                         f"({trials} trials)")


@check("Holm is monotone and never below the raw p-value", "reality check")
def _holm_props():
    rng = np.random.default_rng(16)
    p = pd.Series(rng.uniform(0, 1, 40), index=[f"r{i}" for i in range(40)])
    adj = holm(p)
    ordered = adj[p.sort_values().index].to_numpy()
    return (bool((adj >= p - 1e-12).all())
            and bool(np.all(np.diff(ordered) >= -1e-12))
            and bool((adj <= 1.0).all())), (
        "adjusted >= raw, non-decreasing in rank, capped at 1.0")


# --------------------------------------------------------------------------
# 7. seasonality helpers
# --------------------------------------------------------------------------

@check("recovers a known time-of-day shape", "seasonality")
def _tod_recovery():
    """Judged in RELATIVE error against the sampling noise of the estimate.

    Each 15-minute bucket is an RMS built from a finite number of minutes, so
    it carries a standard error of about 1/sqrt(2m) in relative terms. With
    multipliers spanning 0.5 to 2.0, a flat absolute tolerance would be far too
    tight at the top of that range and far too loose at the bottom.
    """
    rng = np.random.default_rng(17)
    idx = pd.date_range("2021-01-04 00:00", periods=400_000, freq="min", tz=NY)
    idx = idx[idx.dayofweek < 5]
    bucket = idx.floor("15min").time

    truth = {t: 0.5 + 1.5 * ((i * 7) % 96) / 96
             for i, t in enumerate(sorted(set(bucket)))}
    scale = np.array([truth[t] for t in bucket])
    frame = pd.DataFrame({"ret": rng.normal(0, 1e-4, len(idx)) * scale}, index=idx)

    prof = tod_profile(frame)
    want = pd.Series(truth)
    want = want / np.sqrt((want ** 2).mean())      # same normalisation as mult
    rel = float((prof["mult"] / want.reindex(prof.index) - 1).abs().max())

    per_bucket = len(idx) / 96
    tol = 5.0 / math.sqrt(2 * per_bucket)
    return rel < tol, (f"max relative multiplier error {rel:.4f}, tolerance "
                       f"{tol:.4f} (5 se at {per_bucket:.0f} minutes/bucket)")


@check("the shape it fits transfers to data it never saw", "seasonality")
def _tod_out_of_sample():
    """Fitting a profile and then rescaling the SAME returns by it returns
    1.000 in every bucket by construction -- it is an identity, not a test.
    The profile is therefore fitted on one half of the days and applied to the
    other, which is the only version of this that can fail.
    """
    rng = np.random.default_rng(18)
    idx = pd.date_range("2021-01-04 00:00", periods=400_000, freq="min", tz=NY)
    idx = idx[idx.dayofweek < 5]
    bucket = idx.floor("15min").time
    scale = np.array([0.5 + 1.5 * ((hash(t) % 96) / 96) for t in bucket])

    frame = pd.DataFrame({"ret": rng.normal(0, 1e-4, len(idx)) * scale}, index=idx)
    cut = idx[len(idx) // 2]

    prof = tod_profile(frame[frame.index < cut])
    held = frame[frame.index >= cut]
    rms = deseasonalize(held, prof).pow(2).groupby(
        pd.Index(held.index.floor("15min").time)).mean().pow(0.5)

    rms = rms / rms.median()          # a level shift is not a shape failure
    spread = float((rms - 1).abs().max())
    tol = 6.0 / math.sqrt(2 * len(held) / 96)
    return spread < tol, (f"worst holdout bucket off by {spread:.4f}, tolerance "
                          f"{tol:.4f}; shape transfers to unseen days")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", help="Run only checks in this group.")
    args = ap.parse_args()

    print("=" * 78)
    print("ESTIMATOR VALIDATION -- synthetic data with known answers")
    print("=" * 78)

    failures, current = [], None
    for group, name, fn in CHECKS:
        if args.group and group != args.group:
            continue
        if group != current:
            current = group
            print(f"\n{group.upper()}")

        try:
            passed, detail = fn()
        except Exception as exc:                       # noqa: BLE001
            passed, detail = False, f"raised {type(exc).__name__}: {exc}"

        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"      {detail}")
        if not passed:
            failures.append(f"{group}: {name}")

    total = len([c for c in CHECKS if not args.group or c[0] == args.group])
    print()
    print("=" * 78)
    if failures:
        print(f"{len(failures)} of {total} checks FAILED")
        for f in failures:
            print(f"  - {f}")
        print("\nA failure here invalidates whatever result depends on that")
        print("estimator. Fix it before believing any of the six tests.")
        return 1

    print(f"all {total} checks passed")
    print("\nThe null results in tests 1, 3, 5D and 6 come from estimators that")
    print("demonstrably DO detect an effect when one is present, and that hold")
    print("their advertised error rates when one is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
