"""TEST 6 -- Of every directional pattern we have looked at, does ANY survive?

Tests 1 through 5 each looked at a lot of cells: 24 hours by 3 horizons in the
variance ratio, 6 session-by-range combinations in the breakout study, hour and
weekday tables everywhere. Some of those cells looked interesting. This test
exists because "looked interesting" is exactly what a table of that size
produces from pure noise, and nothing so far has paid for that.

The setup: a fixed family of simple directional rules, every one of them
holding inside a single day and flat by 16:00. Each produces one number per
trading day, in gross pips -- cost is held back and applied afterwards as a
hurdle, because subtracting a constant and then t-testing it just proves that
one pip exceeds zero. Then three ways of asking whether the best of them is
real, in increasing order of how hard they are to fool:

  1  INDIVIDUAL      each rule's own t-statistic, as it would be reported if it
                     were the only thing anyone had tried. This is the number
                     that gets a strategy funded and is almost always wrong.

  2  HOLM-BONFERRONI controls the chance of even one false positive across the
                     whole family. Correct but conservative here, because it
                     assumes the rules are unrelated and most of these rules
                     are close to the same trade.

  3  REALITY CHECK   White (2000). Resample DAYS jointly across every rule at
                     once, which preserves the fact that they are all long the
                     same pound at the same time, and build the distribution of
                     the BEST t-statistic in the family under the null that all
                     of them are worthless. The observed best is then read
                     against that distribution. This is the honest one.

Three rules in the family are pure random noise by construction, seeded and
labelled. They are there so the reader can watch a coin flip earn a
respectable-looking t-statistic, and see where it lands in step 3. If a random
rule ever clears the reality check, the reality check is broken.

Nothing here is fitted. The family was written down before it was run, and it
is not re-selected afterwards -- which is the whole point, because re-selecting
it afterwards is the thing being tested for.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxlib import PIP, load_m1, minute_returns, session_day

DAY_START, DAY_CLOSE = 3, 16
SPLIT_HOUR = 9
ASIA_START, ASIA_END = 19, 3
MIN_BARS = 400
COST_PIPS = 1.0
N_BOOT = 5000
N_RANDOM = 3
OOS_FROM = 2024


# --------------------------------------------------------------------------
# the daily panel every rule is built from
# --------------------------------------------------------------------------

def build_panel(r: pd.DataFrame) -> pd.DataFrame:
    """Hourly marks plus session and Asian aggregates, one row per trading day."""
    hour = r.index.hour
    w = r[(hour >= DAY_START) & (hour < DAY_CLOSE)].copy()
    w["day"] = session_day(w.index)
    w["hour"] = w.index.hour
    w = w[pd.DatetimeIndex(w["day"]).dayofweek <= 4]
    w = w[w.groupby("day")["ret"].transform("size") >= MIN_BARS]

    g = w.groupby(["day", "hour"])
    opens = g["open"].first().unstack()
    closes = g["close"].last().unstack()

    panel = pd.DataFrame(index=opens.index)
    panel["session_open"] = opens[DAY_START]
    panel["session_close"] = closes[DAY_CLOSE - 1]
    panel["morning_close"] = closes[SPLIT_HOUR - 1]
    panel["afternoon_open"] = opens[SPLIT_HOUR]

    for h in range(DAY_START, DAY_CLOSE):
        panel[f"drift_{h}"] = (closes[h] - opens[h]) * PIP

    panel["session_net"] = (panel["session_close"] - panel["session_open"]) * PIP
    panel["morning_net"] = (panel["morning_close"] - panel["session_open"]) * PIP
    panel["afternoon_net"] = (panel["session_close"] - panel["afternoon_open"]) * PIP

    # Asian session, which closes before the trading window opens.
    a = r[(hour >= ASIA_START) | (hour < ASIA_END)]
    aday = session_day(a.index, roll_evening=True)
    ag = a.groupby(aday)
    asia = pd.DataFrame({
        "asia_net": (ag["close"].last() - ag["open"].first()) * PIP,
        "asia_range": (ag["high"].max() - ag["low"].min()) * PIP,
        "asia_bars": ag.size(),
    })
    panel = panel.join(asia[asia["asia_bars"] >= 300], how="inner")

    idx = pd.DatetimeIndex(panel.index)
    panel["dow"] = idx.dayofweek
    panel["year"] = idx.year
    panel["prev_session_net"] = panel["session_net"].shift(1)

    # Volatility regime, known before the window opens.
    panel["vol_regime"] = pd.qcut(panel["asia_range"], 3,
                                  labels=["quiet", "normal", "busy"])
    return panel.dropna(subset=["session_net", "asia_net"])


# --------------------------------------------------------------------------
# the family
# --------------------------------------------------------------------------

def build_family(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Every rule as a column of GROSS daily pips. NaN on days it sits out.

    Cost is deliberately NOT charged here. A fixed per-trade cost is a
    constant, so subtracting it and then running a t-test makes every rule in
    the family "significantly negative" -- the test would be confirming that
    one pip is greater than zero, which is not in doubt. The two questions are
    separated instead: these tests ask whether there is any directional
    information at all, and the cost is applied afterwards as a hurdle that
    whatever survives has to clear.
    """
    out = {}

    def add(name, pnl, active=None):
        v = pd.Series(pnl, index=panel.index, dtype=float)
        if active is not None:
            v = v.where(active)
        out[name] = v

    # --- hour-of-day drift: long that hour, every day -----------------------
    for h in range(DAY_START, DAY_CLOSE):
        add(f"long {h:02d}:00-{h + 1:02d}:00", panel[f"drift_{h}"])

    # --- weekday: long the whole session on one weekday ---------------------
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        add(f"long session on {name}", panel["session_net"], panel["dow"] == d)

    # --- carry the prior session's direction into this one ------------------
    add("follow Asian direction",
        np.sign(panel["asia_net"]) * panel["session_net"])
    add("follow yesterday's direction",
        np.sign(panel["prev_session_net"]) * panel["session_net"])

    # --- carry the morning into the afternoon -------------------------------
    add("follow the morning (09:00 on)",
        np.sign(panel["morning_net"]) * panel["afternoon_net"])

    # --- the same three, but only in one volatility regime ------------------
    for regime in ("quiet", "busy"):
        active = panel["vol_regime"] == regime
        add(f"follow Asian direction [{regime}]",
            np.sign(panel["asia_net"]) * panel["session_net"], active)
        add(f"follow the morning [{regime}]",
            np.sign(panel["morning_net"]) * panel["afternoon_net"], active)

    # --- pure noise, included on purpose ------------------------------------
    rng = np.random.default_rng(seed)
    for i in range(N_RANDOM):
        coin = rng.choice([-1.0, 1.0], size=len(panel))
        add(f"RANDOM coin flip #{i + 1}", coin * panel["session_net"].to_numpy())

    return pd.DataFrame(out, index=panel.index)


# --------------------------------------------------------------------------
# the three tests
# --------------------------------------------------------------------------

def individual_stats(fam: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in fam.columns:
        v = fam[name].dropna().to_numpy()
        n = v.size
        sd = v.std(ddof=1)
        t = v.mean() / (sd / np.sqrt(n)) if sd > 0 and n > 1 else np.nan
        rows.append({
            "rule": name, "n_days": n,
            "gross_pips_per_day": v.mean(),
            "t": t, "p_individual": _p(t),
        })
    return pd.DataFrame(rows).set_index("rule")


def _p(t: float) -> float:
    from fxlib import two_sided_p
    return two_sided_p(t) if np.isfinite(t) else np.nan


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values. Controls the family-wise error rate
    with no assumption about how the rules relate to each other -- which makes
    it valid but pessimistic for a family that is mostly the same trade."""
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, (m - i) * p[name])
        adj[name] = min(running, 1.0)
    return pd.Series(adj).reindex(p.index)


def reality_check(fam: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """White's Reality Check on the best t-statistic in the family.

    Days are resampled jointly across every rule, so the bootstrap keeps the
    correlation between rules that are secretly the same trade. Each rule's
    series is recentred on its own mean, imposing the null that nothing works,
    and the statistic is studentized so rules with different variances compete
    on equal terms.
    """
    values = fam.to_numpy()
    mask = np.isfinite(values)
    observed_mean = np.nanmean(values, axis=0)

    n_days, n_rules = values.shape
    rng = np.random.default_rng(seed)

    def t_stats(rows: np.ndarray) -> np.ndarray:
        v = values[rows]
        m = mask[rows]
        counts = m.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            filled = np.where(m, v, 0.0)
            means = filled.sum(axis=0) / counts
            var = (np.where(m, (v - means) ** 2, 0.0).sum(axis=0)
                   / np.maximum(counts - 1, 1))
            return (means - observed_mean) / np.sqrt(var / counts), counts

    obs_t = individual_stats(fam)["t"].to_numpy()
    best_obs = np.nanmax(np.abs(obs_t))

    draws = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n_days, n_days)
        t, counts = t_stats(rows)
        t = np.where(counts >= 20, t, np.nan)
        draws[b] = np.nanmax(np.abs(t))

    draws = draws[np.isfinite(draws)]
    return {
        "best_observed_t": float(best_obs),
        "best_rule": fam.columns[int(np.nanargmax(np.abs(obs_t)))],
        "null_p50": float(np.quantile(draws, 0.50)),
        "null_p95": float(np.quantile(draws, 0.95)),
        "null_p99": float(np.quantile(draws, 0.99)),
        "p_value": float((draws >= best_obs).mean()),
        "n_boot": draws.size,
    }


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
    ap.add_argument("--cost-pips", type=float, default=COST_PIPS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_m1()
    r = minute_returns(df)
    panel = build_panel(r)
    fam = build_family(panel, args.seed)

    show("TEST 6  DATA-SNOOPING AUDIT -- does the best pattern beat the best "
         "accident?")
    print(f"{len(panel):,} trading days, {len(fam.columns)} rules "
          f"({N_RANDOM} of them deliberately random)")
    print(f"every rule holds inside one day and is flat by {DAY_CLOSE:02d}:00")
    print(f"tests run on GROSS pips; {args.cost_pips:.2f} pips/day is then applied")
    print("as a hurdle, so 'is there a signal' and 'is it tradable' stay separate")

    stats = individual_stats(fam)
    stats["p_holm"] = holm(stats["p_individual"])
    stats = stats.sort_values("t", key=lambda s: s.abs(), ascending=False)

    show("6A  EVERY RULE, RANKED BY ITS OWN t-STATISTIC")
    print("This is the table that would be shown to you by someone selling one of")
    print("these. Read the RANDOM rows first -- they had no chance of working.")
    print()
    disp = stats.copy()
    disp["net_pips_per_day"] = (disp["gross_pips_per_day"] - args.cost_pips).round(3)
    disp["clears_cost"] = disp["gross_pips_per_day"].abs() > args.cost_pips
    disp["gross_pips_per_day"] = disp["gross_pips_per_day"].round(3)
    disp["t"] = disp["t"].round(2)
    disp["p_individual"] = disp["p_individual"].round(4)
    disp["p_holm"] = disp["p_holm"].round(3)
    print(disp[["n_days", "gross_pips_per_day", "t", "p_individual", "p_holm",
                "net_pips_per_day", "clears_cost"]].to_string())
    print()
    print("net_pips_per_day is the LONG version net of cost; a rule with a")
    print("negative gross edge would be traded short, so clears_cost asks whether")
    print("the edge in EITHER direction is bigger than the pip it has to pay.")

    naive = int((stats["p_individual"] < 0.05).sum())
    survivors = stats.index[stats["p_holm"] < 0.05].tolist()

    show("6B  WHAT SURVIVES CORRECTION")
    print(f"  rules significant at p<0.05 on their own:        {naive} of "
          f"{len(stats)}")
    print(f"  expected by chance alone if nothing works:       "
          f"{0.05 * len(stats):.1f}")
    print(f"  rules surviving Holm-Bonferroni at 0.05:         {len(survivors)}")
    if survivors:
        print("    " + "; ".join(survivors))
    print()
    best_random = stats.loc[stats.index.str.startswith("RANDOM"), "t"].abs().max()
    print(f"  best |t| achieved by a rule that is literally a coin flip: "
          f"{best_random:.2f}")
    print("  -- that number is the honest yardstick for every other row above.")

    rc = reality_check(fam, args.n_boot, args.seed)

    show("6C  WHITE'S REALITY CHECK")
    print("Distribution of the BEST |t| in a family this size and this correlated,")
    print(f"under the null that no rule works. {rc['n_boot']:,} joint day-resamples.")
    print()
    print(f"  best |t| actually observed        {rc['best_observed_t']:.2f}   "
          f"({rc['best_rule']})")
    print(f"  median best |t| under the null    {rc['null_p50']:.2f}")
    print(f"  95th percentile under the null    {rc['null_p95']:.2f}")
    print(f"  99th percentile under the null    {rc['null_p99']:.2f}")
    print()
    print(f"  reality-check p-value             {rc['p_value']:.4f}")
    print()
    if rc["p_value"] < 0.05:
        print("  -> The best rule beats what a family this size produces by accident.")
        print("     It is still one rule out of many and deserves its own holdout.")
    else:
        print("  -> The best rule in the family does NOT beat what a family this")
        print("     size produces by accident. Every apparently-significant result")
        print("     in 6A is consistent with having looked at 30 things.")

    show("6D  THE SPLIT, FOR THE RULES THAT LOOKED BEST")
    print("Top six by |t|, scored separately on each period. A real effect does")
    print("not live in one half.")
    print()
    rows = []
    for name in stats.index[:6]:
        v = fam[name]
        for label, mask in (("2020-2023", panel["year"] < OOS_FROM),
                            ("2024-2025", panel["year"] >= OOS_FROM)):
            sub = v[mask].dropna()
            if len(sub) < 30:
                continue
            rows.append({
                "rule": name, "period": label, "n": len(sub),
                "mean_pips": round(sub.mean(), 3),
                "t": round(sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))), 2),
            })
    split = pd.DataFrame(rows)
    print(split.to_string(index=False))

    # Replication is a weaker bar than the reality check, but it is a different
    # question: the reality check asks "is this better than the best accident
    # in a family of 28", replication asks "does it show up twice". A rule that
    # fails the first and passes the second is not tradable evidence -- it is
    # the only thing in here worth another look.
    wide = split.pivot(index="rule", columns="period", values="mean_pips")
    if wide.shape[1] == 2:
        a, b = wide.columns
        agree = wide[(np.sign(wide[a]) == np.sign(wide[b]))
                     & (wide[a].abs() > args.cost_pips)
                     & (wide[b].abs() > args.cost_pips)]
        print()
        if agree.empty:
            print("  None of the leaders hold the same sign at a tradable size in")
            print("  both periods. That is the expected result when nothing is real.")
        else:
            print("  Same sign AND bigger than cost in BOTH periods:")
            for rule in agree.index:
                print(f"    {rule:<28s} {a} {agree.loc[rule, a]:+.2f}   "
                      f"{b} {agree.loc[rule, b]:+.2f} pips/day")
            print("  This is NOT a green light -- it already failed 6C. It is the")
            print("  short list worth testing on data none of this has touched.")

    print()
    print("-" * 78)
    print("Reading it: 6A is what selective reporting looks like from the inside.")
    print("6C is the number that matters. If it says nothing survives, that is not")
    print("a failed test -- it is the finding, and it is consistent with tests 1,")
    print("3 and 5D. The tradable structure in this data is in the SIZE of moves")
    print("(tests 2, 4, 5A-5C), which is where the effort belongs.")


if __name__ == "__main__":
    main()
