"""TEST 3 -- After the open, is the first break of the range worth following?

The most-traded intraday pattern there is, put on trial. Mark the high and low
of the first N minutes after a session opens, wait for price to leave that
range, and take the break in the direction it left. Does that make money, or
is it just a way of paying the spread at the moment volatility is highest?

Two sessions, both firmly inside the daytrading day and both flat by the New
York afternoon:

    LONDON    range 03:00 -> 03:00+N, traded until 16:00 New York
    NEW YORK  range 08:00 -> 08:00+N, traded until 16:00 New York

Every trade is fully specified before it is taken, so there is nothing left to
tune after seeing the result:

    entry   a stop order at the range edge, filled at the edge
    stop    the opposite edge of the range
    risk    R = the range height, so results are in R and comparable across
            quiet and busy days without any volatility scaling
    target  a fixed multiple of R
    exit    target, stop, or the session cutoff, whichever comes first

What keeps this from flattering itself:

  costs        Spread and slippage are charged on every trade. At a 20-pip
               range a 1-pip cost is 5% of R, which is roughly the size of the
               entire effect being looked for -- a gross-of-cost ORB backtest
               is not a backtest.
  same-bar     If one minute bar spans both target and stop, this counts it as
               a loss. The feed cannot say which came first, and assuming the
               good one is how a mediocre rule is made to look excellent.
  a benchmark  Every result is set against a coin-flip entry taken at the same
               time of day with the same stop and target distances. That is
               the number that matters: an ORB rule that wins 40% at 2R is
               profitable, so a raw win rate says nothing on its own.
  the fade     The mirror rule -- take the break in the OPPOSITE direction --
               is scored alongside. If breakouts do not work, the interesting
               question is whether the other side of the same trade does.
  a holdout    Fitted on nothing, but reported on 2020-2023 and 2024-2025
               separately, with a bootstrap that resamples whole days.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxlib import PIP, load_m1, day_block_bootstrap, session_day

SESSIONS = {"LONDON": 3, "NEWYORK": 8}
OR_LENGTHS = (15, 30, 60)
TARGET_R = 2.0
DAY_CLOSE = 16              # flat by 16:00 New York, always
DEFAULT_COST_PIPS = 1.0

MIN_OR_COVERAGE = 0.70      # of the N minutes the range is measured over
MIN_RANGE_PIPS = 3.0        # below this the "range" is a frozen feed, not a range
MIN_FORWARD_BARS = 60       # a session with less room left than this is not traded
OOS_FROM = 2024
N_BOOT = 2000


# --------------------------------------------------------------------------
# trade mechanics
# --------------------------------------------------------------------------

def resolve(high: np.ndarray, low: np.ndarray, last_close: float,
            entry: float, stop: float, target: float, is_long: bool,
            risk_pips: float, cost_pips: float) -> tuple:
    """Walk one trade forward. Returns (R multiple, outcome label, bars held).

    Ties inside a single bar go to the stop. The minute bar records only its
    high and low, not their order, so a bar containing both levels is genuinely
    ambiguous -- and resolving ambiguity in the trade's favour is the single
    most common way an intraday backtest invents an edge.
    """
    if high.size == 0:
        return np.nan, "no_data", 0

    hit_t = (high >= target) if is_long else (low <= target)
    hit_s = (low <= stop) if is_long else (high >= stop)

    i_t = int(np.argmax(hit_t)) if hit_t.any() else len(high)
    i_s = int(np.argmax(hit_s)) if hit_s.any() else len(high)

    if i_t < i_s:
        pnl, outcome, bars = TARGET_R * risk_pips, "target", i_t + 1
    elif i_s < len(high):
        pnl, outcome, bars = -risk_pips, "stop", i_s + 1
    else:
        move = (last_close - entry) if is_long else (entry - last_close)
        pnl, outcome, bars = move * PIP, "timeout", len(high)

    return (pnl - cost_pips) / risk_pips, outcome, bars


def session_trades(day_bars: pd.DataFrame, open_hour: int, or_len: int,
                   cost_pips: float, rng: np.random.Generator) -> dict | None:
    """Build the breakout, the fade and the coin-flip control for one session.

    All three share the same entry minute, the same risk distance and the same
    forward window, so the only thing that differs between them is the decision
    rule. That is what makes the comparison mean anything.
    """
    hour = day_bars.index.hour
    minute = hour * 60 + day_bars.index.minute
    open_min = open_hour * 60

    in_range = (minute >= open_min) & (minute < open_min + or_len)
    forward = (minute >= open_min + or_len) & (hour < DAY_CLOSE)

    or_bars = day_bars[in_range]
    fwd = day_bars[forward]

    if len(or_bars) < MIN_OR_COVERAGE * or_len or len(fwd) < MIN_FORWARD_BARS:
        return None

    or_hi, or_lo = or_bars["high"].max(), or_bars["low"].min()
    risk_pips = (or_hi - or_lo) * PIP
    if risk_pips < MIN_RANGE_PIPS:
        return None

    high = fwd["high"].to_numpy()
    low = fwd["low"].to_numpy()
    last_close = float(fwd["close"].iloc[-1])

    # First departure from the range, in either direction.
    up, dn = high >= or_hi, low <= or_lo
    i_up = int(np.argmax(up)) if up.any() else len(high)
    i_dn = int(np.argmax(dn)) if dn.any() else len(high)

    if i_up == len(high) and i_dn == len(high):
        return {"outcome": "no_break", "risk_pips": risk_pips}
    if i_up == i_dn:
        # One bar left the range on both sides. Which edge it touched first is
        # unknowable from OHLC, so the setup is dropped rather than guessed.
        return {"outcome": "ambiguous_break", "risk_pips": risk_pips}

    is_long = i_up < i_dn
    i = min(i_up, i_dn)
    entry = or_hi if is_long else or_lo

    # The break bar itself is included: a bar can spike through the entry and
    # reach the stop within the same minute, and that trade did happen.
    h, l = high[i:], low[i:]

    span = or_hi - or_lo
    go_r, go_out, go_bars = resolve(
        h, l, last_close, entry,
        stop=or_lo if is_long else or_hi,
        target=entry + TARGET_R * span if is_long else entry - TARGET_R * span,
        is_long=is_long, risk_pips=risk_pips, cost_pips=cost_pips)

    # The fade: same entry price and same risk, opposite direction.
    fade_long = not is_long
    fade_entry = entry
    fade_r, fade_out, _ = resolve(
        h, l, last_close, fade_entry,
        stop=fade_entry - span if fade_long else fade_entry + span,
        target=(fade_entry + TARGET_R * span if fade_long
                else fade_entry - TARGET_R * span),
        is_long=fade_long, risk_pips=risk_pips, cost_pips=cost_pips)

    # The control: a coin flip entered at the open of the forward window, with
    # identical geometry. It answers "is this just what any 2R trade pays here?"
    ctrl_long = bool(rng.random() < 0.5)
    ctrl_entry = float(fwd["open"].iloc[0])
    ctrl_r, ctrl_out, _ = resolve(
        high, low, last_close, ctrl_entry,
        stop=ctrl_entry - span if ctrl_long else ctrl_entry + span,
        target=(ctrl_entry + TARGET_R * span if ctrl_long
                else ctrl_entry - TARGET_R * span),
        is_long=ctrl_long, risk_pips=risk_pips, cost_pips=cost_pips)

    return {
        "outcome": "traded",
        "direction": "long" if is_long else "short",
        "risk_pips": risk_pips,
        "minutes_to_break": int(minute.to_numpy()[forward][i] - (open_min + or_len)),
        "break_r": go_r, "break_outcome": go_out, "break_bars": go_bars,
        "fade_r": fade_r, "fade_outcome": fade_out,
        "ctrl_r": ctrl_r, "ctrl_outcome": ctrl_out,
    }


def run_session(s: pd.DataFrame, open_hour: int, or_len: int,
                cost_pips: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for day, bars in s.groupby(session_day(s.index)):
        if pd.Timestamp(day).dayofweek > 4:
            continue
        res = session_trades(bars, open_hour, or_len, cost_pips, rng)
        if res is not None:
            rows.append({"day": day, **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["year"] = pd.DatetimeIndex(out["day"]).year
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def summarize(traded: pd.DataFrame, col: str) -> dict:
    r = traded[f"{col}_r"].dropna().to_numpy()
    if r.size == 0:
        return {"n": 0}
    outcomes = traded[f"{col}_outcome"].value_counts(normalize=True)
    return {
        "n": r.size,
        "expectancy_R": float(r.mean()),
        "median_R": float(np.median(r)),
        "win_rate": float((r > 0).mean()),
        "target_rate": float(outcomes.get("target", 0.0)),
        "stop_rate": float(outcomes.get("stop", 0.0)),
        "timeout_rate": float(outcomes.get("timeout", 0.0)),
        "total_R": float(r.sum()),
    }


def expectancy_ci(traded: pd.DataFrame, col: str, seed: int) -> tuple:
    """Day-block bootstrap CI on mean R.

    Days are the resampling unit even though there is one trade per day per
    session, because the same day feeds the London and New York books and
    volatility clusters across adjacent trades.
    """
    values = traded[f"{col}_r"].to_numpy()
    ok = np.isfinite(values)
    if ok.sum() < 30:
        return (np.nan, np.nan)
    return day_block_bootstrap(
        traded.loc[ok, "day"].to_numpy(),
        lambda idx: float(np.mean(values[ok][idx])),
        n_boot=N_BOOT, seed=seed)


def score_table(traded: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = {}
    for col, label in (("break", "BREAKOUT"), ("fade", "FADE"), ("ctrl", "coin flip")):
        stats = summarize(traded, col)
        lo, hi = expectancy_ci(traded, col, seed)
        stats["ci_lo_R"], stats["ci_hi_R"] = lo, hi
        stats["verdict"] = ("profitable" if np.isfinite(lo) and lo > 0 else
                            "loses money" if np.isfinite(hi) and hi < 0 else
                            "indistinguishable from zero")
        # Round here rather than on the frame: the verdict column makes the
        # transposed frame object-dtype, and .round() silently skips it.
        rows[label] = {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in stats.items()}
    return pd.DataFrame(rows).T


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
    ap.add_argument("--cost-pips", type=float, default=DEFAULT_COST_PIPS,
                    help="Round-trip spread plus slippage, in pips "
                         f"(default {DEFAULT_COST_PIPS}).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_m1()
    s = df.set_index("timestamp_ny").sort_index()

    show("TEST 3  OPENING RANGE BREAKOUT -- follow the break, or fade it?")
    print(f"target {TARGET_R:.0f}R, stop at the opposite range edge, flat by "
          f"{DAY_CLOSE:02d}:00 New York")
    print(f"cost charged {args.cost_pips:.2f} pips round trip on every trade")
    print("same-bar target-and-stop resolved as a LOSS throughout")

    all_results = {}
    for name, open_hour in SESSIONS.items():
        for or_len in OR_LENGTHS:
            key = (name, or_len)
            all_results[key] = run_session(s, open_hour, or_len,
                                           args.cost_pips, args.seed)

    show("3A  SETUP FREQUENCY -- how often the pattern is even available")
    freq = []
    for (name, or_len), res in all_results.items():
        counts = res["outcome"].value_counts()
        total = len(res)
        freq.append({
            "session": name, "OR_min": or_len, "sessions": total,
            "traded": counts.get("traded", 0),
            "no_break": counts.get("no_break", 0),
            "ambiguous": counts.get("ambiguous_break", 0),
            "traded_pct": round(100 * counts.get("traded", 0) / total, 1),
            "median_risk_pips": round(res["risk_pips"].median(), 1),
        })
    print(pd.DataFrame(freq).to_string(index=False))
    print()
    print("'ambiguous' is a single minute bar that left the range on both sides;")
    print("those are discarded rather than guessed, which is why they are counted.")

    show(f"3B  FULL SAMPLE  -- expectancy in R per trade, net of "
         f"{args.cost_pips:.2f} pips")
    print("The coin-flip row is the benchmark. A breakout row only means something")
    print("if it clears that row, not if it merely clears zero.")
    print("CI is a 95% day-block bootstrap; the verdict reads the CI.")

    for (name, or_len), res in all_results.items():
        traded = res[res["outcome"] == "traded"]
        if len(traded) < 50:
            continue
        print()
        print(f"  {name}  opening range {or_len} min   "
              f"({len(traded):,} trades, median risk "
              f"{traded['risk_pips'].median():.1f} pips, "
              f"median {traded['minutes_to_break'].median():.0f} min to break)")
        tbl = score_table(traded, args.seed)
        cols = ["n", "expectancy_R", "win_rate", "target_rate", "stop_rate",
                "timeout_rate", "ci_lo_R", "ci_hi_R", "verdict"]
        print(tbl[cols].to_string().replace("\n", "\n  "))

    show("3C  IN-SAMPLE / OUT-OF-SAMPLE  -- does any of it survive the split?")
    print("Nothing here was fitted, so this is a stability check rather than a")
    print("validation: an edge that only exists in one half is not an edge.")
    print()
    rows = []
    for (name, or_len), res in all_results.items():
        traded = res[res["outcome"] == "traded"]
        for label, mask in (("2020-2023", traded["year"] < OOS_FROM),
                            ("2024-2025", traded["year"] >= OOS_FROM)):
            sub = traded[mask]
            if len(sub) < 30:
                continue
            rows.append({
                "session": name, "OR_min": or_len, "period": label, "n": len(sub),
                "break_R": round(summarize(sub, "break")["expectancy_R"], 3),
                "fade_R": round(summarize(sub, "fade")["expectancy_R"], 3),
                "coin_R": round(summarize(sub, "ctrl")["expectancy_R"], 3),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    show("3D  WHERE THE BREAK HAPPENS -- does an early break behave differently?")
    print("A break in the first few minutes is momentum arriving; one that shows up")
    print("hours later is drift. Split at the median wait, per session.")
    print()
    rows = []
    for (name, or_len), res in all_results.items():
        traded = res[res["outcome"] == "traded"].copy()
        if len(traded) < 100:
            continue
        cut = traded["minutes_to_break"].median()
        for label, mask in (("early", traded["minutes_to_break"] <= cut),
                            ("late", traded["minutes_to_break"] > cut)):
            sub = traded[mask]
            if len(sub) < 30:
                continue
            st = summarize(sub, "break")
            rows.append({
                "session": name, "OR_min": or_len, "timing": label,
                "cut_min": int(cut), "n": st["n"],
                "break_R": round(st["expectancy_R"], 3),
                "win_rate": round(st["win_rate"], 3),
                "coin_R": round(summarize(sub, "ctrl")["expectancy_R"], 3),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("-" * 78)
    print("Reading it: check 3B's breakout CI against the coin-flip row in the same")
    print("block first. If they overlap, the range edge carried no information and")
    print("the pattern is only selecting for volatility. Then check 3C -- a result")
    print("present in one period and absent in the other is a period, not an edge.")
    print(f"Re-run with --cost-pips 0 to see how much of any result is the spread,")
    print("and with a wider cost to see what it survives at a worse broker.")


if __name__ == "__main__":
    main()
