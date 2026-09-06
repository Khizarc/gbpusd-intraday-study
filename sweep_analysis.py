"""Liquidity sweep study for GBP/USD, New York (== Toronto) clock.

Two reference levels, run through identical machinery:

  ASIA    level from 19:00-02:59, finalized 03:00, traded 03:00-13:59
  LONDON  level from 03:00-08:59, finalized 09:00, traded 09:00-13:59

Every qualifying sweep is counted (no first-per-day restriction). MFE/MAE/net
are measured from the sweep candle's close to 14:00, so the forward window
shrinks as the hour advances -- which is why pooled sweep-vs-baseline numbers
are confounded and an hour-matched comparison is reported alongside them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fxlib import load_m1

PARQUET = Path(__file__).parent / "gbpusd_m1.parquet"
PIP = 10_000
SEED = 42
THRESHOLDS = (0.3, 0.5, 0.7)
MAIN_THRESHOLD = 0.5

STUDIES = {
    "ASIA": dict(level_hours=(19, 20, 21, 22, 23, 0, 1, 2), roll_evening=True,
                 trade=(3, 13)),
    "LONDON": dict(level_hours=(3, 4, 5, 6, 7, 8), roll_evening=False,
                   trade=(9, 13)),
}


def session_day(index: pd.DatetimeIndex, roll_evening: bool) -> np.ndarray:
    """Naive session-date label. Evening bars (>=19:00) roll to the next day.

    Kept naive on purpose: adding a day to a tz-aware midnight lands on 23:00
    or 01:00 across a DST boundary, which would corrupt the grouping.
    """
    base = pd.to_datetime(index.date)
    if not roll_evening:
        return base.values
    return (base + pd.to_timedelta((index.hour >= 19).astype(int), unit="D")).values


def build_candles(df: pd.DataFrame) -> pd.DataFrame:
    s = df.set_index("timestamp_ny").sort_index()
    c = s.resample("15min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), n=("close", "size"),
    )
    return c[c["n"] > 0]


def levels(c: pd.DataFrame, level_hours, roll_evening: bool) -> pd.DataFrame:
    sel = c[c.index.hour.isin(level_hours)]
    sess = session_day(sel.index, roll_evening)
    lev = sel.groupby(sess).agg(
        lvl_high=("high", "max"), lvl_low=("low", "min"), lvl_bars=("n", "sum"),
    )
    lev.index.name = "session_day"
    return lev


def build_trading(c: pd.DataFrame, lev: pd.DataFrame, trade) -> pd.DataFrame:
    lo, hi = trade
    h = c.index.hour
    t = c[(h >= lo) & (h <= hi)].copy()
    t["session_day"] = session_day(t.index, roll_evening=False)
    t["hour"] = t.index.hour
    t = t[t["session_day"].dt.dayofweek <= 4]          # Mon-Fri trading days
    t = t.merge(lev, left_on="session_day", right_index=True, how="inner")

    # Forward-looking extremes, strictly AFTER each candle, within the same day.
    g = t.groupby("session_day", sort=False)
    t["fwd_min_low"] = g["low"].transform(lambda x: x[::-1].cummin()[::-1].shift(-1))
    t["fwd_max_high"] = g["high"].transform(lambda x: x[::-1].cummax()[::-1].shift(-1))
    t["close_1400"] = g["close"].transform("last")     # last close before 14:00
    return t


def excursions(close, fwd_lo, fwd_hi, last, short):
    """MFE/MAE as positive magnitudes; net signed so positive == favourable."""
    down = (close - fwd_lo) * PIP      # favourable for a short
    up = (fwd_hi - close) * PIP        # favourable for a long
    net_short = (close - last) * PIP
    return {
        "mfe": np.where(short, down, up).clip(min=0),
        "mae": np.where(short, up, down).clip(min=0),
        "net": np.where(short, net_short, -net_short),
    }


def find_sweeps(t: pd.DataFrame) -> pd.DataFrame:
    bear = (t["high"] > t["lvl_high"]) & (t["close"] < t["lvl_high"])
    bull = (t["low"] < t["lvl_low"]) & (t["close"] > t["lvl_low"])

    both = int((bear & bull).sum())
    if both:  # resolve by the deeper poke
        poke_b = (t["high"] - t["lvl_high"]).where(bear, -np.inf)
        poke_l = (t["lvl_low"] - t["low"]).where(bull, -np.inf)
        keep_bear = poke_b >= poke_l
        bear, bull = bear & keep_bear, bull & ~keep_bear

    sel = bear | bull
    sw = t[sel].copy()
    sw["direction"] = np.where(bear[sel], "bearish", "bullish")
    sw["short"] = sw["direction"] == "bearish"

    body_hi = sw[["open", "close"]].max(axis=1)
    body_lo = sw[["open", "close"]].min(axis=1)
    span = (sw["high"] - sw["low"]).replace(0, np.nan)     # avoid 0/0
    sw["wick_ratio"] = np.where(
        sw["short"], (sw["high"] - body_hi) / span, (body_lo - sw["low"]) / span)
    sw["pips_poked"] = np.where(
        sw["short"], (sw["high"] - sw["lvl_high"]) * PIP,
        (sw["lvl_low"] - sw["low"]) * PIP)

    ex = excursions(sw["close"].values, sw["fwd_min_low"].values,
                    sw["fwd_max_high"].values, sw["close_1400"].values,
                    sw["short"].values)
    for k, v in ex.items():
        sw[k] = v

    sw = sw.sort_index()
    sw.attrs["both_direction_candles"] = both
    sw.attrs["zero_span_candles"] = int(span.isna().sum())
    return sw


def build_baseline(t: pd.DataFrame, sweeps: pd.DataFrame) -> pd.DataFrame:
    """One random candle per (day, hour) cell containing no qualifying sweep."""
    rng = np.random.default_rng(SEED)

    counts = sweeps.groupby(["hour", "direction"]).size().unstack(fill_value=0)
    for col in ("bearish", "bullish"):
        if col not in counts:
            counts[col] = 0
    bear_bias = (counts["bearish"] > counts["bullish"]).to_dict()

    taken = set(zip(sweeps["session_day"], sweeps["hour"]))
    keys = pd.MultiIndex.from_arrays([t["session_day"], t["hour"]])
    pool = (t[~keys.isin(taken)].reset_index()
             .sort_values(["session_day", "hour"]).reset_index(drop=True))

    sizes = pool.groupby(["session_day", "hour"], sort=True).size().to_numpy()
    starts = np.concatenate([[0], sizes.cumsum()[:-1]])
    b = pool.iloc[starts + rng.integers(0, sizes)].copy()

    biased = b["hour"].map(bear_bias).fillna(False).to_numpy()
    b["short"] = np.where(biased, True, rng.random(len(b)) < 0.5)
    b["direction"] = np.where(b["short"], "bearish", "bullish")

    ex = excursions(b["close"].values, b["fwd_min_low"].values,
                    b["fwd_max_high"].values, b["close_1400"].values,
                    b["short"].values)
    for k, v in ex.items():
        b[k] = v
    b.attrs["bear_bias_hours"] = [h for h, v in bear_bias.items() if v]
    return b


def medians(frame: pd.DataFrame, trade) -> pd.DataFrame:
    out = frame.dropna(subset=["mfe"]).groupby("hour").agg(
        median_mfe=("mfe", "median"), median_mae=("mae", "median"),
        median_net=("net", "median"), n=("mfe", "size"),
    )
    return out.reindex(range(trade[0], trade[1] + 1)).round(2)


def run_study(name: str, c: pd.DataFrame, cfg: dict) -> dict:
    lev = levels(c, cfg["level_hours"], cfg["roll_evening"])
    t = build_trading(c, lev, cfg["trade"])
    sweeps = find_sweeps(t)
    main_sw = sweeps[sweeps["wick_ratio"] >= MAIN_THRESHOLD]
    base = build_baseline(t, main_sw)
    trade = cfg["trade"]

    bar = "=" * 68
    print(bar)
    print("{}  level {:02d}:00-{:02d}:59  ->  traded {:02d}:00-{:02d}:59"
          .format(name, min(cfg["level_hours"]) if not cfg["roll_evening"] else 19,
                  (max(cfg["level_hours"]) if not cfg["roll_evening"] else 2),
                  trade[0], trade[1]))
    print(bar)
    print("sessions {:,}; median level M1 bars {:.0f}; trading candles {:,} "
          "over {:,} weekdays".format(len(lev), lev["lvl_bars"].median(),
                                      len(t), t["session_day"].nunique()))
    print("candles meeting BOTH directions: {}; zero-span: {}"
          .format(sweeps.attrs["both_direction_candles"],
                  sweeps.attrs["zero_span_candles"]))
    days_with = sweeps["session_day"].nunique()
    print("days with >=1 sweep (any wick): {:,} of {:,}; sweeps per such day {:.2f}"
          .format(days_with, t["session_day"].nunique(), len(sweeps) / days_with))
    print()

    print("1. SWEEP COUNTS BY HOUR, AT EACH WICK THRESHOLD")
    tbl = pd.DataFrame({
        "wick>={}".format(th): sweeps[sweeps["wick_ratio"] >= th].groupby("hour").size()
        for th in THRESHOLDS
    }).reindex(range(trade[0], trade[1] + 1)).fillna(0).astype(int)
    tbl["all_sweeps"] = (sweeps.groupby("hour").size()
                         .reindex(tbl.index).fillna(0).astype(int))
    print(tbl.to_string())
    print("totals:", {col: int(tbl[col].sum()) for col in tbl.columns})
    print()

    print("2. SWEEPS AT WICK >= {}: MEDIAN MFE / MAE / NET (pips)".format(MAIN_THRESHOLD))
    print(medians(main_sw, trade).to_string())
    print("by direction:")
    print(main_sw.dropna(subset=["mfe"]).groupby("direction")[["mfe", "mae", "net"]]
          .median().round(2).to_string())
    print()

    print("3. BASELINE: MEDIAN MFE / MAE / NET (pips)")
    print("hours treated as short-biased:", base.attrs["bear_bias_hours"])
    print(medians(base, trade).to_string())
    print()

    s_ok, b_ok = main_sw.dropna(subset=["mfe"]), base.dropna(subset=["mfe"])
    print("4. TOTALS")
    print("sweeps (wick>={}): {:,} ({:,} with a forward window)"
          .format(MAIN_THRESHOLD, len(main_sw), len(s_ok)))
    print("baseline samples:  {:,} ({:,} with a forward window)"
          .format(len(base), len(b_ok)))
    print("median pips poked beyond level: {:.2f}".format(main_sw["pips_poked"].median()))
    print()
    print("pooled (CONFOUNDED by hour mix)   MFE     MAE     NET")
    print("  sweeps                      {:7.2f} {:7.2f} {:7.2f}"
          .format(s_ok["mfe"].median(), s_ok["mae"].median(), s_ok["net"].median()))
    print("  baseline                    {:7.2f} {:7.2f} {:7.2f}"
          .format(b_ok["mfe"].median(), b_ok["mae"].median(), b_ok["net"].median()))
    print()

    w = s_ok["hour"].value_counts(normalize=True).sort_index()
    sm = s_ok.groupby("hour")[["mfe", "mae", "net"]].median()
    bm = b_ok.groupby("hour")[["mfe", "mae", "net"]].median()
    common = w.index.intersection(bm.index)
    w = w[common] / w[common].sum()
    print("HOUR-MATCHED (baseline reweighted to the sweep hour mix)")
    for col in ("mfe", "mae", "net"):
        sv = float((sm.loc[common, col] * w).sum())
        bv = float((bm.loc[common, col] * w).sum())
        print("  {:4s}  sweeps {:7.2f}   baseline {:7.2f}   delta {:+7.2f}"
              .format(col.upper(), sv, bv, sv - bv))
    print()
    print("  MFE>MAE share   sweeps {:.3f}   baseline {:.3f}"
          .format(float((s_ok["mfe"] > s_ok["mae"]).mean()),
                  float((b_ok["mfe"] > b_ok["mae"]).mean())))
    print("  net>0 share     sweeps {:.3f}   baseline {:.3f}"
          .format(float((s_ok["net"] > 0).mean()), float((b_ok["net"] > 0).mean())))
    print()
    print("  per-hour delta (sweep minus baseline):")
    d = (sm - bm).round(2)
    d["n_sweeps"] = s_ok.groupby("hour").size()
    print(d.to_string())
    print()

    main_sw.to_csv(Path(__file__).parent / "sweeps_{}.csv".format(name.lower()),
                   index=False)
    return {"sweeps": sweeps, "main": main_sw, "base": base}


def four_hour_trend(df: pd.DataFrame, sma_len: int = 50) -> pd.DataFrame:
    """4H bars anchored at 00:00 UTC with a 50-period SMA of the close.

    available_at is the bar's CLOSE time (bin start + 4h). Tagging a sweep by
    the last bar whose available_at <= sweep time is what keeps this free of
    lookahead: a bar is only usable once it has finished.
    """
    s = df.set_index("timestamp_utc").sort_index()
    bars = s.resample("4h", origin="start_day").agg(
        close=("close", "last"), n=("close", "size"))
    bars = bars[bars["n"] > 0]                 # drop empty weekend bins first,
    bars["sma"] = bars["close"].rolling(sma_len).mean()   # so the SMA counts
    bars["trend"] = np.where(bars["close"] > bars["sma"], "bullish", "bearish")
    bars.loc[bars["sma"].isna(), "trend"] = None          # real bars only
    bars["available_at"] = bars.index + pd.Timedelta(hours=4)
    bars.attrs["exact_ties"] = int((bars["close"] == bars["sma"]).sum())
    return bars


def tag_trend(frame: pd.DataFrame, ts_ny: pd.Series, bars: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recently closed 4H bar's trend state to each row."""
    f = frame.copy()
    f["ts_utc"] = pd.DatetimeIndex(ts_ny).tz_convert("UTC")
    f = f.sort_values("ts_utc")
    right = bars[["available_at", "trend"]].sort_values("available_at")
    out = pd.merge_asof(f, right, left_on="ts_utc", right_on="available_at",
                        direction="backward", allow_exact_matches=True)
    return out


def hour_match(s_ok: pd.DataFrame, b_ok: pd.DataFrame) -> dict:
    """Baseline medians reweighted to the sweep group's hour mix."""
    w = s_ok["hour"].value_counts(normalize=True).sort_index()
    sm = s_ok.groupby("hour")[["mfe", "mae", "net"]].median()
    bm = b_ok.groupby("hour")[["mfe", "mae", "net"]].median()
    k = w.index.intersection(bm.index)
    w = w[k] / w[k].sum()
    res = {}
    for col in ("mfe", "mae", "net"):
        res[col] = (float((sm.loc[k, col] * w).sum()),
                    float((bm.loc[k, col] * w).sum()))
    res["hit"] = (float((s_ok["mfe"] > s_ok["mae"]).mean()),
                  float((b_ok["mfe"] > b_ok["mae"]).mean()))
    res["n"] = len(s_ok)
    res["n_base"] = len(b_ok)
    return res


def _report(label: str, r: dict) -> None:
    print("  {:<22s} n={:<5d} (baseline n={:,})".format(label, r["n"], r["n_base"]))
    for col in ("mfe", "mae", "net"):
        sv, bv = r[col]
        print("      {:4s} sweeps {:7.2f}   baseline {:7.2f}   delta {:+7.2f}"
              .format(col.upper(), sv, bv, sv - bv))
    hs, hb = r["hit"]
    se = np.sqrt(hs * (1 - hs) / max(r["n"], 1)
                 + hb * (1 - hb) / max(r["n_base"], 1))
    print("      MFE>MAE {:.3f}   baseline {:.3f}   delta {:+.3f}  (z={:+.2f})"
          .format(hs, hb, hs - hb, (hs - hb) / se if se else 0.0))


def trend_study(c: pd.DataFrame, df: pd.DataFrame, threshold: float) -> None:
    """London-level sweeps split by 4H trend alignment."""
    cfg = STUDIES["LONDON"]
    lev = levels(c, cfg["level_hours"], cfg["roll_evening"])
    t = build_trading(c, lev, cfg["trade"])
    sweeps = find_sweeps(t)
    main_sw = sweeps[sweeps["wick_ratio"] >= threshold]
    base = build_baseline(t, main_sw)

    bars = four_hour_trend(df)
    sw = tag_trend(main_sw, main_sw.index, bars)
    bs = tag_trend(base, base["timestamp_ny"], bars)

    sw = sw.dropna(subset=["mfe", "trend"])
    bs_ok = bs.dropna(subset=["mfe"])
    sw["aligned"] = np.where(
        ((sw["direction"] == "bearish") & (sw["trend"] == "bearish")) |
        ((sw["direction"] == "bullish") & (sw["trend"] == "bullish")),
        "ALIGNED", "COUNTER")

    print("=" * 68)
    print("LONDON SWEEPS x 4H TREND (50 SMA, bars anchored 00:00 UTC), "
          "wick>={}".format(threshold))
    print("=" * 68)
    print("4H bars {:,} (SMA warmup drops {}); exact close==SMA ties {}"
          .format(len(bars), int(bars["trend"].isna().sum()),
                  bars.attrs["exact_ties"]))
    print("4H bar trend mix:", bars["trend"].value_counts().to_dict())
    print()
    print("counts:")
    print(pd.crosstab(sw["direction"], sw["trend"]).to_string())
    print(sw["aligned"].value_counts().to_string())
    print()

    print("vs the SAME baseline as before (direction assigned by hour bias/coin):")
    for grp in ("ALIGNED", "COUNTER"):
        g = sw[sw["aligned"] == grp]
        _report(grp, hour_match(g, bs_ok))
    print()

    # Confound check: in a trending regime a random trade WITH the trend already
    # earns drift. Re-score baseline candles in the group's own trend state and
    # forced to the group's own direction, so only the sweep signal is left.
    print("vs a TREND-CONDITIONED baseline (same regime, same direction):")
    for grp in ("ALIGNED", "COUNTER"):
        g = sw[sw["aligned"] == grp]
        parts = []
        for direction in ("bearish", "bullish"):
            sub = g[g["direction"] == direction]
            if sub.empty:
                continue
            regime = sub["trend"].iloc[0]
            pool = bs_ok[bs_ok["trend"] == regime].copy()
            ex = excursions(pool["close"].values, pool["fwd_min_low"].values,
                            pool["fwd_max_high"].values, pool["close_1400"].values,
                            np.full(len(pool), direction == "bearish"))
            for k, v in ex.items():
                pool[k] = v
            parts.append(pool)
        _report(grp, hour_match(g, pd.concat(parts)))
    print()


def main() -> None:
    # load_m1 checks the parquet's build metadata and raises if it was
    # built without cleaning, so a stale rebuild fails loudly here
    # rather than quietly changing every number below.
    df = load_m1(PARQUET)
    c = build_candles(df)
    for name, cfg in STUDIES.items():
        run_study(name, c, cfg)
    for th in (MAIN_THRESHOLD, 0.3):
        trend_study(c, df, th)


if __name__ == "__main__":
    main()
