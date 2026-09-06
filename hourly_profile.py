"""Hourly volatility profile for GBP/USD, bucketed by New York local clock.

Reads gbpusd_m1.parquet (build it with --dedupe --exclude-2023-h1), resamples
the M1 bars into 1H OHLC bars on the timestamp_ny wall clock, and reports
range and net move in pips by hour of day. Sunday sessions are excluded.
"""

import argparse
from pathlib import Path

import pandas as pd

from fxlib import load_m1

PARQUET = Path(__file__).parent / "gbpusd_m1.parquet"
PIP = 10_000  # quotes carry 6dp, but a pip is still 0.0001


def build_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """M1 -> 1H OHLC on the NY wall clock, empty and Sunday buckets removed."""
    s = df.set_index("timestamp_ny").sort_index()

    bars = s.resample("1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        m1_bars=("close", "size"),
    )

    # Weekend/holiday gaps produce empty buckets; drop them rather than
    # letting them count as zero-range hours.
    bars = bars[bars["m1_bars"] > 0]

    # dayofweek: Monday=0 ... Sunday=6. Drops the Sunday-evening reopen.
    bars = bars[bars.index.dayofweek != 6]

    bars["range"] = (bars["high"] - bars["low"]) * PIP
    bars["net_move"] = (bars["close"] - bars["open"]).abs() * PIP
    return bars


def profile(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.groupby(bars.index.hour).agg(
        mean_range=("range", "mean"),
        median_range=("range", "median"),
        mean_net_move=("net_move", "mean"),
        median_net_move=("net_move", "median"),
        count=("range", "size"),
        median_m1_bars=("m1_bars", "median"),
    )
    out.index.name = "hour"
    return out.reindex(range(24))


def by_year(bars: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    """Hour-of-day (rows) x calendar year (columns) table of range in pips."""
    return bars.pivot_table(
        index=bars.index.hour,
        columns=bars.index.year,
        values="range",
        aggfunc=stat,
    ).rename_axis(index="hour", columns="year").reindex(range(24))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--by-year", action="store_true",
                        help="Median range per hour, broken out by calendar year.")
    args = parser.parse_args()

    # load_m1 checks the parquet's build metadata and raises if it was
    # built without cleaning, so a stale rebuild fails loudly here
    # rather than quietly changing every number below.
    df = load_m1(PARQUET)
    bars = build_hourly(df)

    if args.by_year:
        med = by_year(bars, "median").round(2)
        cnt = by_year(bars, "size").astype("Int64")
        print("GBP/USD MEDIAN hourly range (pips) -- timestamp_ny hour x year, Mon-Fri")
        print(f"source: {len(df):,} M1 rows -> {len(bars):,} hourly bars")
        print()
        print(med.to_string())
        print()
        print("Hourly-bar counts behind each cell:")
        print(cnt.to_string())
        print()
        print("column totals:", cnt.sum().to_dict())
        return

    out = profile(bars)

    display = out.drop(columns=["median_m1_bars"]).round(2)
    print("GBP/USD hourly profile -- pips, by timestamp_ny hour (Mon-Fri)")
    print(f"source: {len(df):,} M1 rows -> {len(bars):,} hourly bars\n")
    print(display.to_string())

    thin = out[out["median_m1_bars"] < 55]
    if not thin.empty:
        print("\nHours whose bars are typically INCOMPLETE (median M1 bars < 55 of 60):")
        print(thin[["count", "median_m1_bars"]].to_string())


if __name__ == "__main__":
    main()
