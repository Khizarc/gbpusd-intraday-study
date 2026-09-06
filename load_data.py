"""Build a validated, UTC-indexed M1 parquet for GBP/USD from HistData CSVs.

Source format (verified, not assumed): semicolon-delimited, no header, six
fields -- datetime;open;high;low;close;volume -- with datetime as
YYYYMMDD HHMMSS and prices quoted to 6dp (fractional pips). Volume is
identically zero for forex; it is checked and then dropped.

Timestamps are a FIXED UTC-5 offset year-round -- HistData never applies
daylight saving to these files. They are localized as a plain fixed offset
(not as America/New_York, which would wrongly impose DST shifts) and then
converted to UTC. A second column re-expresses the same instants on the real
New York wall clock, which is what session boundaries are defined on.

Cleaning is ON by default, because every downstream script needs the cleaned
build; --keep-duplicates and --keep-2023-h1 opt back out.

  duplicates   On the last Sunday of October each year (the day European
               summer time ends), the feed emits the 19:00-19:59 source-local
               hour twice. Both copies carry byte-identical OHLC and the
               surrounding series has no missing hour, so they are pure
               redundancy. That identity is re-verified on every build rather
               than trusted, and --dedupe refuses to run if it ever breaks.

  2023 H1      The feed's minute coverage collapses from Feb through Jul 2023.
               The COVERAGE section prints the per-month evidence; the
               exclusion covers Feb 1 - Jul 31 2023, New York local dates.

Everything the build learns is printed as a report and, unless --report-only,
recorded in the parquet's key-value metadata under "gbpusd_m1" so downstream
code can assert what it is reading (see read_build_metadata).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
DATA_DIR = HERE / "gbpusdData"
OUTPUT_PATH = HERE / "gbpusd_m1.parquet"
YEARS = range(2020, 2026)

# HistData's fixed offset. Deliberately NOT 'Etc/GMT+5' (POSIX sign convention
# makes that UTC-5's opposite in spelling and a common source of 10h errors)
# and NOT 'EST' (reads as US Eastern, which observes DST -- this data does not).
SOURCE_TZ = timezone(timedelta(hours=-5))
DISPLAY_TZ = "America/New_York"

EXCLUDE_START = pd.Timestamp("2023-02-01", tz=DISPLAY_TZ)
EXCLUDE_END = pd.Timestamp("2023-08-01", tz=DISPLAY_TZ)  # exclusive

RAW_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
PIP = 10_000  # GBP/USD is quoted to 6dp, but a pip is still 0.0001

# The FX week on the New York clock: opens Sunday 17:00, closes Friday 17:00.
# Used to size the "expected minutes" denominator for coverage.
WEEK_OPEN_DOW, WEEK_OPEN_HOUR = 6, 17    # Sunday
WEEK_CLOSE_DOW, WEEK_CLOSE_HOUR = 4, 17  # Friday

# A minute return this large is a candidate bad tick rather than a real move.
# 40x the median absolute return is far outside anything NFP or a BoE decision
# produced in this sample; the threshold only flags, it never drops.
OUTLIER_MAD_MULTIPLE = 40.0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass
class BuildReport:
    """Everything the build learned. Printed, and embedded in the parquet."""
    rows_raw: int = 0
    rows_out: int = 0
    rows_per_file: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_timestamps: int = 0
    duplicates_ohlc_identical: bool = True
    duplicates_dropped: bool = False
    excluded_rows: int = 0
    exclusion_applied: bool = False
    ohlc_violations: dict = field(default_factory=dict)
    nonpositive_rows: int = 0
    nonfinite_rows: int = 0
    nonzero_volume_rows: int = 0
    frozen_bars: int = 0
    outlier_returns: int = 0
    outlier_threshold_pips: float = 0.0
    first_utc: str = ""
    last_utc: str = ""

    def to_metadata(self) -> dict:
        payload = dict(self.__dict__)
        payload["rows_per_file"] = {str(k): v for k, v in self.rows_per_file.items()}
        payload["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {"gbpusd_m1": json.dumps(payload, default=str)}


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------

def read_year(year: int, report: BuildReport) -> pd.DataFrame:
    """Read one year's CSV and confirm the volume column really is all zero."""
    path = DATA_DIR / f"DAT_ASCII_GBPUSD_M1_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected data file is missing: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=RAW_COLUMNS,
        dtype={col: "float64" for col in [*PRICE_COLS, "volume"]},
    )
    report.rows_per_file[year] = len(df)
    report.nonzero_volume_rows += int((df["volume"] != 0).sum())
    return df.drop(columns=["volume"])


def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Attach timestamp_utc / timestamp_ny and sort stably by instant."""
    # Explicit format: inference on all-digit datetimes is slow and risky.
    naive = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")

    # A fixed offset cannot produce ambiguous or nonexistent times, so this
    # localization is total -- no DST gap/overlap handling is needed.
    df["timestamp_utc"] = naive.dt.tz_localize(SOURCE_TZ).dt.tz_convert("UTC")

    # Same instants, re-expressed with real DST rules so session boundaries
    # line up with actual New York local trading hours year-round.
    df["timestamp_ny"] = df["timestamp_utc"].dt.tz_convert(DISPLAY_TZ)

    df = df.drop(columns=["datetime"])
    df = df[["timestamp_utc", "timestamp_ny", *PRICE_COLS]]

    # Stable sort keeps original file order within any duplicated timestamp, so
    # "keep first" means "the one that appeared first in the source file".
    return df.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# validation -- these inspect the data, they never modify it
# --------------------------------------------------------------------------

def check_ohlc(df: pd.DataFrame, report: BuildReport) -> None:
    """Bar-level integrity. A violation here means the feed is unusable as-is."""
    o, h, l, c = (df[col].to_numpy() for col in PRICE_COLS)
    prices = df[PRICE_COLS].to_numpy()

    report.ohlc_violations = {
        "high_below_low": int((h < l).sum()),
        "high_below_open_or_close": int((h < np.maximum(o, c)).sum()),
        "low_above_open_or_close": int((l > np.minimum(o, c)).sum()),
    }
    report.nonpositive_rows = int((prices <= 0).any(axis=1).sum())
    report.nonfinite_rows = int((~np.isfinite(prices)).any(axis=1).sum())

    # A "frozen" bar has no range at all: the feed saw a single tick, or none
    # and carried the last one forward. They are legitimate rows, but they bias
    # every volatility estimate downward, so downstream code needs the count.
    report.frozen_bars = int((h == l).sum())


def check_duplicates(df: pd.DataFrame, report: BuildReport) -> pd.Series:
    """Locate repeated timestamps and verify their OHLC really are identical."""
    dup_mask = df["timestamp_utc"].duplicated(keep="first")
    report.duplicate_rows = int(dup_mask.sum())

    all_copies = df["timestamp_utc"].duplicated(keep=False)
    report.duplicate_timestamps = int(df.loc[all_copies, "timestamp_utc"].nunique())

    if report.duplicate_rows:
        spread = (df.loc[all_copies]
                    .groupby("timestamp_utc")[PRICE_COLS]
                    .nunique()
                    .max(axis=1))
        report.duplicates_ohlc_identical = bool((spread == 1).all())
    return dup_mask


def check_outliers(df: pd.DataFrame, report: BuildReport) -> pd.DataFrame:
    """Flag close-to-close jumps far outside the robust scale of the sample.

    Only consecutive minutes are compared: a jump measured across a weekend or
    a data gap is real repricing, not a bad tick, and would otherwise dominate
    the tail and hide the thing we are looking for.
    """
    contiguous = df["timestamp_utc"].diff() == pd.Timedelta(minutes=1)
    ret = np.log(df["close"]).diff().where(contiguous)

    threshold = OUTLIER_MAD_MULTIPLE * ret.abs().median()
    report.outlier_threshold_pips = float(threshold * PIP)

    hit = ret.abs() > threshold
    flagged = df.loc[hit].copy()
    flagged["jump_pips"] = (ret[hit] * PIP).round(1)
    report.outlier_returns = len(flagged)

    order = flagged["jump_pips"].abs().sort_values(ascending=False).index
    return flagged.loc[order]


def fx_week_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Mask: is this New York minute inside the Sun 17:00 - Fri 17:00 week?"""
    dow, hour = index.dayofweek, index.hour
    after_open = (dow == WEEK_OPEN_DOW) & (hour >= WEEK_OPEN_HOUR)
    before_close = (dow == WEEK_CLOSE_DOW) & (hour < WEEK_CLOSE_HOUR)
    weekday = dow < WEEK_CLOSE_DOW
    return after_open | before_close | weekday


def coverage_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Observed minutes vs minutes the FX week implies should exist, per month.

    Holidays count as expected, so every month runs a few points short of 100%;
    that steady baseline is exactly what makes a real outage stand out.
    """
    ny = pd.DatetimeIndex(df["timestamp_ny"])
    grid = pd.date_range(ny[0].floor("min"), ny[-1].ceil("min"),
                         freq="min", tz=DISPLAY_TZ)
    grid = grid[fx_week_minutes(grid)]

    expected = pd.Series(1, index=grid).resample("MS").sum()
    observed = pd.Series(1, index=ny.unique()).resample("MS").sum()

    out = pd.DataFrame({"expected": expected, "observed": observed}).fillna(0)
    out["observed"] = out["observed"].astype(int)
    out["coverage_pct"] = (100 * out["observed"] / out["expected"]).round(1)
    out.index = out.index.strftime("%Y-%m")
    out.index.name = "month"
    return out


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows and distinct trading days per calendar year, by New York local date."""
    ny = df["timestamp_ny"]
    year = ny.dt.year
    # normalize() collapses each instant to NY-local midnight, so distinct
    # values are distinct local dates -- i.e. days with at least one bar.
    summary = pd.DataFrame({
        "rows": ny.groupby(year).size(),
        "trading_days": ny.dt.normalize().groupby(year).nunique(),
    })
    summary.index.name = "year"
    summary["avg_rows_per_day"] = (summary["rows"] / summary["trading_days"]).round(1)
    return summary


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build(dedupe: bool, exclude_2023_h1: bool, coverage: bool = True):
    """Read, validate and clean.

    Returns (frame, report, flagged outliers, coverage table). Coverage is
    measured BEFORE the 2023 exclusion -- it is the evidence for that
    exclusion, so measuring it afterwards would only show the hole we cut.
    """
    report = BuildReport()

    df = to_utc(pd.concat([read_year(y, report) for y in YEARS], ignore_index=True))
    report.rows_raw = len(df)

    check_ohlc(df, report)
    dup_mask = check_duplicates(df, report)
    outliers = check_outliers(df, report)

    if dedupe and report.duplicate_rows:
        if not report.duplicates_ohlc_identical:
            raise ValueError(
                "Refusing to deduplicate: some repeated timestamps carry "
                "DIFFERENT OHLC, so 'keep first' would be an arbitrary choice "
                "between two real quotes. Inspect them with --keep-duplicates."
            )
        df = df.loc[~dup_mask].reset_index(drop=True)
        report.duplicates_dropped = True

    cov = coverage_by_month(df) if coverage else None

    if exclude_2023_h1:
        ny = df["timestamp_ny"]
        window = (ny >= EXCLUDE_START) & (ny < EXCLUDE_END)
        report.excluded_rows = int(window.sum())
        report.exclusion_applied = True
        df = df.loc[~window].reset_index(drop=True)

    report.rows_out = len(df)
    report.first_utc = str(df["timestamp_utc"].iloc[0])
    report.last_utc = str(df["timestamp_utc"].iloc[-1])
    return df, report, outliers, cov


def write_parquet(df: pd.DataFrame, report: BuildReport, path: Path) -> None:
    """Write with the build report embedded in the parquet key-value metadata."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = {**(table.schema.metadata or {}), **report.to_metadata()}
    pq.write_table(table.replace_schema_metadata(metadata), path, compression="zstd")


def read_build_metadata(path: Path = OUTPUT_PATH) -> dict:
    """Read back the build report a parquet was written with ({} if absent)."""
    raw = (pq.read_schema(path).metadata or {}).get(b"gbpusd_m1")
    return json.loads(raw) if raw else {}


def print_report(df: pd.DataFrame, report: BuildReport, outliers: pd.DataFrame,
                 cov: pd.DataFrame | None) -> None:
    rule = "=" * 74
    print(rule)
    print("GBP/USD M1 BUILD")
    print(rule)
    print("rows read   {:,}   {}".format(
        report.rows_raw,
        "  ".join(f"{y}:{n:,}" for y, n in report.rows_per_file.items())))
    print(f"rows out    {report.rows_out:,}")
    print(f"span        {report.first_utc}  ..  {report.last_utc}  (UTC)")
    print()

    print("INTEGRITY")
    for name, count in report.ohlc_violations.items():
        print(f"  {name:<26s} {count:,}" + ("" if count == 0 else "   <-- BAD"))
    print(f"  {'nonpositive prices':<26s} {report.nonpositive_rows:,}")
    print(f"  {'non-finite prices':<26s} {report.nonfinite_rows:,}")
    print(f"  {'nonzero volume':<26s} {report.nonzero_volume_rows:,}"
          "   (expected 0: the forex feed carries no volume)")
    frozen_pct = 100 * report.frozen_bars / max(report.rows_raw, 1)
    print(f"  {'frozen bars (high==low)':<26s} {report.frozen_bars:,} "
          f"({frozen_pct:.2f}%)   quiet minutes, not errors, but they bias")
    print(f"  {'':<26s} {'':<12s}   realized vol downward -- see var-ratio test")
    print()

    print("DUPLICATE TIMESTAMPS")
    disposition = ("dropped" if report.duplicates_dropped
                   else "KEPT (--keep-duplicates was passed)")
    print(f"  {report.duplicate_rows:,} redundant rows across "
          f"{report.duplicate_timestamps:,} timestamps -- {disposition}")
    print(f"  every copy carries identical OHLC: {report.duplicates_ohlc_identical}")
    print()

    print("PRICE JUMPS")
    print(f"  {report.outlier_returns:,} consecutive-minute moves beyond "
          f"{report.outlier_threshold_pips:.1f} pips "
          f"({OUTLIER_MAD_MULTIPLE:.0f}x the median absolute minute move).")
    print("  Flagged, never dropped: these read as news prints rather than bad")
    print("  ticks unless one spikes and fully retraces in the following minute.")
    if not outliers.empty:
        show = outliers.head(8)[["timestamp_ny", *PRICE_COLS, "jump_pips"]]
        print(show.to_string(index=False))
    print()

    if cov is not None:
        print("COVERAGE   observed minutes / minutes the Sun 17:00-Fri 17:00 NY "
              "week implies")
        print("  measured BEFORE any exclusion, so it stands as the evidence for one.")
        print(f"  median month {cov['coverage_pct'].median():.1f}% -- the shortfall "
              "in a normal month is holidays.")
        print("  twelve worst months:")
        print(cov.nsmallest(12, "coverage_pct").to_string())
        print()

    if report.exclusion_applied:
        print(f"EXCLUDED    {report.excluded_rows:,} rows in {EXCLUDE_START.date()} "
              f".. {(EXCLUDE_END - pd.Timedelta(days=1)).date()} (New York local)")
        print()

    print("PER CALENDAR YEAR (New York local dates)")
    print(yearly_summary(df).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--keep-duplicates", action="store_true",
        help="Keep the last-Sunday-of-October repeated hour. Default: drop it.")
    parser.add_argument(
        "--keep-2023-h1", action="store_true",
        help="Keep Feb 1 - Jul 31 2023, where the feed's minute coverage is "
             "degraded. Default: exclude it.")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Run every check and print the report without writing the parquet.")
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="Skip the per-month coverage table (the slowest part of the report).")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH,
                        help=f"Output parquet path (default: {OUTPUT_PATH.name}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, report, outliers, cov = build(dedupe=not args.keep_duplicates,
                                      exclude_2023_h1=not args.keep_2023_h1,
                                      coverage=not args.no_coverage)
    print_report(df, report, outliers, cov)

    if args.report_only:
        print("\n--report-only: nothing written.")
        return

    write_parquet(df, report, args.out)
    print(f"\nSaved to {args.out}  (build report embedded in parquet metadata)")


if __name__ == "__main__":
    main()
