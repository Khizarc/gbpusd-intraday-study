# GBP/USD Intraday Microstructure Study

A statistical study of six years of one-minute GBP/USD data (2020–2025), aimed
squarely at the daytrading horizon: every position opens and closes inside a
single New York session, and nothing is ever held overnight.

The project has two halves. The first is a **data pipeline** that turns raw
vendor CSVs into a validated, provenance-stamped parquet and refuses to hand
you a file that was built wrong. The second is **six independent studies** plus
a **validation suite** that checks the statistics themselves against synthetic
data with known answers.

The headline finding, which every study reaches from a different direction:

> **The size of GBP/USD's intraday moves is highly structured and genuinely
> forecastable. The direction is not.**

That is not a disappointing result. It says precisely where effort is worth
spending — position sizing, stop placement, session timing, risk budgeting —
and where it is not.

---

## Table of contents

- [Quick start](#quick-start)
- [Getting the data](#getting-the-data)
- [Repository layout](#repository-layout)
- [The data pipeline](#the-data-pipeline)
- [The six studies](#the-six-studies)
  - [Test 1 — Variance ratio](#test-1--variance-ratio-trend-or-mean-reversion)
  - [Test 2 — Volatility](#test-2--volatility-shape-and-level)
  - [Test 3 — Opening range breakout](#test-3--opening-range-breakout)
  - [Test 4 — Jumps](#test-4--jumps-versus-diffusion)
  - [Test 5 — Day anatomy](#test-5--the-anatomy-of-a-day)
  - [Test 6 — Data-snooping audit](#test-6--data-snooping-audit)
- [Validating the statistics](#validating-the-statistics)
- [Findings, collected](#findings-collected)
- [What this means in practice](#what-this-means-in-practice)
- [Methodology notes](#methodology-notes)
- [Known limitations](#known-limitations)
- [Conventions](#conventions)

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

# Place the HistData CSVs in gbpusdData/ (see "Getting the data"), then:
python load_data.py             # build + validate the parquet
python test_estimators.py       # confirm the statistics are correct (~4 min)

# The studies, in any order:
python test_variance_ratio.py   # ~6 min  (--no-bootstrap for ~20s)
python test_vol_forecast.py     # ~1 min
python test_orb.py              # ~2 min
python test_jumps.py            # ~1 min
python test_day_anatomy.py      # ~1 min
python test_snooping.py         # ~1 min
```

Every script is read-only against the parquet, takes `--help`, and prints a
self-contained report with its own interpretation guide at the bottom.

## Getting the data

The price data is **not in this repository** — it is vendor data, it is not
mine to redistribute, and it is 114 MB. Download it yourself:

1. Go to [histdata.com](https://www.histdata.com/download-free-forex-data/),
   choose **ASCII / M1 Bars**, instrument **GBPUSD**, years **2020–2025**.
2. Unzip each archive into `gbpusdData/`, so you have:

   ```
   gbpusdData/DAT_ASCII_GBPUSD_M1_2020.csv
   gbpusdData/DAT_ASCII_GBPUSD_M1_2021.csv
   ...
   gbpusdData/DAT_ASCII_GBPUSD_M1_2025.csv
   ```
3. Run `python load_data.py`.

The parquet is a build artefact and is gitignored — it is fully reproducible
from the CSVs in about 40 seconds.

## Repository layout

| File | Role |
| --- | --- |
| `load_data.py` | Builds and validates `gbpusd_m1.parquet` from the yearly CSVs. |
| `fxlib.py` | Shared loading, session labelling, returns, and statistics. |
| `test_estimators.py` | **Validation suite.** Checks every estimator against synthetic data. |
| `test_variance_ratio.py` | **Test 1** — trend vs mean reversion at 2–60 minutes. |
| `test_vol_forecast.py` | **Test 2** — volatility's fixed shape and forecastable level. |
| `test_orb.py` | **Test 3** — opening range breakout, put on trial. |
| `test_jumps.py` | **Test 4** — jump vs diffusion decomposition, and the news clock. |
| `test_day_anatomy.py` | **Test 5** — when highs and lows form; trend days vs chop. |
| `test_snooping.py` | **Test 6** — multiple-testing audit of every directional pattern. |
| `hourly_profile.py` | Hourly range and net-move profile on the New York clock. |
| `sweep_analysis.py` | Liquidity-sweep study against Asian and London session levels. |

`fxlib.py` holds everything shared: the provenance-checking loader, session-day
labelling that survives DST, gap-aware return construction, the time-of-day
volatility profile, Newey–West OLS, and the day-block bootstrap. Only pandas,
numpy and pyarrow are required — every estimator is written out rather than
imported, which is why `test_estimators.py` exists.

---

## The data pipeline

### `load_data.py`

Reads six yearly CSVs (~2.18 M rows), validates them, cleans them, and writes a
single parquet with its own build report embedded in the file metadata.

```bash
python load_data.py                 # clean build — this is what everything reads
python load_data.py --report-only   # run every check, write nothing
python load_data.py --keep-2023-h1  # opt out of an exclusion, deliberately
```

**Timezone handling.** HistData's timestamps are a *fixed* UTC−5 offset
year-round — the feed never applies daylight saving. They are localized as a
plain fixed offset and converted to UTC, then a second column re-expresses the
same instants on the real New York wall clock with true DST rules. Session
boundaries are defined on that second column. Treating the source as
`America/New_York` (which observes DST) or as `Etc/GMT+5` (whose POSIX sign
convention means the opposite of what it reads) are the two standard ways to
put every session boundary an hour out for half the year.

**What it verifies on every build, rather than asserting in a comment:**

| Check | Result on this dataset |
| --- | --- |
| OHLC ordering (`high ≥ low`, brackets open/close) | 0 violations |
| Non-positive or non-finite prices | 0 |
| Volume column really is all zero | confirmed (forex feed carries none) |
| Frozen bars (`high == low`) | 25,946 (1.19%) — reported, not removed |
| Consecutive-minute jumps beyond 40× the median | 206 — flagged, not removed |
| Per-month coverage vs the Sun 17:00–Fri 17:00 NY week | median month 99.3% |

**What it cleans, by default, with the evidence printed each time:**

- **360 duplicate rows.** On the last Sunday of October each year, the feed
  emits the 19:00–19:59 source-local hour twice. The copies are verified
  byte-identical *before* either is dropped — and if that ever stops being
  true, the build **refuses to deduplicate** rather than arbitrarily picking
  between two real quotes.
- **February–July 2023.** Minute coverage collapses to 67–72% against the FX
  trading week, versus a 99.3% median month. The coverage table is computed
  *before* the exclusion, so it stands as the evidence for it rather than just
  showing the hole that was cut.

**What it deliberately does not clean.** The 206 flagged price jumps turn out
to be real events — the September 2022 mini-budget crash tops the list at
−161 pips — not bad ticks. Frozen bars are legitimate quiet minutes. Both are
surfaced with counts because they matter downstream (frozen bars bias every
volatility estimate downward), but removing them would be fabricating data.

**Provenance.** The build report is written into the parquet's key-value
metadata, and `fxlib.load_m1()` reads it back and **raises** if the file was
built with cleaning disabled. Rebuilding with the wrong flags is a loud failure
rather than a quiet wrong answer six scripts downstream.

---

## The six studies

### Test 1 — Variance ratio: trend or mean reversion?

**Question.** Over the 2-to-60-minute horizons a day trade lives on, do moves
persist (trade breakouts) or fade (trade reversion)? Does the answer change
with the session clock?

**Method.** The Lo–MacKinlay (1988) variance ratio,
`VR(q) = Var[q-minute return] / (q · Var[1-minute return])`. Under a random
walk VR = 1 at every horizon; above 1 means momentum, below 1 means reversion.
Reported with a heteroskedasticity-robust z-statistic and a **day-block
bootstrap** confidence interval — the CI is what decides the verdict, because
overlapping windows make the analytic z optimistic.

Three specific hazards are handled: q-minute windows never span a feed gap
(a weekend repricing counted as a one-minute move inflates VR at every horizon
at once); the bootstrap resamples whole days, not observations; and a
**volatility signature plot** separates real reversion from bid-ask bounce.

**Findings.**

| q (min) | VR | 95% bootstrap CI | Verdict |
| --- | --- | --- | --- |
| 2 | 0.9874 | [0.9831, 0.9919] | mean-reverting |
| 3 | 0.9805 | [0.9741, 0.9868] | mean-reverting |
| 5 | 0.9713 | [0.9623, 0.9798] | mean-reverting |
| 10 | 0.9587 | [0.9448, 0.9705] | mean-reverting |
| 15 | 0.9582 | [0.9421, 0.9723] | mean-reverting |
| 30 | 0.9685 | [0.9454, 0.9866] | mean-reverting |
| 60 | 0.9906 | [0.9626, 1.0162] | random walk |

<sub>500 bootstrap replicates, days resampled as blocks.</sub>

- Statistically real, economically trivial. The strongest reversion is **4% of
  variance** at 10–15 minutes. Against a 1-pip spread there is nothing to
  collect.
- **It does not survive the split.** On 2024–2025 alone every horizon is a
  random walk, and q=60 flips positive.
- The volatility signature is **flat** (RV at 1-minute sampling is 1.005× RV at
  60-minute), so this is *not* bid-ask bounce. Microstructure noise in this
  feed is negligible. The reversion is genuine and simply tiny.
- By hour, the extremes are **02:00 NY trending** (VR₃₀ = 1.36) and
  **17:00–18:00 reverting hard** (0.61–0.64). The latter is the daily rollover
  and is an illiquidity artefact — Test 4 identifies the same hours
  independently.

### Test 2 — Volatility: shape and level

**Question.** How big is today going to be, and how much of that is knowable
before the London open?

**Method.** Two things that are routinely conflated get measured separately.

- **Shape** — a volatility multiplier per 15-minute bucket of the New York
  clock, built from mean squared return (variance is what scales) and reported
  as its square root (a stop is a distance). Validated **out of sample**:
  fitted on 2020–2023, applied to 2024–2025.
- **Level** — a HAR regression (Corsi 2009) of log realized volatility on
  daily, weekly and monthly lags plus the just-closed Asian session. Every
  regressor completes before 03:00 NY on the day it predicts. Newey–West
  standard errors throughout, scored on an untouched 2024–2025 holdout with
  out-of-sample R² measured against the *in-sample* mean.

**Findings.**

- **The intraday shape spans 3.9×.** 09:30 NY (the equity open) is **1.92×**
  the average minute; 08:30 (US data) is 1.74×; the quietest buckets around
  00:15 and 19:30 are ~0.50×. A stop quoted in fixed pips means something
  completely different at 01:00 than at 09:30.
- **The shape is stable out of sample.** Applying the 2020–2023 profile to
  2024–2025, **70% of the 96 buckets land within 10%** of their fitted shape.
  The worst is 18:00 (the rollover again), off by 50%. Separately, 2024–2025 is
  ~32% calmer overall — a level shift, not a shape failure.
- **The level is genuinely forecastable:**

  | Model | OOS R² | IS R² |
  | --- | --- | --- |
  | constant only | 0.000 | 0.000 |
  | yesterday only | 0.528 | 0.609 |
  | Asia only | 0.535 | 0.591 |
  | HAR (day + week + month) | 0.632 | 0.674 |
  | **HAR + Asia** | **0.654** | 0.700 |

  The Asian term is highly significant (t = 7.85) but adds only **+0.022 OOS
  R²** over HAR alone — most of what it tells you, persistence already told
  you. Honest reporting of that gap is the point of the table.
- **The sizing table** turns this into something usable: Asian-range decile →
  median London/NY range, with p25/p75. Bottom decile: 14–20 pip Asian range →
  ~62–77 pip session. Top decile: 67–84 → ~89–135. The `day/asia` ratio falls
  monotonically from ~4.0 to ~1.4, so the relationship is strongly
  mean-reverting in ratio terms.

### Test 3 — Opening range breakout

**Question.** The most-traded intraday pattern there is: mark the first N
minutes after a session opens, then trade the break. Does it work?

**Method.** London (03:00) and New York (08:00) opens; ranges of 15, 30 and 60
minutes; entry as a stop order at the range edge; stop at the opposite edge;
risk `R` = the range height (so results are volatility-normalised); 2R target;
flat by 16:00.

Four things stop it flattering itself:

- **Costs are charged** — 1 pip round trip by default (`--cost-pips`). At a
  20-pip range that is 5% of R, about the size of the effect being looked for.
- **Same-bar ties count as losses.** If one minute spans both target and stop,
  the feed cannot say which came first; assuming the good one is how a mediocre
  rule is made to look excellent.
- **A coin-flip control** with identical timing, stop and target geometry. A 2R
  rule that wins 40% is profitable, so a raw win rate says nothing alone.
- **The mirror rule** — fading the break — is scored alongside.

**Findings.**

**Null, comprehensively.** Across both sessions and all three range lengths,
breakout ≈ fade ≈ coin flip, all clustered slightly negative at roughly the
cost charged.

| Session | OR | Breakout (R) | Fade (R) | Coin flip (R) |
| --- | --- | --- | --- | --- |
| London | 15 | −0.093 | −0.068 | −0.053 |
| London | 30 | −0.021 | −0.080 | −0.048 |
| London | 60 | −0.016 | −0.069 | −0.063 |
| New York | 15 | −0.077 | −0.093 | −0.121 |
| New York | 30 | −0.017 | −0.087 | −0.060 |
| New York | 60 | −0.076 | −0.042 | −0.032 |

Not one breakout cell clears its own coin-flip benchmark, and nothing is stable
across the 2020–23 / 2024–25 split. The range edge carries no information; the
pattern is selecting for volatility, not direction.

### Test 4 — Jumps versus diffusion

**Question.** "Volatility" covers two things a stop order does not treat
alike: price grinding continuously through every level, versus price
relocating in one minute with nothing traded in between. Which is this?

**Method.** Two estimators doing different jobs.

- **Barndorff-Nielsen & Shephard**, daily. Realized variance counts
  everything; bipower variation (products of *adjacent* absolute returns) is
  immune to isolated jumps. The gap is the jump component; their ratio
  statistic, standardised by tripower quarticity, says whether a given day's
  gap exceeds sampling noise.
- **Lee & Mykland**, per minute. Score each return against a *local* bipower
  estimate of normal built from the 270 minutes strictly before it. With ~2 M
  minutes the null distribution of the maximum is Gumbel, which sets the
  threshold at **7.38 local sigmas** — the stated α is the chance of one false
  positive across the entire sample, not per minute.

Returns are **deseasonalized first** using Test 2's profile. Skipping that step
makes the detector rediscover the time-of-day curve and call every New York
open a jump (it finds 483 extra "jumps" on raw returns).

**Findings.**

- **7.3% of daily variance arrives as jumps**, and **30.7% of days** carry a
  statistically significant one. Jump days have similar bipower variation to
  quiet days but higher realized variance — the jumps are genuinely extra
  movement, not an artefact of busy days being easier to test.
- **The detector reconstructs the economic calendar from prices alone.** No
  calendar was used as input. The busiest minutes of the day for jumps are
  **09:30** (NY equity open), **08:30** (US data), **03:00** (London open),
  **11:00**, **04:30** (UK data), **10:00**, **05:30**.
- **A large artefact, correctly quarantined.** 37% of raw detections land in
  the **17:00–18:59 rollover**, where liquidity leaves at the FX day boundary
  and a single tick moves price several pips against a shape curve saying
  minutes there should be tiny. These are illiquidity, not news. Excluding them
  changes the conclusion of the next point entirely.
- **Nothing tradable follows a jump.** Session-window jumps drift
  **−0.19 pips** over the next 30 minutes, 95% CI **[−1.35, +1.09]** — a clean
  null. (Rollover artefacts "revert" at a 26% continuation rate, which is what
  a spurious tick unwinding looks like, and which would have produced a
  confidently wrong "fade the spike" conclusion if left in.)
- Median jump size 7.9 pips, p90 21.8, max 161.2 (the mini-budget crash).

**Practical reading:** this is a risk result, not an opportunity. 7% of
variance arrives in minutes where a stop does not fill at its level. The
correct response to the 08:30/09:30 cluster is to be *smaller* into it, not to
trade it.

### Test 5 — The anatomy of a day

**Question.** What does a session actually look like from the inside? When do
the extremes form, how much of the range is left by lunchtime, and is any of it
knowable while the day is still running?

**Method.** Descriptive throughout except 5D, which is the only part making a
claim and the only part carrying confidence intervals (day-block bootstraps).
Measurements over 03:00–15:59 NY: range, path length, efficiency, and close
location value.

**Findings.**

- **~20% of sessions put both the high AND the low in during the 03:00 hour**
  — the London open dominates extreme formation. A second, smaller cluster
  appears at 15:00.
- **Half the day's full range is set by 12:00**, 70% by 14:00, and only 7% by
  08:00. On average 67% of the eventual range is already established by the end
  of the 08:00 hour. This is the measurable cost of waiting for confirmation.
- **Trend days are 45% of sessions** (net move ≥ 50% of range); chop days
  (≤ 20%) are 22%. Sessions close in the top or bottom 20% of their range 45%
  of the time.
- **5D is the cleanest result in the project.** Split at 09:00 —
  - direction: `corr(morning efficiency, afternoon efficiency) = +0.011`,
    CI [−0.042, +0.064]. `P(afternoon continues morning) = 0.495`,
    CI [0.470, 0.521]. Conditioning on a *decisive* morning does not help
    (0.513, CI [0.459, 0.569]).
  - size: `corr(morning range, afternoon range) = +0.511`.

  The same answer as Tests 1 and 3, from a completely different angle.

**A methodological note worth reading.** The Kaufman efficiency ratio
(`|net| / path`) is **not scale-free** and is routinely quoted as though it
were. Path length grows roughly linearly as you sample faster while net move
does not, so the ratio falls toward zero by arithmetic alone:

| sampling | 1 min | 5 min | 15 min | 30 min | 60 min |
| --- | --- | --- | --- | --- | --- |
| median efficiency | 0.032 | 0.072 | 0.123 | 0.180 | 0.264 |

Quoting "efficiency = 0.04" without stating the interval says nothing. The
trend/chop split therefore uses `|net| / range`, which is scale-free.

### Test 6 — Data-snooping audit

**Question.** Tests 1–5 examined a lot of cells: 24 hours × 3 horizons, 6
session/range combinations, hour and weekday tables. Some looked interesting.
Does *any* of it beat what a search of that size produces by accident?

**Method.** A **pre-specified** family of 28 intraday directional rules — hour
drift, weekday, session carryover, morning-to-afternoon momentum, and
volatility-regime-conditioned variants — each producing one gross-pip number
per trading day, all flat by 16:00. **Three of them are seeded coin flips**,
included so the reader can watch pure noise earn a respectable t-statistic.
Then three escalating tests:

1. **Individual t-statistics** — what gets reported when a rule is presented as
   though it were the only thing anyone tried.
2. **Holm–Bonferroni** — controls family-wise error with no assumption about
   how rules relate. Valid but conservative here, since most of these rules are
   secretly the same trade.
3. **White's Reality Check (2000)** — resample **days jointly across every
   rule**, preserving their correlation, and build the null distribution of the
   *best* t-statistic in a family this size. The observed best is read against
   that.

Cost is applied **after** testing, as a hurdle. Subtracting a fixed cost and
then t-testing would make every rule "significantly negative" — it would be
confirming that one pip exceeds zero.

**Findings.**

- 2 of 28 rules are significant at p < 0.05 individually; **1.4 are expected by
  chance**. A literal coin flip in the family earned **|t| = 1.24**.
- **0 rules survive Holm–Bonferroni.**
- **Reality Check: nothing survives.** Best observed |t| = **2.75**; the median
  best |t| under the null is **2.23** and the 95th percentile is **3.13**.
  **p = 0.15.**
- The one nuance kept: **short 07:00–08:00 NY** replicates at −1.21 and −1.06
  pips/day across both periods and clears the 1-pip cost. It failed the family
  test, so it is not evidence — it is the only item worth a fresh holdout.

---

## Validating the statistics

Every estimator here is hand-rolled in numpy rather than imported from a tested
library. Any one could carry a sign slip or a normalisation error and still
produce entirely reasonable-looking numbers on real data, because on real data
there is nothing to check them against.

```bash
python test_estimators.py            # ~4 min, exits non-zero on failure
python test_estimators.py --group "variance ratio"
```

**18 checks, in two flavours:**

- **Calibration** — feed the estimator data with *no* effect and confirm it
  says so, at the advertised rate. A test rejecting 20% of the time at a
  nominal 5% is broken, and every null result in this repo would be worthless.
- **Power** — feed it a *known* effect of known size and confirm it finds it at
  roughly the right magnitude. This half matters most here, because the
  headline findings of Tests 1, 3, 5D and 6 are all null results, and **a null
  result from a blind estimator is not evidence of absence.**

Selected results:

| Check | Result |
| --- | --- |
| VR on a random walk, 12 walks × 100k, vs each horizon's own standard error | q=2 0.9992, q=60 1.0063, all \|z\| < 1 |
| VR on AR(1), φ = ±0.20, vs the closed form `1 + 2Σ(1−k/q)φᵏ` | max error 0.008 |
| Robust z rejection rate at nominal 5% | 0.027 over 300 trials |
| Newey–West vs a hand-computed lag-1 sandwich | max difference 2.8e−17 |
| 95% CI coverage with AR(1) errors: Newey–West vs plain OLS | **0.917** vs 0.785 |
| Lee–Mykland on 40 clean 30k-minute runs | 1 detection total (2.5% of runs) |
| Lee–Mykland recovering injected 12σ jumps | **11/11**, 0 false positives |
| BNS on a continuous path | jump share 0.011, rejects 1.0% at nominal 1% |
| BNS on an injected jump | detected 200/200; share 0.399 vs 0.445 predicted |
| Day-block bootstrap coverage vs iid resampling | **0.935** vs 0.470 |
| Reality Check on a family of pure noise | rejects 1.7% at nominal 5% |
| Reality Check finding one planted rule among 20 | identified **90%** of the time |

That bootstrap row is worth dwelling on: resampling *observations* instead of
*days* gives 47% coverage on a nominal 95% interval. Every confidence interval
in this project would have been roughly twice as narrow as it should be.

**The suite earned its keep during development** — it caught two tolerance
errors in its own checks, one of which revealed that a "does the shape fit?"
test was tautological (rescaling returns by a profile fitted on those same
returns returns 1.000 by construction). That check is now fitted on one half of
the days and evaluated on the other. The same tautology had been present in
Test 2 and was fixed there too.

---

## Findings, collected

| Question | Answer | Where |
| --- | --- | --- |
| Does price trend or revert intraday? | Neither, tradably. VR ≈ 0.96 at 10–15 min, dies out of sample | Test 1 |
| Is short-horizon reversion just bid-ask bounce? | No — the volatility signature is flat. It is real and tiny | Test 1 |
| Is volatility time-of-day structured? | Yes, **3.9×** peak-to-trough; 09:30 is 1.92× the average minute | Test 2 |
| Is that shape stable? | Yes — 70% of buckets within 10% out of sample | Test 2 |
| Is today's volatility forecastable? | Yes — **OOS R² = 0.65** | Test 2 |
| Do opening range breakouts work? | No — indistinguishable from a coin flip in every cell | Test 3 |
| Is movement smooth or jumpy? | **7.3%** of variance is jumps, on **30.7%** of days | Test 4 |
| When does news hit? | 09:30, 08:30, 03:00, 11:00, 04:30 — recovered from prices alone | Test 4 |
| Can you trade a jump after it prints? | No — 30-min drift CI [−1.35, +1.09] pips | Test 4 |
| When is the day's range set? | Half by **12:00**; ~20% of extremes form in the 03:00 hour | Test 5 |
| How often is it a trend day? | **45%** capture ≥ 50% of their range | Test 5 |
| Does the morning predict the afternoon? | Direction **+0.011**. Size **+0.511** | Test 5 |
| Does any directional pattern survive multiple testing? | **No** — Reality Check p = 0.15 | Test 6 |

## What this means in practice

Everything actionable is about size and timing, not direction:

- **Scale every stop and target by the time-of-day multiplier.** A fixed pip
  stop is 3.9× tighter in effective terms at 09:30 than at 01:00. This is the
  single highest-value result here and it is a near-certainty, not a forecast.
- **Plan the day's range from the Asian session**, using the decile table. The
  p25/p75 columns say how wrong that plan is routinely allowed to be.
- **Know that half the range is gone by noon.** A setup that only becomes
  obvious at 13:00 is competing for the remaining ~5%.
- **Be smaller into 08:30 and 09:30**, where 7% of the day's variance arrives
  in minutes that do not fill at your stop level.
- **Do not hold through the 17:00–18:59 rollover.** It is the worst spread of
  the day and the detected "jumps" there are the market not functioning.
- **Do not pay for directional signals in this instrument at this horizon.**
  Six independent tests, one of them explicitly designed to find any survivor
  among 28 candidates, found nothing.

## Methodology notes

Applied consistently across every study:

- **Fixed split.** In-sample 2020–2023, holdout 2024–2025, chosen once and
  identical everywhere. Every result is reported on both halves.
- **Day-block bootstrap.** Intraday observations within a day are heavily
  dependent and overlapping windows are dependent by construction, so the
  resampling unit is always the whole trading day. The validation suite
  demonstrates why: observation-level resampling gives 47% coverage.
- **Newey–West everywhere it matters.** Realized volatility is strongly
  autocorrelated; plain OLS standard errors on it are meaningless.
- **Gap awareness.** No multi-minute window ever spans a weekend, holiday or
  feed outage. A repricing counted as a one-minute move contaminates every
  horizon simultaneously.
- **Costs charged, ties resolved against the trade.** Same-bar target-and-stop
  is a loss; costs are explicit and tunable.
- **Benchmarks, not zero.** A strategy result is compared against a matched
  random control, never merely against breakeven.
- **Deseasonalize before comparing magnitudes.** Any test that compares a move
  against "normal" must divide out the time-of-day curve first, or it will
  rediscover the curve and call it a signal.

## Known limitations

Stated plainly, because they bound what the findings support:

- **One instrument, one venue, six years.** Nothing here is claimed to
  generalise to other pairs, other periods, or other data vendors.
- **No bid-ask data.** HistData provides mid-ish bar OHLC with no spread.
  Costs are modelled as a flat parametrised pip charge, not measured. Real
  spreads widen precisely during the jump minutes Test 4 identifies, so the
  true cost of trading news is understated here.
- **No order book, no volume.** Forex has no consolidated tape; the volume
  column is identically zero. Every liquidity inference is made from price
  behaviour alone.
- **The 2023 exclusion is slightly conservative.** February 2023 has 89.2%
  coverage, clearly better than the 67–72% of March–July. The window starts
  1 February anyway; a case could be made for starting 1 March.
- **`sweep_analysis.py` predates the audit framework** and has not been put
  through Test 6's multiple-testing treatment. Its hour-by-hour deltas should
  be read with that in mind.
- **Test 6's family is not exhaustive.** It covers 28 rules of a particular
  shape. A rule outside that shape is untested, not disproven — though the
  consistency of the null across five other studies makes a large hidden
  directional effect unlikely.
- **Survivorship of the method, not the data.** Everything here was written
  while looking at this dataset. The `07:00–08:00` result, the only survivor of
  any kind, needs data none of this has touched before it means anything.

## Conventions

- **All session logic runs on the New York wall clock** with true DST rules.
  The source feed's fixed UTC−5 timestamps are converted, never reinterpreted.
- **1 pip = 0.0001.** Quotes carry 6 decimal places, so sub-pip precision is
  real and is preserved.
- **The daytrading window is 03:00–15:59 NY** (London open through the New York
  afternoon). Nothing is held past it.
- **The Asian reference window is 19:00–02:59 NY**, rolled so an evening
  belongs to the following session day.
- **Sunday's thin reopen is excluded** from anything hour-of-day shaped.
- **17:00–18:59 NY is treated as a microstructure regime, not a trading
  window.** Test 4 quarantines it explicitly.

---

## License

The **code** in this repository is MIT licensed — see [LICENSE](LICENSE).

That licence covers the code only. It does **not** cover the GBP/USD price
data the code analyses: that is supplied by
[HistData.com](https://www.histdata.com/) under their own terms, is not
included in this repository, and is not redistributed by it. See
[Getting the data](#getting-the-data).
