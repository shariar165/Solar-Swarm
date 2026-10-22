# SOLAR SWARM — real-data upgrade

The pipeline fixes the five weaknesses found in the capstone review. Each fix writes
its numbers to `results/`, and the UI (`ui/solar-swarm-v11.html`) reads them from
`ui/results.js` and shows them in a new **Validation** tab. Since v11 every replay is a
different, realistic event drawn from priors fitted to the same data (`ui/event_priors.js`);
see [v11](#v11--every-replay-is-a-different-event).

| # | Weakness | Fix | Module |
|---|----------|-----|--------|
| 1 | Warning gain is S5's idea, not ours | Gain measured on **every real CME-driven shock** (real post-shock speeds), plus how much the arrival forecast tightens after S1 contact | `ssw/lead_time.py` |
| 2 | Optimizer runs on made-up values | Structure size λ **measured in real shock sheaths** (1-s/3-s magnetometer); Δv weight swept to show the decision is (or is not) robust | `ssw/structure_scale.py`, `ssw/planner.py` |
| 3 | 5th satellite weakly justified | Exact reliability calculation: 4 vs 5 (scout) vs 5 (spare) craft, swept over failure rates | `ssw/reliability.py` |
| 4 | No validation | DBM hindcast on DONKI CME→Earth pairs, chronological split, vs NASA's WSA-Enlil, bootstrap confidence intervals | `ssw/validate_dbm.py` |
| 5 | ML too light | Gradient-boosted shock detector trained on years of 1-min L1 data with DONKI labels, compared to the old rule | `ssw/shock_detector.py` |

## Run it (needs internet; ~1–2 h for the first download)

```bash
pip install -r requirements.txt
python -m unittest discover tests        # 19 tests, ~1 min (the first run builds mock data; Node.js for the JS checks)
python run_all.py probe                   # checks dataset/parameter names on CDAWeb
python run_all.py all --start 2012-01-01  # download + every analysis
python check_feeds.py                     # checks the LIVE-mode feeds (no key needed)
python run_all.py priors                  # rebuild only ui/event_priors.js from results/*.csv
python run_all.py montecarlo              # 1,000 v11 replays -> results/montecarlo.json (~30 s)
```

Then open **`ui/solar-swarm-v11.html`** (the main UI) in a browser and look at the **Validation**,
**Statistics** and **Motion** tabs; the LIVE button switches it to the real feeds.
The earlier versions v5–v10 are kept for comparison in `ui/archive/` (unchanged apart from the path
to `ui/results.js`).
Downloads are cached in `data/raw/`, so `python run_all.py analyze` re-runs only the analyses.

DONKI (CMEs, shocks, Enlil runs) comes from CCMC at `https://ccmc.gsfc.nasa.gov/DONKI-API/get/`,
for both the pipeline and the UI's LIVE mode; no API key is needed. CCMC moved it there from
`kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/` on 2026-09-30. Enlil runs come only with the `CME`
endpoint (inside `cmeAnalyses`), not `CMEAnalysis`. If `probe` says a parameter is MISSING,
edit `ssw/config.py`.

## What to check after the first real run

- `results/summary.json` — every number shown in the UI.
- `results/events.csv` — the CME→shock pairs. Spot-check a few against DONKI.
- `results/dbm_test_predictions.csv` — per-event errors on the test years.
- `results/priors.json` (= `ui/event_priors.js`) — the event priors v11 draws from, each with its source.
- `results/structure_per_event.csv`, `results/dbm_train_gstar.csv` — the per-event τ<sub>c</sub>, λ and γ behind them.
- `results/montecarlo.json` — the v11 replay statistics.

## Test mode (not results)

`python run_all.py mock` runs everything on synthetic data with known answers
(planted γ ≈ 6×10⁻⁸ km⁻¹ → calibration finds 5×10⁻⁸; planted λ = 60,000 km → method
finds ~47,000 km). Open `ui/solar-swarm-v11.html?mock` to see the UI with those numbers;
a red banner marks them as mock.

## Honest limits

- Observed arrival = DONKI interplanetary shock linked to the CME. DONKI's links and
  shock list are incomplete; some pairs may be wrong, some shocks missing.
- Only CMEs that did arrive are scored (no miss/false-alarm skill for arrival forecasts).
- Sentinel gain assumes the shock keeps its L1-measured speed over the last 2.2 million km.
- The real-gain statistics use the shock speed from mass-flux conservation (OMNI density and
  speed, medians 25 min either side of the jump) and only the **240 of 430 events with a clean
  density jump (n2/n1 ≥ 1.2)**, so weak shocks are under-represented. With the post-shock plasma
  speed instead, the gain was overstated (29.2 → 26.1 min median).
- λ is the correlation length of sheath fluctuations, used as a proxy for structure size.
- Failure rates in the reliability table are assumptions; the table sweeps them.
- The five satellites are still a simulation; everything labelled REAL comes from
  spacecraft that exist (ACE, Wind, DSCOVR, GOES) and NASA catalogues.
- DONKI writes 999 for an unknown CME direction; those values are treated as missing.
- The drag model can reproduce only 274 of 430 real arrivals (64 %) with the DONKI speed and
  the measured wind (the rest arrive slower than the wind allows or faster than no drag
  allows). The 40 historical replays are chosen among those 274.
- v11's forecast is scored inside its own model: the truth is drawn from the same population the
  filter assumes, so its ~95 % coverage says the filter is consistent, not that real forecasts
  would be that well calibrated. Its error *size* is tied to reality by calibration only: the v0
  error and a model-error term were chosen so the first forecast's mean error equals the real
  drag-model test error (13.9 h); see "Calibrated to the real test error" below.
- The (v0, γ, w<sub>eff</sub>) data triples come from the 86 of 240 clean shocks (36 %) that the drag
  model can fit to both arrival time and shock speed, and they lean toward fast CMEs (median v0
  926 km/s, against 594 km/s for the events that do not solve). The Monte Carlo transit times
  still match the real clean-shock events (see below).

## v11 — every replay is a different event

**Only S5 can react in hours; S1–S4 follow their halo orbits.**

`ui/ss-physics-v11.js` is the physics part (no DOM; the page, the Statistics tab and the Node
tests all use it); `ssw/montecarlo.py` is a line-by-line Python port that gives the same events
for the same seeds (bit-exact random streams). One event:

1. **Hidden truth** from a seed: (v0, γ, w<sub>eff</sub>) resampled from the drag-model triples solved on
   real training-year events (each triple reproduces that event's L1 arrival time *and* its
   mass-flux shock speed), with jitter v0 ±5 %, γ ×/÷1.2, w<sub>eff</sub> ±20 km/s plus the calibrated
   model-error term (±110 km/s); half-angle, longitude, sheath Bz and
   sheath correlation time τ<sub>c</sub> from the event priors; shock tilt θ ~ N(0°, 25°) (clip ±60°),
   azimuth uniform, random halo phases. Earth-impacting events only: a draw that misses Earth
   (|lon| ≥ half-angle), takes outside 0.8–5 days or forms no shock at L1 (Mach ≤ 1.2) is redrawn.
2. **Halo orbits** (assumed): S1 120,000 × 80,000 km around sub-L1, the L1 reference monitor
   (DSCOVR-like) 250,000 × 150,000 km, the S2–S5 formation centre 50,000 × 30,000 km, 180-day
   period. The warning gain is contact(L1 monitor) − contact(S1) at their real positions with the
   tilted shock.
3. **What the forecaster sees**: v0 ×/÷ e<sup>0.35</sup> (log-normal, **calibrated to the real test
   error**), half-angle ±5°, the measured upstream wind ±40 km/s (assumed).
4. **Forecast**: a 200-member particle filter on (v0, w, γ). The members are drawn from the same
   data triples as the truth (joint prior, with the same jitter and model-error term) and weighted
   by the likelihood of the observed v0; then re-weighted every 6 h by an L5
   heliospheric imager (assumed, Vigil-like, 5 % range error) until the front reaches 0.7 AU;
   systematic resampling below 50 % effective sample size with Liu–West jitter. After S1 contact,
   the in-situ speed (3 % error) gives the remaining time.
5. **S5 re-planning**: decisions at T+0.8 h, at each imager update, every 6 h after that, and a last
   one 3 h before the forecast S1 contact. λ prior = V̂(L1)·τ<sub>c</sub> with
   σ<sub>ln</sub> = √(σ<sub>τ</sub>² + (σ<sub>V</sub>/V̂)²); 14 separations from 500 to 30,000 km (log-spaced, ×1.37) plus hold;
   J = U + 0.004·Δv; move only if U drops by more than 0.03 and J beats holding. A new plan first
   stops S5 at its current acceleration, then flies a bang-bang transfer, so position and speed stay
   continuous; Δv = |v| + 4·|ΔL|/T, within 40 m/s per event (assumed). The transfer must end
   2 h + 2σ before the *forecast* S1 contact, because the true contact time is hidden.

**UI.** `Event: Random | Historical | Custom`, a seed box and *New event* in the footer;
`?event=1234` loads a seed, `?event=hist:<cme_id>` a real event (γ solved so the L1 monitor sees the
shock at the real time; the real Enlil forecast is shown next to the model's). Status shows the
expected warning gain until S1 contact, then the measured one, and a *Forecast vs truth* chart
whose truth appears only at arrival. The scout inset draws the planned L(t) with its burns and the
Δv budget; the Motion tab lists every decision. The timelines reveal the true event only as it
happens. Everything assumed is listed in the ⓘ panel.

**Monte Carlo** (1,000 events, real priors; Statistics tab or `python run_all.py montecarlo`):

| | Monte Carlo, median (10–90 %) | real data |
|---|---|---|
| Warning-time gain (halo orbits, tilted shock) | 24.3 min (16.0–36.7) | — |
| Warning-time gain on the Sun–Earth line | 24.5 min (16.6–36.0) | 26.1 min (17.7–33.2), 240 clean shocks |
| Shock speed at L1 | 510 km/s (347–755) | 479 km/s (376–708) |
| Transit 21.5 R☉ → L1 | 65.0 h (44.1–97.2) | 64.8 h (44.9–93.9) |
| ETA error, first forecast (T+0.3 h): mean · median | **13.9 h · 12.3 h** | 13.9 h · 11.4 h (M1 drag model, 129 test events) |
| ETA error at T+12 h / T+24 h / last before S1 contact | 6.0 / 3.8 / 2.7 h | — |
| ETA error after S1 contact | 1.7 min (0.3–4.5) | — |
| Truth inside ±2σ: first guess · all updates · last | 96.4 % · 94.7 % · 94.6 % | — |
| S5 home separation (training-year priors) · moved | 8,220 km · 0 % (3,000 km start: 58.0 %) | — |
| S5 expected U before contact · below 0.25 | 0.206 (0.196–0.255) · 87.8 % | — |

**Calibrated to the real test error.** The real M1 drag model misses the arrival by 13.9 h on
average on the test years (mean absolute error; median 11.4 h). The Monte Carlo is matched on the
same statistic, the mean |ETA error| of its first forecast (`node tests/js/calibrate_errors.js` →
`results/calibration_errors.json`):

1. *Joint prior first.* Forecast members now resample the same (v0, γ, w<sub>eff</sub>) triples as the
   truth, weighted by the likelihood of the observed v0, instead of independent v0/γ/w priors with
   w taken from an observation (which had let the forecaster see w<sub>eff</sub>, a quantity no
   instrument measures). Coverage inside ±2σ went from 93.6 % to 96.3 % (all updates) at the old
   10 % v0 error; the mean first-forecast error rose from ~6 h to 7.9 h.
2. *v0 error, capped at 35 %.* Mean error 7.9 h (10 %) → 9.1 h (20 %) → 10.4 h (35 %). It saturates
   because a poor observation sends the forecaster back to the population prior.
3. *Model-error term for the rest.* Extra w<sub>eff</sub> spread around each triple, in the truth and in
   the members alike (so the forecaster knows its size), bisected to **±110 km/s** → mean 13.85 h,
   median 12.3 h. A γ spread works too (×/÷e<sup>1.6</sup>) but widens the shock-speed tail more
   (90th percentile 793 km/s against 755). Adding the spread to the members only does *not* raise
   the error (10.5 h); it only widens σ (99.6 % inside ±2σ).

Sources of the first forecast's squared error (Shapley average over both orders): unknown γ and
w<sub>eff</sub> within the data triples 26 %, the v0 observation error 27 %, the model-error term 47 %.
The model term is as large as the whole triple population's w<sub>eff</sub> spread, a measure of how
much the real arrivals scatter beyond what a drag model fitted to clean shocks explains.

*Side effect — shock-speed drift.* The ±110 km/s w<sub>eff</sub> spread is symmetric, but draws with a
low wind push the transit past 5 days and are redrawn (events needing a redraw: 269 → 317 of
1,000), so the accepted truth's mean w<sub>eff</sub> rises by 17 km/s (469 → 486). The Monte Carlo
shock speed at L1 drifts from 490 to **510 km/s** median against the real **479 km/s**
(240 clean shocks). The warning gain scales as separation / V<sub>sh</sub>, so the on-axis gain is
**24.5 min** against the real **26.1 min** (−6 %): the Monte Carlo gain is slightly pessimistic.
The calibration targets the ETA error, not the gain.

**Finding — 52 events reach L1 faster than both their DONKI speed and their own shock speed.**
A drag-model speed falls (or rises) monotonically from v0 to V<sub>sh</sub>, so the mean Sun–L1 speed
must lie between them. For 52 of the 240 clean shocks it is higher than both (median v0 435 km/s,
V<sub>sh</sub> 442 km/s, mean speed 581 km/s); no γ or w of any size fits them. They are most likely
DONKI speeds underestimated by projection, or a shock paired with the wrong CME. Their IDs are in
`results/inconsistent_events.csv`. The other unsolved events: 35 slower than both, 51 with no
drag-model solution, 16 that need w or γ outside 250–800 km/s and 1e-10–1e-5 km⁻¹
(`results/dbm_triples_all.csv` has every event and its reason).

**S5 home separation — computed from the priors** (`node tests/js/s5_home.js` → `results/s5_home.json`;
`homeSep()` in the physics, `home_sep()` in the Python port). S5 now waits between storms at the
separation that minimises the expected error U over the whole event population *as the forecaster
sees it before an event*: λ = V<sub>L1</sub> · τ<sub>c</sub>, with V<sub>L1</sub> from the forecaster's joint prior
(1,000 draws of the real data triples with their jitter and model-error term, accepted like a
member) and τ<sub>c</sub> from training-year sheaths (median 149 s, σ<sub>ln</sub> 0.28, n = 287), not from any
event's truth. Everything behind it is training-year data (before 2024-05-14).

- Optimum **8,220 km**, expected U 0.22 (at 3,000 km: 0.47). U is flat: it stays within +0.02 of the
  minimum from **6,390 to 10,400 km**. Prior λ 52,000–113,000 km (10–90 %).
- Cross-checks (not used by the model): from each Monte Carlo event's first forecast 8,520 km; from
  the true λ of each event 8,110 km; from the *measured* λ of the 287 training-year sheaths (no drag
  model) **7,130 km** (flat 5,500–8,960). The drag-model optimum sits higher because its shock
  speeds are faster than the measured ones (the drift above). This replaces the rough "~11,000 km"
  quoted earlier; the planner's candidates nearest the optimum are 6,212 and 8,511 km.
- *Reaching home (assumed between storms).* The one-time move 3,000 → 8,220 km (5,220 km) costs
  2·ΔL/T with two impulses and a coast: 1.34 m/s in 3 months, **0.67 m/s in 6 months**; Δv < 1 m/s
  needs ≥ 121 days (the planner's continuous bang-bang profile costs twice that). Afterwards S5
  never leaves home during an event (below), so there is no return trip — which matters, because
  real CME-driven shocks at L1 are only 6.5 days apart (median; 10 % under 1 day; `results/events.csv`).
  Free-space estimate: the natural relative dynamics near L1 are outside the model.

| 1,000 events | home 3,000 km (v10) | home 8,220 km (priors) | 8,220 km, 20 m/s | 3,000 km, 20 m/s |
|---|---|---|---|---|
| Expected U at home, first forecast | 0.470 (0.405–0.539) | 0.214 (0.209–0.226) | same | same as 40 m/s |
| **Expected U before contact**, median (10–90 %) | 0.413 (0.280–0.615) | **0.206 (0.196–0.255)** | 0.206 (0.196–0.255) | 0.413 (0.292–0.615) |
| Events with U < 0.25 before contact | 3.3 % | **87.8 %** | 87.8 % | 1.6 % |
| Realised error vs true λ at contact, median · < 0.25 | 0.391 · 14.5 % | 0.195 · 79.3 % | same | 0.394 · 13.0 % |
| S5 moved | 58.0 % | **0 %** | 0 % | 58.0 % |
| inward · outward | 0 · 580 | 0 · 0 | 0 · 0 | 0 · 580 |
| burns per event (0 / 1 / 2) | 420 / 573 / 7 | 1,000 / 0 / 0 | 1,000 / 0 / 0 | 420 / 580 / 0 |
| Δv mean · max | 6.0 · 32.7 m/s | 0 · 0 | 0 · 0 | 5.0 · 16.5 m/s |

(`results/montecarlo.json`, `montecarlo_L3000.json`, `montecarlo_budget20.json`,
`montecarlo_L3000_budget20.json`; `python run_all.py montecarlo [--home 3000] [--budget 20] [--out …]`.)

**Finding — at its home separation S5 never moves during an event, and the event-time budget no
longer matters.** Starting from 8,220 km, S5 holds in all 1,000 events at 20, 40, 80, 150 or 300 m/s.
The event-time re-planner is still in the code and was checked at every one of the 11,870 decision
times: U was at or below the 0.25 limit at 10,876 (92 %); at the other 994 the best affordable move
gained less than the 0.03 hysteresis (620) or nothing was reachable in the budget and time left
(374) — holding was always cheaper. U at home is already near its floor (~0.2: magnetometer noise
dominates on short baselines, curvature on long ones), so it exceeds 0.25 only for unusual events —
very slow (seed 1958: V̂<sub>L1</sub> 253 km/s, λ̂ 37,000 km, U 0.39) or very fast (seed 1295: V̂<sub>L1</sub>
1,070 km/s, λ̂ 157,000 km, U 0.34). The candidates that would help them (4,534 or 15,979 km) are
≥ 3,500 km away; in the ~1–2 days before the transfer deadline that costs > 100 m/s, and with
J = U + 0.004·Δv no such move beats holding even when the budget allows it. With Δv free (weight 0)
and 40 m/s, 17 of 1,000 events would move, all inward. (The planner's log names the nearest
candidate, 8,511 km, as "best": its U gain is below the hysteresis.) Choosing where S5 waits does
the job the event-time re-planning was trying to do: the share of events with U < 0.25 before
contact rises from 3 % to 88 %.

**How robust is the home?** Sweeping the magnetometer noise ratio SN (0.3 nT on 4 nT = 0.075) and
τ<sub>c</sub> (×0.5, ×1, ×2 the training-year median); each cell: optimal home · range where U stays
within +0.02 of the minimum · U at home (Motion tab, `results/s5_home.json`):

| SN \ τ<sub>c</sub> | 75 s (×0.5) | 149 s (×1) | 298 s (×2) |
|---|---|---|---|
| 0.03 | 3,030 km · 2,120–4,120 · 0.12 | 6,060 km · 4,250–8,270 · 0.12 | 12,120 km · 8,440–16,440 · 0.12 |
| **0.075** (used) | 4,110 km · 3,180–5,180 · 0.22 | **8,220 km · 6,390–10,400 · 0.22** | 16,450 km · 12,690–20,660 · 0.22 |
| 0.15 | 5,190 km · 4,250–6,260 · 0.35 | 10,380 km · 8,440–12,560 · 0.35 | 20,770 km · 16,930–24,960 · 0.35 |

The home is a fixed fraction of λ, so it scales linearly with τ<sub>c</sub>, and as SN<sup>1/3</sup> with the
noise (the minimum of ε balances the noise term ∝ SN/L against the curvature term ∝ L²). U at home
depends on the noise only: a quieter magnetometer helps more than any choice of separation. With
τ<sub>c</sub> doubled the optimum doubles too (16,450 km) — every optimum and flat range now lies inside
the separation grid, which reaches 30,000 km. In every case 3,000 km is only
optimal for a short τ<sub>c</sub> and a very quiet magnetometer.

**Uncertainty-driven motion on the slow time scale — the rolling home** (`results/home_rolling.json`,
Motion tab). Every month the home is recomputed from the *measured* λ = V<sub>shock</sub> · τ<sub>c</sub> of
the last 20 real CME-driven shocks before that date (402 dated sheaths, 2012–2026); a historical
replay starts S5 at the home for its date and shows the change since the previous month and its Δv
(2·|ΔL| per month, two impulses). The Monte Carlo keeps the training-year population home above.

- Home 7,340 km median (6,620–7,850 km, 10–90 %; range 5,870–8,330 km), October 2012 to October 2026.
- Monthly change 90 km median (no change in 27 % of months); **Δv 0.07 m/s per month median**
  (54 % of months under 0.1 m/s), mean 0.14, max 0.86 m/s; about **1.7 m/s per year**. The expected
  "< 0.1 m/s per month" holds for a typical month, not for the busiest ones.
- The solar cycle shows most in *how often* the home moves: 20 shocks span 110–170 days near
  maximum (2022–2025) but 700–1,060 days near minimum (2018–2020), and the Δv to follow the home is
  2.5–3.8 m/s per year at maximum (2014, 2023, 2024) against 0.5–0.7 m/s at minimum (2018–2020).
  The value itself moves less: about 15 % higher at minimum (7,750–7,850 km in 2018–19) than in
  the active years (6,500–7,000 km in 2014–15 and 2022–23), because τ<sub>c</sub> is longer at minimum
  (160–166 s against 132–148 s) while the median shock speed hardly changes (400–450 km/s).
- It sits below the Monte Carlo's population home (8,220 km) for the reason given in the
  cross-checks: measured shock speeds are slower than the drag-model ones.

**Halo station-keeping (assumed).** All five craft trim their halo orbits every 21 ± 2 days. S2–S5 use
0.1–0.5 m/s per trim (≈ 5.2 m/s per craft per year; assumed, labelled in the UI); S1 is a solar sail and
holds station with sail attitude, not propellant, so its trims cost no Δv. This is bookkeeping only —
the trims are what keeps each craft on the halo it is drawn on — with its own random streams, so no
other number changes. The replay shows each trim as a dot under the speed timeline (S1: an open ring, "sail attitude
trim", also marked on S1's halo track in the swarm view) and as a ring on S5's L(t) track, and the Motion tab keeps a running Δv per craft since 1 January (real date for a
historical event, a seeded day otherwise). Compared with the 1.7 m/s per year for the rolling home, routine
station-keeping is the larger cost of S5's motion.

**For comparison — the old 3,000 km start (kept in `results/montecarlo_L3000.json`).** From 3,000 km,
S5 only ever widens the baseline: final separations 3,000 km (42 %, held), 3,309 km (58 %, the next
candidate up, 309 km away) or 4,534 km (4 %), never smaller. Most of these moves are the cheap one-step
hop; they lift U before contact only from 0.47 to 0.41 (median), because 3,309 km is still far below
the optimum. The budget barely matters there (moved in 58.0 % of events at 20 m/s and at 40 m/s; at
20 m/s the second step to 4,534 km is no longer affordable). With the assumed fallback
τ<sub>c</sub> = 15 s (λ ≈ 9,000 km) the same planner trims inward instead (S5 moves in 98 % of events,
mostly to 2,415 or 1,762 km). These are results of the data, not a tuning.

**Frame time.** Median `render(P)` over a whole replay in Chrome on the development laptop:
v11 7–13 ms depending on the event (fast CMEs draw more plasma parcels), v10 8–12 ms on the same
machine; the ~8 ms target is not reliably met by either. The physics runs once per event
(~2–4 ms), not per frame. A batch of 1,000 events takes ~6.5 s in the browser, ~4 s in Node.

## Animation (introduced in v10, `ui/archive/solar-swarm-v10.html`; v11 builds on it)

Everything that moves is computed:

- **CME**: closed-form drag-based model (exact, no integration step); cone model with the
  real half-angle and direction (from DONKI in LIVE mode); sheath ≈ 0.12 r and magnetic
  flux rope ≈ 0.26 r behind the shock (typical sizes at 1 AU).
  Drawn as an ice-cream-cone outline with the classic three-part look (bright front, dark
  cavity, bright core). Before 21.5 R☉ the front is back-extrapolated at constant speed to
  the onset (as in the LASCO/CDAW catalogue), so the replay opens with the flare and the
  eruption. The coiled flux-rope strands and the coronal-wave rings are schematic.
  Rendered as optically thin white light (Thomson scattering): it adds light instead of hiding
  what is behind it, with a warm tint for the sheath and a pale violet tint for the flux rope.
  It fades as it expands (∝ R^-0.5, floor 0.62; really ∝ R⁻², compressed so it stays
  visible). The rope's front is slightly lumpy (a few %, schematic). The shock envelope stays
  smooth, because its nose is the drag-model position the swarm view is built on.
- **Sun and sky**: limb-darkened Sun at true size in a K-corona (brightness ∝ r^-2.5 on a
  log-like display scale; streamer-belt pattern schematic), zodiacal dust glow, and stars with
  a realistic brightness distribution and colours plus a faint Milky Way (not a real star map).
- **Solar wind**: Parker spiral field lines for the actual wind speed, rotating with the Sun
  (25.38 days). Plasma moves at its true speed for the time compression shown, drawn as motion
  streaks (length = distance covered in 15 min in the Sun–Earth view, 25 s in the swarm view).
- **Shock crossing the swarm**: planar shock with a selectable tilt; contact times per craft
  to the second; the 4-spacecraft timing method recovers the shock normal and speed.
- **Rankine–Hugoniot**: compression ratio from the fast-magnetosonic Mach number; sheath
  plasma and magnetic field lines are compressed and bent accordingly. The sheath brightness
  follows the compression. Sheath field lines are turbulent: three octaves with Kolmogorov
  scaling δB/B ∝ λ^⅓ (0.4 at 1 million km, level assumed), frozen into the moving plasma.
  The shock is a thin line (it is thinner than a pixel at 3,900 km/px).
- **Foreshock**: θBn (upstream Parker field vs the 3D shock normal) is computed. When it is
  below ~45° (quasi-parallel), ULF-wave ripples on the upstream field lines and a faint glow
  extend ~1 million km ahead of the shock (depth and amplitude schematic). With the default
  25° tilt θBn ≈ 71° (no foreshock); pick shock tilt −25° to see one (θBn ≈ 24°).
- **Earth**: magnetopause (Shue et al. 1998) and bow shock (Farris & Russell 1994), at true
  scale in the swarm view with the magnetosheath shaded and magnetotail lobes (schematic), and
  magnified in the inset; shows when GEO satellites are exposed. Planets show their night side.
- **Mercury and Venus**: on their orbits, enlarged ~100× like Earth, moving at their true speed
  relative to Earth. In the replay both sit inside the CME cone, so the storm hits Mercury, then
  Venus, then Earth; in LIVE mode they are at today's real positions (JPL approximate elements)
  and the real DONKI cone decides whether they are hit. Arrival at each planet comes from the
  same cone + drag model; the wind there is scaled from 1 AU (n ∝ r⁻², Parker field).
- **Magnetic shields inset** (why Earth is safe): Mercury | Venus | Earth side by side, same scale in
  planet radii, Sun to the left, north up. Earth: Shue magnetopause with dipole field lines, GEO,
  aurora. Mercury: 195 nT offset dipole; magnetopause from pressure balance, scaled so Earth matches
  Shue (gives 1.48 R☿ in normal wind, MESSENGER mean 1.45); a fast CME sheath pushes it below the
  ground and the wind hits the surface. Venus: no global field; the ionosphere (~300 → 250 km,
  assumed) is the only obstacle, with draped field and an O⁺ tail that doubles in CMEs (×1.9,
  Edberg et al. 2011). Field-line shapes are schematic; Mercury's core induction is ignored.
- **3D inset**: drag to rotate; shock plane passing through the tetrahedron, true vs
  timing-derived normal.
- **Camera**: the Sun–Earth view and the swarm view differ by a ~38× zoom. The change of view is
  one continuous camera zoom into L1 (both layers share km per pixel and keep L1 at the same
  screen point at every step), with the detail resolving from the centre. The stars stay fixed.
- Scenario controls: CME speed, shock tilt (−25°, 0°, 25°, 45°), sheath Bz.
- Layout: the scene fills the stage at any window shape (the viewBox follows the stage, with
  content drawn into a bleed margin). The ⛶ button or the F key toggles fullscreen. Technical
  notes are in the ⓘ panel (or the I key) instead of over the scene. In the swarm view the
  magnetic-shields inset moves to the bottom left so it never covers the swarm or Earth.
  v5–v9 are kept for comparison.
