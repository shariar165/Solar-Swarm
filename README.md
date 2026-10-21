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

Then open `ui/solar-swarm-v11.html` (the main UI) in a browser and look at the **Validation** and
**Statistics** tabs. v5–v10 are kept unchanged for comparison.
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
finds ~47,000 km). Open `ui/solar-swarm-v10.html?mock` to see the UI with those numbers;
a red banner marks them as mock.

## Honest limits

- Observed arrival = DONKI interplanetary shock linked to the CME. DONKI's links and
  shock list are incomplete; some pairs may be wrong, some shocks missing.
- Only CMEs that did arrive are scored (no miss/false-alarm skill for arrival forecasts).
- Sentinel gain assumes the shock keeps its L1-measured speed over the last 2.2 million km.
- λ is the correlation length of sheath fluctuations, used as a proxy for structure size.
- Failure rates in the reliability table are assumptions; the table sweeps them.
- The five satellites are still a simulation; everything labelled REAL comes from
  spacecraft that exist (ACE, Wind, DSCOVR, GOES) and NASA catalogues.
- DONKI writes 999 for an unknown CME direction; those values are treated as missing.
- The drag model can reproduce only 274 of 430 real arrivals (64 %) with the DONKI speed and
  the measured wind (the rest arrive slower than the wind allows or faster than no drag
  allows). The 40 historical replays are chosen among those 274.
- v11's forecast is scored inside its own model: the truth is drawn from the same priors the
  filter assumes, so its ~95 % coverage says the filter is consistent, not that real forecasts
  would be that well calibrated (the real DBM test error is ~13 h, see Validation).

## v11 — every replay is a different event

**Only S5 can react in hours; S1–S4 follow their halo orbits.**

`ui/ss-physics-v11.js` is the physics part (no DOM; the page, the Statistics tab and the Node
tests all use it); `ssw/montecarlo.py` is a line-by-line Python port that gives the same events
for the same seeds (bit-exact random streams). One event:

1. **Hidden truth** from a seed: v0, ambient wind w, drag γ, half-angle, longitude, sheath Bz and
   sheath correlation time τ<sub>c</sub> from the event priors; shock tilt θ ~ N(0°, 25°) (clip ±60°),
   azimuth uniform, random halo phases. Earth-impacting events only: a draw that misses Earth
   (|lon| ≥ half-angle), takes outside 0.8–5 days or forms no shock at L1 (Mach ≤ 1.2) is redrawn.
2. **Halo orbits** (assumed): S1 120,000 × 80,000 km around sub-L1, the L1 reference monitor
   (DSCOVR-like) 250,000 × 150,000 km, the S2–S5 formation centre 50,000 × 30,000 km, 180-day
   period. The warning gain is contact(L1 monitor) − contact(S1) at their real positions with the
   tilted shock.
3. **What the forecaster sees** (assumed): v0 ±10 %, half-angle ±5°, wind ±40 km/s.
4. **Forecast**: a 200-member particle filter on (v0, w, γ), re-weighted every 6 h by an L5
   heliospheric imager (assumed, Vigil-like, 5 % range error) until the front reaches 0.7 AU;
   systematic resampling below 50 % effective sample size with Liu–West jitter. After S1 contact,
   the in-situ speed (3 % error) gives the remaining time.
5. **S5 re-planning**: decisions at T+0.8 h, at each imager update, every 6 h after that, and a last
   one 3 h before the forecast S1 contact. λ prior = V̂(L1)·τ<sub>c</sub> with
   σ<sub>ln</sub> = √(σ<sub>τ</sub>² + (σ<sub>V</sub>/V̂)²); 12 separations from 500 to 15,000 km plus hold;
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

| | median | 10–90 % |
|---|---|---|
| Warning-time gain | 20.0 min | 13.2–26.9 min |
| ETA error at T+0.3 h / T+12 h / T+24 h | 4.9 / 2.9 / 2.1 h | — |
| ETA error after S1 contact | 1.3 min | 0.2–3.5 min |
| Truth inside ±2σ (all forecast updates) | 95.6 % | — |
| S5 moved / mean Δv | 79.1 % / 14.2 m/s | — |

**Finding — S5 only ever widens the baseline, by one step.** With τ<sub>c</sub> ≈ 147 s measured in
real sheaths, λ is 65,000–160,000 km, so the error U is lowest near 11,000 km, far above the
3,000 km start. The 40 m/s budget reaches only the next candidate up: final separations are
3,196 km (45 %), 4,355 km (34 %) or 3,000 km (21 %, held), never smaller. The first decision is the
cheapest, so later re-plans almost always hold. With the assumed fallback τ<sub>c</sub> = 15 s
(λ ≈ 9,000 km) the same planner trims inward instead (S5 moves in 98 % of events, mostly to 1,722 or 2,346 km). This is a result of the data, not a tuning.

**Frame time.** Median `render(P)` over a whole replay in Chrome on the development laptop:
v11 7–13 ms depending on the event (fast CMEs draw more plasma parcels), v10 8–12 ms on the same
machine; the ~8 ms target is not reliably met by either. The physics runs once per event
(~2–4 ms), not per frame. A batch of 1,000 events takes ~6.5 s in the browser, ~4 s in Node.

## Animation (ui/solar-swarm-v10.html)

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
