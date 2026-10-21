# Solar Swarm v11 — varied events and uncertainty-driven satellite motion

Put this file in the repo as `docs/v11_plan.md`. Work through the phases in order and stop at each
**Checkpoint** to show the result before going on.

## Why this is needed

In the v10 replay two things are effectively fixed:

1. **Warning time.** `buildEvent` uses a fixed ambient wind (w = 400 km/s) and a fixed drag
   (γ = 2×10⁻⁸ km⁻¹). The only input that varies is the CME-speed select (3 values), so the gain is
   always 22.7, 20.4 or 18.1 min. The same DBM gives 12–32 min for a 1180 km/s CME once w and γ vary,
   and S1's halo-orbit position adds another ±1–2 min when the shock is tilted.
2. **S5 motion.** The decision uses a fixed λ prior (9,000 km ×/÷1.65), so U is the same every time
   and S5 almost always trims 3,000 → 1,800 km, in a single decision made at T+0.8 h.

The goal: every replay is a different, realistic event. The forecast does not know the truth. The
satellites react to the forecast's uncertainty as it changes.

## Rules for the whole job

- Build `ui/solar-swarm-v11.html` from v10. Keep v10 unchanged. LIVE mode must keep working.
- Everything not from data is labelled *assumed* in the UI and listed in the ⓘ panel.
- Each event comes from a seed, and the same seed gives the same event (use the existing `mulberry32`).
- Split `buildEvent` into a **physics part** (trajectories, contacts, forecast, planner; no DOM, no
  particles) and a **scene part** (particles, field lines). The Monte Carlo batch uses only the physics part.
- Frame time stays under ~8 ms (measure as before). Update the README and the ⓘ panel.
- Run `python -m unittest discover tests` after every Python change.

---

## Phase 0 — finish the real-data run (blocking)

`ui/results.js` in the repo is still the 2026-10-14 "partial" file, so the Validation tab shows no
real numbers.

1. Check whether `python run_all.py all --start 2012-01-01` finished. If not, rerun
   `python run_all.py analyze` (the downloads are cached) and fix any errors.
2. Show me the end of the log, `results/summary.json`, and the first 20 lines of `results/events.csv`.
3. Commit the regenerated `ui/results.js`. `results/` and `data/` stay git-ignored.

**Checkpoint 0:** report the numbers before going on.

---

## Phase 1 — export event priors from the pipeline (`ssw/priors.py`, `run_all.py priors`)

Write `ui/event_priors.js` (`window.SS_PRIORS = {...}`). `analyze` calls it at the end; it can also be
run alone as `python run_all.py priors`.

| Key | Source | Fallback if no data (label *assumed*) |
|---|---|---|
| `v0` | log-normal fit to `v0` of plausible events | median 800 km/s, σ_ln 0.40, clip 450–3000 |
| `half_angle` | empirical quantiles | uniform 25–60° |
| `lon` | empirical quantiles | N(0, 15°), clipped so \|lon\| < half-angle |
| `w` | `w_obs` (real 24-h ambient wind before launch) | N(430, 90) km/s, clip 300–750 |
| `gamma` | log-normal fit to per-event `g_star` (train set, `validate_dbm`) | median 2e-8, σ_ln 0.7 |
| `gamma_m1` | calibrated M1 γ | 2e-8 |
| `bz_sheath` | **new:** min Bz in OMNI from shock to +6 h (add `bz_min_sheath` in `events.add_insitu`) | N(−6, 7) nT |
| `tau_c` | per-event λ / v_shock from `structure_scale` (D = 30 min) | median 15 s, σ_ln 0.5 (reproduces today's 9,000 km at 600 km/s) |
| `historical` | up to 40 events from `events.csv`: `cme_id, t21, v0, half_angle, lon, lat, w_obs, v_shock, bz_min_sheath, t_obs, enlil_arr` | empty list |

Add a test: `priors` runs on the mock data and the fitted medians are close to the planted values.

---

## Phase 2 — truth, observation and forecast (v11 physics part)

### 2a. Hidden truth, sampled from the seed

`v0, w, γ, half-angle, lon, shock tilt θ ~ N(0°, 25°)` (clip ±60°), `φ ~ U(0, 360°)`, `sheath Bz`.
Keep only events that hit Earth (`|lon| < half-angle`), have a 0.8–5 day transit, and form a shock
at L1 (fast-mode Mach > 1.2 with `shockJump`). Otherwise draw again with the next seed. The UI states
"Earth-impacting events only".

### 2b. Halo orbits (assumed amplitudes)

- **S1 around sub-L1:** `y = 120,000·sin(φ₁)`, `z = 80,000·cos(φ₁)` km, random phase.
- **L1 reference monitor (DSCOVR-like):** `y = 250,000·sin(φ₂)`, `z = 150,000·cos(φ₂)` km, random phase.
- **S2–S4:** keep the tetrahedron, but let the whole formation drift along its 180-day halo during
  the event (a few thousand km, visible in the swarm view).
- The gain is `t_contact(L1 monitor at its real position) − t_contact(S1 at its real position)`,
  both from the tilted planar shock. Not `(tL1 − tSub)` on the axis.

### 2c. Observation (what the forecaster sees)

`v0_obs = v0·(1 + 0.10·N)`, `half_obs = half + 5°·N`, `w_obs = w + 40·N` (all assumed, labelled).

### 2d. Forecast with data assimilation

- Use an ensemble of 200 members: v0 from the v0_obs error, w from the w_obs error, γ from the
  `gamma` prior. Each member gives an arrival time and a shock speed at L1.
- Every 6 h, from T+6 h until the front reaches 0.7 AU, there is a heliospheric-imager measurement of
  the front distance, `r_obs = r_true + 5 %·r·N`. Label it "assumed: Vigil-like L5 imager".
  Re-weight the members with a Gaussian likelihood (a particle filter). Resample when the effective
  sample size drops below 50 %.
- Store a forecast history: at each update time keep the ETA mean and σ, the shock speed at L1
  (mean and σ), and the expected gain (mean and σ).
- After S1 contact: in-situ update as now (3 % speed error), then the sigma for the remaining time.

---

## Phase 3 — re-planning S5 from uncertainty (replaces the single decision)

- **Decision times:** T+0.8 h, then at each imager update, then every 6 h, with the last one at
  `t_S1 − 3 h`.
- **λ prior per decision:** `λ_med = V̂_L1 · τ_c,med` and `σ_ln = sqrt(σ_ln,τ² + (σ_V / V̂)²)`.
  Use the existing `Eeps` with these values instead of the global `LAMP`/`LSIG`.
- **Candidates:** 12 log-spaced separations from 500 to 15,000 km, plus "hold".
- **Δv of a new plan:** first cancel the current drift velocity, then a two-burn transfer arriving at
  `t_S1 − 2 h`:
  `Δv = |v_current| + 4·|L_target − L_now| / T_remaining`.
  Infeasible if it exceeds the remaining budget (40 m/s per event, assumed).
- **Cost:** `J = U + w_Δv·Δv`, with `w_Δv = 0.004` as now. Hysteresis: move only if
  `U_now − U_new > 0.03` **and** `J_new < J_hold`.
- **Kinematics:** piecewise bang-bang between plans, so the position and speed in the scout inset
  stay continuous.
- **Log every decision** with its reason, for example: `+12h: V̂(L1) 640 ± 70 km/s → λ ≈ 9,600 km
  (×/÷1.6); U(3,000) 0.31 > 0.25 → retarget 2,400 km, Δv 6.2 m/s, 27.4 m/s left`, or
  `hold: gain 0.01 < 0.03`.
- **Scout inset:** draw the L(t) track with a dot for each burn, the target ring, and a Δv-budget bar.
- **Motion tab:** a table of all decisions.
- **Honesty line (UI and README):** "Only S5 can react in hours; S1–S4 follow their halo orbits."

---

## Phase 4 — UI for events

- Replace the three footer selects with:
  `Event: [Random ▾ | Historical ▾ | Custom]`, a seed box, and a **New event** button.
  "Custom" shows the old CME speed / tilt / Bz selects.
- URL `?event=1234` loads seed 1234. `?event=hist:<cme_id>` loads a historical event.
- **Historical events:** the truth comes from the real event (`v0`, half-angle, lon, `w_obs`, γ chosen
  so the DBM arrival matches `t_obs`, `bz_min_sheath`). Show the real observed arrival and the real
  Enlil arrival next to the model's forecast. If `SS_PRIORS.historical` is empty, the option says
  "run the data pipeline first".
- **Status card "Forecast vs truth":** the ETA ± σ line as it narrows over time (small chart from the
  forecast history). The truth stays hidden until arrival, then its error is shown.
- **Warning time card:** before S1 contact show "expected +19 ± 4 min" from the forecast; at S1
  contact show the measured value.
- The mission log comes from the event (decisions, imager updates, contacts).

**Checkpoint 2:** screenshots of 3 different seeds plus one historical event, with the gain, the
number of S5 burns and the total Δv for each.

---

## Phase 5 — Monte Carlo statistics

- In a new **Statistics** tab, a "Run 1,000 events" button runs the physics part only, in chunks or a
  Web Worker so the UI stays responsive, in under ~10 s.
- **Histograms:**
  - warning-time gain
  - ETA error at T+12 h, at T+24 h and after S1 contact
  - S5 total Δv
  - number of burns
  - % of events where S5 moved
- **Plus:** the median and the 10–90 % range for each.
- Add `ssw/montecarlo.py`, the same model in Python for the report (same priors file), saving
  `results/montecarlo.json`. Add a test that the JS and Python medians agree within 5 % for the same
  priors (run the JS through Node).

---

## Phase 6 — tests and checks

- For tilt 0 and both craft on the axis, the gain ≈ 750,000 km / V_shock(L1) within 3 %.
- The same seed gives an identical event (gain, contact times and decisions).
- The particle filter: σ_ETA never grows after an imager update on average, and the truth falls
  inside ±2σ in about 95 % of 1,000 events. Report the coverage.
- The Δv budget is never exceeded, and S5 position and velocity are continuous.
- A historical event reproduces its observed arrival within 1 min (γ is solved for).
- Frame time < 8 ms, and the batch of 1,000 events < 10 s.

**Final checkpoint:**
- the Monte Carlo summary (gain median and 10–90 %, ETA-error improvement, % of S5 moves, mean Δv)
- the coverage number
- screenshots

If real λ values make S5 always go to the largest or smallest separation, say so plainly. That is a
finding, not a bug to hide.
