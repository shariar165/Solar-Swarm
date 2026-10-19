# SOLAR SWARM — real-data upgrade

The pipeline fixes the five weaknesses found in the capstone review. Each fix writes
its numbers to `results/`, and the UI (`ui/solar-swarm-v10.html`) reads them from
`ui/results.js` and shows them in a new **Validation** tab.

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
python -m unittest discover tests        # 8 tests, a few seconds
python run_all.py probe                   # checks dataset/parameter names on CDAWeb
python run_all.py all --start 2012-01-01  # download + every analysis
python check_feeds.py                     # checks the LIVE-mode feeds (no key needed)
```

Then open `ui/solar-swarm-v10.html` (the main UI) in a browser and look at the **Validation** tab.
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
