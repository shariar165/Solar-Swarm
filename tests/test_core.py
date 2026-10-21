"""Unit tests:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from ssw import dbm, planner, reliability, config, events  # noqa: E402


class DBM(unittest.TestCase):
    def test_closed_form_matches_numerical_integration(self):
        for v0, w, g in [(1180, 400, 2e-8), (350, 450, 1e-7), (2000, 380, 5e-8)]:
            r, v, t, dt = dbm.R0_KM, float(v0), 0.0, 5.0
            while r < 1.4e8:                      # RK4 on dv/dt = -g (v-w)|v-w|
                f = lambda x: -g * (x - w) * abs(x - w)
                k1 = f(v); k2 = f(v + dt * k1 / 2); k3 = f(v + dt * k2 / 2); k4 = f(v + dt * k3)
                r += dt * (v + dt * (k1 + 2 * k2 + 2 * k3) / 6 / 2)
                v += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                t += dt
            ta = dbm.arrival_time(v0, w, g, r)
            self.assertAlmostEqual(ta / 3600, t / 3600, delta=0.05)
            self.assertAlmostEqual(float(dbm.v_of_t(ta, v0, w, g)), v, delta=2)

    def test_gamma_inverse(self):
        r = dbm.l1_km("2024-05-10")
        for g in [1e-8, 5e-8, 2e-7]:
            t = dbm.arrival_time(900, 420, g, r)
            self.assertAlmostEqual(dbm.gamma_for_arrival(900, 420, t, r) / g, 1, delta=1e-3)

    def test_earth_distance_season(self):
        self.assertLess(dbm.sun_earth_km("2025-01-04"), 1.4715e8)   # perihelion
        self.assertGreater(dbm.sun_earth_km("2025-07-04"), 1.5205e8)  # aphelion

    def test_replay_scenario_gain(self):
        # UI replay: 1180 km/s, w=400, gamma=2e-8, 750,000 km offset -> ~20 min
        t_s1 = dbm.arrival_time(1180, 400, 2e-8, dbm.AU_KM - 1.5e6 - 0.75e6)
        t_l1 = dbm.arrival_time(1180, 400, 2e-8, dbm.AU_KM - 1.5e6)
        self.assertAlmostEqual((t_l1 - t_s1) / 60, 20.4, delta=0.6)


class ShockSpeed(unittest.TestCase):
    def test_mass_flux_shock_speed(self):
        # n 5 -> 15 cm^-3, v 400 -> 600 km/s: Vsh = (15*600 - 5*400) / (15 - 5) = 700 km/s,
        # with the jump 40 min after the DONKI time (OMNI is shifted to the bow shock)
        t0 = pd.Timestamp("2024-01-01T12:00Z")
        idx = pd.date_range(t0 - pd.Timedelta(hours=3), t0 + pd.Timedelta(hours=4), freq="1min")
        after = idx >= t0 + pd.Timedelta(minutes=40)
        l1 = pd.DataFrame({"V": np.where(after, 600.0, 400.0), "N": np.where(after, 15.0, 5.0)}, index=idx)
        m = events.shock_mass_flux(l1, t0)
        self.assertAlmostEqual(m["v_shock_mf"], 700.0, places=6)
        self.assertAlmostEqual(m["n_ratio"], 3.0, places=6)
        self.assertEqual(abs((m["t_shock_omni"] - t0).total_seconds() / 60 - 40) <= 1, True)


class Swarm(unittest.TestCase):
    def test_regular_tetrahedron_q_is_one(self):
        p = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        self.assertAlmostEqual(reliability.tet_q(p), 1.0, places=6)

    def test_no_failures_means_capable(self):
        self.assertAlmostEqual(reliability.p_capable(config.SAT, 0.0), 1.0)

    def test_four_craft_need_all_four(self):
        four = {k: config.SAT[k] for k in ["S1", "S2", "S3", "S4"]}
        self.assertAlmostEqual(reliability.p_capable(four, 0.1), 0.9 ** 4, places=9)

    def test_planner_holds_when_error_small(self):
        cfg = dict(config.SCOUT, u_threshold=10)
        self.assertEqual(planner.plan(cfg, 9000, 0.5, 0.004)["decision"]["action"], "hold (U within limit)")


if __name__ == "__main__":
    unittest.main()
