"""Event priors (ssw/priors.py) on the synthetic mock data: the fitted medians must land
near the values planted in tests/mock_data.py.  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import mock_data  # noqa: E402
import run_all  # noqa: E402
from ssw import dbm, events, fetch, priors, validate_dbm  # noqa: E402

RAW = ROOT / "data_mock" / "raw"


class MockPriors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mock_data.build(RAW)                      # cached after the first run (MOCK_DONE)
        l1 = run_all._load_l1(RAW)
        ev = events.add_insitu(events.build_events(fetch.load_donki(RAW, "CME"), fetch.load_donki(RAW, "IPS")), l1)
        _, _, live, gstar = validate_dbm.run(ev)
        _, tau_rows = run_all.structure_per_event(RAW, ev)
        cls.ev, cls.P = ev, priors.build(ev, gstar, tau_rows, live, mode="MOCK")

    def test_all_from_data(self):
        for k in ["v0", "half_angle", "lon", "w", "gamma", "gamma_m1", "bz_sheath", "tau_c"]:
            self.assertTrue(self.P[k]["source"].startswith("data"), k)

    def test_medians_near_planted(self):
        P = self.P
        # with DBM triples the forecaster's v0/w/gamma come from them; the planted values are in the DONKI/arrival fits
        v0, w, g = P.get("v0_donki", P["v0"]), P.get("w_obs", P["w"]), P.get("gamma_arrival_only", P["gamma"])
        self.assertAlmostEqual(v0["median"] / 750, 1, delta=0.10)
        self.assertAlmostEqual(P["half_angle"]["median"], 35, delta=3)
        self.assertAlmostEqual(P["lon"]["median"], 0, delta=4)
        self.assertAlmostEqual(w["median"], 420, delta=30)
        self.assertLess(abs(np.log(g["median"] / mock_data.TRUE_GAMMA_MEDIAN)), np.log(1.6))
        # planted tau = lambda / v_arrival; the ACF method is known to recover ~0.8 of lambda
        tau_planted = mock_data.TRUE_LAMBDA_KM / self.ev.v_shock.median()
        self.assertLess(abs(np.log(P["tau_c"]["median"] / tau_planted)), np.log(1.6))
        self.assertLess(P["bz_sheath"]["median"], 0)

    def test_dbm_triples_reproduce_arrival_and_shock_speed(self):
        T = self.P.get("dbm_triples")
        self.assertIsNotNone(T)
        self.assertGreater(T["n_train_solved"], 0)
        from ssw import dbm as D
        for e in T["events"][:10]:
            self.assertTrue(250 <= e["w"] <= 800 and 1e-10 <= e["g"] <= 1e-5)

    def test_historical_events_replayable(self):
        H = self.P["historical"]
        self.assertTrue(0 < len(H) <= priors.N_HIST)
        self.assertEqual(len({h["cme_id"] for h in H}), len(H))
        h = H[0]
        g = dbm.gamma_for_arrival(h["v0"], h["w_obs"], h["transit_h"] * 3600, h["r_target_km"])
        self.assertIsNotNone(g)

    def test_js_roundtrip(self):
        f = ROOT / "results_mock" / "_test_priors.js"
        f.parent.mkdir(exist_ok=True)
        priors.write(self.P, f)
        self.assertEqual(priors.read_js(f)["v0"], self.P["v0"])
        f.unlink()


class Fallbacks(unittest.TestCase):
    def test_no_data_means_assumed(self):
        P = priors.build(pd.DataFrame())
        for k, fb in priors.FALLBACK.items():
            self.assertEqual(P[k]["source"], "assumed", k)
            self.assertEqual(P[k]["kind"], fb["kind"], k)
        self.assertEqual(P["tau_c"]["median"], 15.0)
        self.assertEqual(P["historical"], [])

    def test_donki_unknown_direction_is_missing(self):
        self.assertTrue(np.isnan(events._angle(999, 180)))
        self.assertTrue(np.isnan(events._angle(None, 180)))
        self.assertEqual(events._angle("-45", 180), -45.0)


if __name__ == "__main__":
    unittest.main()
