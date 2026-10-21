"""The Python Monte Carlo (ssw/montecarlo.py) against the browser model (ui/ss-physics-v11.js,
run through Node), and the Phase-6 checks on the browser model.  python -m unittest discover tests"""
import json
import math
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ssw import montecarlo as mc  # noqa: E402

NODE = shutil.which("node")
N = 200


def _node(*args, timeout=300):
    r = subprocess.run([NODE, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


class RandomStreams(unittest.TestCase):
    def test_mulberry32_reference_values(self):
        # first outputs of mulberry32(1) in JavaScript (node: require('./ui/ss-physics-v11.js').mulberry32(1))
        r = mc.mulberry32(1)
        self.assertEqual([round(r() * 4294967296) for _ in range(3)], [2693262067, 11749833, 2265367787])

    def test_same_seed_same_event(self):
        P = mc.load_priors()
        self.assertEqual(mc.make_event(42, P), mc.make_event(42, P))


@unittest.skipUnless(NODE, "node not installed")
class PythonMatchesBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code, out, err = _node("tests/js/mc_events.js", str(N), "1")
        if code:
            raise RuntimeError(err)
        cls.js = json.loads(out)
        P = mc.load_priors()
        cls.py = [mc.make_event(1 + i, P) for i in range(N)]

    def test_medians_within_5pct(self):
        for k, tol_abs in [("gain", .05), ("err0", 60), ("err12", 60), ("err24", 60), ("errPost", 5), ("dv", .5), ("burns", 0), ("Lfinal", 1)]:
            a = np.nanmedian([np.nan if s[k] is None else float(s[k]) for s in self.js])
            b = np.nanmedian([float(s[k]) for s in self.py])
            self.assertLessEqual(abs(a - b), max(.05 * abs(a), tol_abs), f"{k}: JS {a} vs Python {b}")

    def test_same_seed_same_event_in_both_languages(self):
        same = sum(abs(j["gain"] - p["gain"]) < 1e-6 and j["burns"] == p["burns"] and abs(j["dv"] - p["dv"]) < 1e-6
                   for j, p in zip(self.js, self.py))
        self.assertGreaterEqual(same / N, .95, f"only {same}/{N} events identical")


@unittest.skipUnless(NODE, "node not installed")
class BrowserModelChecks(unittest.TestCase):
    """tests/js/v11_checks.js: axis gain, determinism, filter coverage, Δv budget, continuity,
    historical arrivals, batch time."""
    def test_all_checks_pass(self):
        code, out, err = _node("tests/js/v11_checks.js", "1000", timeout=600)
        res = json.loads(out)
        failed = {k: v for k, v in res["checks"].items() if not v["pass"]}
        self.assertFalse(failed, json.dumps(failed, indent=1))
        cov = res["numbers"]["coverage_2sigma"]["all_updates"]
        self.assertTrue(.9 < cov < .99 and math.isfinite(cov))


if __name__ == "__main__":
    unittest.main()
