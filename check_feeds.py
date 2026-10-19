"""Check the LIVE-mode feeds against the real servers (run on a computer with internet).

  python check_feeds.py

Prints, for each feed the UI reads, whether it loads and whether the field names the
UI expects are really there. If anything says MISSING, send the printed keys back.
"""
import json
import urllib.request

NOAA = "https://services.swpc.noaa.gov/json/"
# CCMC moved DONKI off kauai.ccmc.gsfc.nasa.gov on 2026-09-30; the new host needs no API key.
DONKI = "https://ccmc.gsfc.nasa.gov/DONKI-API/get/"
CHECKS = [
    ("L1 wind", NOAA + "rtsw/rtsw_wind_1m.json", ["time_tag", "proton_speed", "proton_density", "active", "source"]),
    ("L1 mag", NOAA + "rtsw/rtsw_mag_1m.json", ["time_tag", "bz_gsm", "bt", "active"]),
    ("GOES X-ray", NOAA + "goes/primary/xrays-6-hour.json", ["time_tag", "flux", "energy"]),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "solar-swarm-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    ok = True
    for name, url, need in CHECKS:
        try:
            d = get(url)
            row = d[-1] if isinstance(d, list) and d else {}
            miss = [k for k in need if k not in row]
            act = sorted({r.get("source") for r in d if isinstance(r, dict) and r.get("active")}) if "active" in need else ""
            print(f"{name:11s} {'OK' if not miss else 'MISSING ' + str(miss)}  rows={len(d)}  active={act}")
            if miss:
                print("            keys present:", sorted(row))
                ok = False
        except Exception as e:
            print(f"{name:11s} FAILED  {e}")
            ok = False
    try:
        # The UI reads CME records and, inside each, its most accurate entry in cmeAnalyses.
        d = get(DONKI + "CME?startDate=2024-05-08&endDate=2024-05-12")
        an = [a for r in d for a in (r.get("cmeAnalyses") or [])]
        best = [a for a in an if a.get("isMostAccurate")]
        row, arow = (d[0] if d else {}), ((best or an or [{}])[0])
        miss = [k for k in ["activityID", "cmeAnalyses"] if k not in row]
        miss += [k for k in ["time21_5", "speed", "halfAngle", "latitude", "longitude", "isMostAccurate", "enlilList"]
                 if k not in arow]
        n_en = sum(1 for a in best if a.get("enlilList"))
        print(f"{'DONKI':11s} {'OK' if not miss else 'MISSING ' + str(miss)}  rows={len(d)}  "
              f"most-accurate analyses={len(best)}, with Enlil={n_en} (May 2024 Gannon storm window)")
        if miss:
            print("            keys present:", sorted(row), sorted(arow))
        ok &= not miss and n_en > 0
    except Exception as e:
        print(f"{'DONKI':11s} FAILED  {e}")
        ok = False
    print("\nAll feeds match the UI." if ok else "\nSomething differs — send this output back.")


if __name__ == "__main__":
    main()
