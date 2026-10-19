"""Download real data: NASA DONKI (CMEs, interplanetary shocks) and L1 solar wind
from CDAWeb's HAPI server. Everything is cached under data/raw/ so re-runs are free.

Run on a computer with internet access (see README).
"""
from __future__ import annotations

import gzip
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DONKI_BASES = {
    # CCMC's own web service: no API key, no hourly limit. (Moved off kauai.ccmc... on 2026-09-30.)
    "ccmc": "https://ccmc.gsfc.nasa.gov/DONKI-API/get/{kind}?startDate={s}&endDate={e}",
    # api.nasa.gov gateway: needs a key (DEMO_KEY is heavily rate-limited).
    "nasa": "https://api.nasa.gov/DONKI/{kind}?startDate={s}&endDate={e}&api_key={key}",
}
HAPI = "https://cdaweb.gsfc.nasa.gov/hapi"
UA = {"User-Agent": "solar-swarm-capstone/1.0 (research; contact via GitHub)"}


def _get(url: str, tries: int = 4, timeout: int = 120) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 404):
                raise
            time.sleep(5 * (i + 1) if e.code != 429 else 60)
        except Exception as e:  # network hiccup
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url} ({last})")


def _months(start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cur = pd.Timestamp(year=s.year, month=s.month, day=1)
    while cur <= e:
        nxt = cur + pd.offsets.MonthBegin(1)
        yield max(cur, s), min(nxt - pd.Timedelta(days=1), e)
        cur = nxt


# ------------------------------------------------------------------ DONKI
def fetch_donki(kind: str, start: str, end: str, raw_dir: Path, source="ccmc", key="DEMO_KEY"):
    """kind: 'CME' or 'IPS'. Saves one JSON file per month."""
    out = Path(raw_dir) / "donki" / kind
    out.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for s, e in _months(start, end):
        f = out / f"{s:%Y-%m}.json"
        if f.exists() and s.to_period("M") < pd.Timestamp.now().to_period("M"):
            continue
        url = DONKI_BASES[source].format(kind=kind, s=f"{s:%Y-%m-%d}", e=f"{e:%Y-%m-%d}",
                                         key=urllib.parse.quote(key))
        body = _get(url).decode("utf-8").strip() or "[]"
        data = json.loads(body)
        f.write_text(json.dumps(data))
        n_new += 1
        print(f"  DONKI {kind} {s:%Y-%m}: {len(data)} records")
        if source == "nasa":
            time.sleep(1.0)
    return n_new


def load_donki(raw_dir: Path, kind: str) -> list[dict]:
    recs = {}
    for f in sorted((Path(raw_dir) / "donki" / kind).glob("*.json")):
        for r in json.loads(f.read_text()) or []:
            recs[r.get("activityID") or json.dumps(r, sort_keys=True)] = r
    return list(recs.values())


# ------------------------------------------------------------------ HAPI (CDAWeb)
def hapi_info(dataset: str) -> dict:
    ds = urllib.parse.quote(dataset)
    try:
        return json.loads(_get(f"{HAPI}/info?id={ds}"))                 # HAPI 2.x spelling
    except urllib.error.HTTPError:
        return json.loads(_get(f"{HAPI}/info?dataset={ds}"))


def _hapi_csv(dataset, params, start, stop):
    q = {"dataset": dataset, "parameters": ",".join(params),
         "start": f"{start:%Y-%m-%dT%H:%M:%SZ}", "stop": f"{stop:%Y-%m-%dT%H:%M:%SZ}",
         "format": "csv"}
    try:
        return _get(f"{HAPI}/data?" + urllib.parse.urlencode(q))
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        q = {"id": dataset, "parameters": ",".join(params), "time.min": q["start"],
             "time.max": q["stop"], "format": "csv"}           # HAPI 2.x spelling
        return _get(f"{HAPI}/data?" + urllib.parse.urlencode(q))


def _in_info_order(params: list[str], info: dict) -> list[str]:
    """CDAWeb rejects parameters listed out of dataset order (HAPI error 1411)."""
    names = [p["name"] for p in info["parameters"]]
    return sorted(params, key=lambda p: names.index(p) if p in names else len(names))


def _columns(info: dict, params: list[str]):
    cols, fills = [], {}
    meta = {p["name"]: p for p in info["parameters"]}
    for p in params:
        m = meta.get(p)
        if m is None:
            raise KeyError(f"parameter {p!r} not in dataset; available: {list(meta)}")
        size = (m.get("size") or [1])[0]
        names = [p] if size == 1 else [f"{p}_{i}" for i in range(size)]
        cols += names
        for n in names:
            fills[n] = m.get("fill")
    return cols, fills


def _parse(body: bytes, info: dict, params: list[str]) -> pd.DataFrame:
    cols, fills = _columns(info, params)
    if not body.strip() or b'"code": 1201' in body[:300]:            # 1201 = OK, no data in range
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    if b'"status"' in body[:300]:                                   # HAPI error sent as 200
        raise RuntimeError(f"HAPI server error: {body[:300].decode(errors='replace').strip()}")
    df = pd.read_csv(io.BytesIO(body), header=None, names=["time"] + cols)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    for c, fv in fills.items():
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if fv not in (None, "", "NaN", "nan"):
            df.loc[df[c] == float(fv), c] = float("nan")
    return df


def fetch_hapi_months(dataset, params, start, end, raw_dir: Path, info=None):
    """Month-by-month download, cached as gzip CSV."""
    out = Path(raw_dir) / "hapi" / dataset
    out.mkdir(parents=True, exist_ok=True)
    info = info or hapi_info(dataset)
    params = _in_info_order(params, info)
    for s, e in _months(start, end):
        f = out / f"{s:%Y-%m}.csv.gz"
        if f.exists():
            continue
        body = _hapi_csv(dataset, params, s, e + pd.Timedelta(days=1))
        df = _parse(body, info, params)
        if df.empty:                                                # not cached: data may arrive later
            print(f"  {dataset} {s:%Y-%m}: no data yet")
            continue
        df.to_csv(f, compression="gzip")
        print(f"  {dataset} {s:%Y-%m}: {len(df)} rows")


def fetch_hapi_window(dataset, params, start, stop, raw_dir: Path, tag: str, info=None):
    """One window (e.g. 3 h of 1-s magnetometer data after a shock)."""
    out = Path(raw_dir) / "hapi" / dataset / "windows"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{tag}.csv.gz"
    if not f.exists():
        info = info or hapi_info(dataset)
        params = _in_info_order(params, info)
        df = _parse(_hapi_csv(dataset, params, start, stop), info, params)
        df.to_csv(f, compression="gzip")
    return load_csv(f)


def load_csv(f) -> pd.DataFrame:
    df = pd.read_csv(f, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_hapi_months(raw_dir: Path, dataset: str) -> pd.DataFrame:
    files = sorted((Path(raw_dir) / "hapi" / dataset).glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([load_csv(f) for f in files]).sort_index()
    return df[~df.index.duplicated()]
