#!/usr/bin/env python3
"""Build static/small_bodies.json: real asteroid, Jupiter-Trojan and trans-Neptunian orbits from NASA/JPL's
Small-Body Database (SBDB) Query API. Not needed at runtime.

The viewer advances each object from its osculating elements (two-body motion), so the dots are real objects moving
the way JPL's elements say they do. Only the brightest objects of each class are kept so the file stays small.

Usage:  python tools/build_small_bodies.py --out static/small_bodies.json
Needs:  requests, and internet access to ssd-api.jpl.nasa.gov
API:    https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
"""
import argparse
import datetime as dt
import json

import requests

API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
# class code(s) in SBDB, label shown in the viewer, absolute-magnitude cut-off (smaller H = bigger/brighter object)
CLASSES = {
    "neo":    ("group:neo", "Near-Earth objects", 17.75),
    "belt":   ("IMB,MBA,OMB", "Main-belt asteroids", 12.5),
    "trojan": ("TJN", "Jupiter Trojans", 13.5),
    "tno":    ("TNO", "Trans-Neptunian objects (Kuiper belt)", 8.0),
}
# orbit: semi-major axis (AU), eccentricity, inclination, node, perihelion arg, mean anomaly (deg), epoch (JD TDB);
# then absolute magnitude H, diameter (km, only where measured), name, numbered/provisional kind, NEO and PHA flags
FIELDS = "a,e,i,om,w,ma,epoch,H,diameter,full_name,kind,neo,pha"


def query(sb_class, hmax=None, limit=None):
    p = {"fields": FIELDS, "full-prec": "1"}
    if sb_class.startswith("group:"):
        p["sb-group"] = sb_class[6:]
    else:
        p["sb-class"] = sb_class
    if hmax is not None:
        p["sb-cdata"] = json.dumps({"AND": [f"H|LT|{hmax}"]})
    if limit is not None:
        p["limit"] = str(limit)
    r = requests.get(API, params=p, timeout=180)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"source": "NASA/JPL Small-Body Database (SBDB) Query API", "retrieved": dt.date.today().isoformat(), "classes": {}}
    for key, (code, label, hmax) in CLASSES.items():
        total = query(code, limit=1)["count"]
        d = query(code, hmax=hmax)
        now = dt.datetime.now(dt.timezone.utc).timestamp() / 86400 + 2440587.5
        rows = []
        for row in d["data"]:
            orbit = [round(float(x), 6) for x in row[:7]]
            h = round(float(row[7]), 2) if row[7] not in (None, "") else None
            diam = round(float(row[8]), 2) if row[8] not in (None, "") else None
            flags = ("n" if (row[10] or "")[1:2] == "n" else "p") + ("N" if row[11] == "Y" else "") + ("P" if row[12] == "Y" else "")
            rows.append(orbit + [h, diam, (row[9] or "").strip(), flags])
        before = len(rows)
        # a two-body advance from elements older than ~2 orbital periods drifts too far: drop those few objects
        rows = [r for r in rows if (now - r[6]) < 2 * r[0] ** 1.5 * 365.25]
        if before != len(rows):
            print(f"   dropped {before - len(rows)} objects whose elements are more than 2 orbital periods old")
        out["classes"][key] = {"columns": ["a_au", "e", "i_deg", "node_deg", "peri_deg", "M_deg", "epoch_jd_tdb", "H", "diameter_km", "name", "flags (n numbered / p provisional, N NEO, P PHA)"], "label": label, "total_known": total, "cutoff": f"absolute magnitude H < {hmax}", "rows": rows}
        print(f"{label:40s} {len(rows):6d} objects kept of {total:,} known ({hmax=})")
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
