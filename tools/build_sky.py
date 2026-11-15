#!/usr/bin/env python3
"""Build webapp/static/sky.json (the star sky used by the viewer). Not needed at runtime.

Stars  (positions, proper motions, magnitudes)
    --source esa   ESA Hipparcos new reduction (van Leeuwen 2007), straight from ESA's archive through
                   ESA's TAP service (plain HTTPS, no astroquery). This is the intended source. Needs internet access to gea.esac.esa.int.
    --source hyg   Fallback for machines that cannot reach ESA: the HYG compilation (CC BY-SA 4.0), which is
                   built on the same Hipparcos catalogue. Known limitation: it caps proper motions at
                   9999.99 mas/yr (wrong for Barnard's Star only, which is far too faint to be drawn anyway).
Figures, boundaries, star names
    Stellarium "modern" sky culture = the IAU constellations as lists of Hipparcos numbers (CC BY-SA 4.0).
Label positions
    Computed here from the IAU boundaries: the point inside each constellation farthest from its border.

Positions are stored at epoch J2000.0 (ESA positions are for J1991.25 and are advanced with their proper
motions), so the viewer only needs  dt = (JD - 2451545) / 365.25  years to move any star.

Usage
    curl -O https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern/index.json
    python tools/build_sky.py --source esa --stellarium index.json --out webapp/static/sky.json --check
    # or, if ESA is unreachable:  --source hyg --hyg hygdata_v41.csv

Needs numpy, astropy and requests.
"""
import argparse
import csv
import json
import math
import sys
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import FK5, SkyCoord, get_constellation
from astropy.time import Time

warnings.filterwarnings("ignore")
STEP_DEG = 0.5            # boundary sampling step (arc length)
MAG_LIMIT = 6.5           # naked-eye limit; every star used by a constellation figure is kept regardless
NAME_MAG = 3.0            # only stars at least this bright get a printed name


# ----------------------------------------------------------------------------------------- stars ---
def stars_hyg(path, figure_hips):
    """-> dict hip -> (ra_deg, dec_deg, pmra, pmdec, mag, bv) at epoch J2000.0."""
    out = {}
    for row in csv.DictReader(open(path)):
        if not row["hip"]:
            continue
        hip, mag = int(row["hip"]), float(row["mag"])
        if mag > MAG_LIMIT and hip not in figure_hips:
            continue
        bv = float(row["ci"]) if row["ci"] else 0.6
        out[hip] = (float(row["ra"]) * 15.0, float(row["dec"]), float(row["pmra"]), float(row["pmdec"]), mag, bv)
    return out


TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"          # ESA's Gaia archive (also hosts Hipparcos)
ESA_TABLE = "public.hipparcos_newreduction"                          # van Leeuwen (2007); columns: hip, ra, dec, plx, pm_ra, pm_de, hp_mag, b_v ...


def tap_csv(adql):
    """Run an ADQL query on ESA's TAP service over plain HTTPS and return the rows as dicts."""
    import requests
    r = requests.post(TAP_URL, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, timeout=300)
    if r.status_code != 200:
        sys.exit(f"ESA TAP error {r.status_code}:\n{r.text[:600]}")
    return list(csv.DictReader(r.text.splitlines()))


def stars_esa(figure_hips):
    """Hipparcos new reduction straight from ESA; advanced from its epoch J1991.25 to J2000.0 with proper motions."""
    hips = ",".join(str(h) for h in sorted(figure_hips))
    rows = tap_csv(f"SELECT hip, ra, dec, plx, pm_ra, pm_de, hp_mag, b_v FROM {ESA_TABLE} WHERE hp_mag < {MAG_LIMIT} OR hip IN ({hips})")
    print(f"ESA returned {len(rows)} rows from {ESA_TABLE}")
    f = lambda r, k, d=float("nan"): float(r[k]) if r[k] not in ("", None) else d
    ra = np.array([f(r, "ra") for r in rows]); dec = np.array([f(r, "dec") for r in rows])
    pmra = np.array([f(r, "pm_ra", 0.0) for r in rows]); pmde = np.array([f(r, "pm_de", 0.0) for r in rows])
    plx = np.array([f(r, "plx", 0.0) for r in rows])
    dist = np.where(plx > 0.5, 1000.0 / np.maximum(plx, 0.5), 10000.0) * u.pc          # tiny parallax -> far away (motion is distance-independent)
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=dist, pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmde * u.mas / u.yr,
                  radial_velocity=0 * u.km / u.s, frame="icrs", obstime=Time(1991.25, format="jyear", scale="tdb"))
    sc = sc.apply_space_motion(new_obstime=Time(2000.0, format="jyear", scale="tdb"))
    out = {}
    for i, r in enumerate(rows):
        bv = f(r, "b_v")
        out[int(r["hip"])] = (sc.ra.deg[i], sc.dec.deg[i], pmra[i], pmde[i], f(r, "hp_mag"), bv if np.isfinite(bv) else 0.6)
    return out


# ------------------------------------------------------------------------------------ boundaries ---
def hms(s):
    h, m, sec = (float(x) for x in s.split(":"))
    return h + m / 60 + sec / 3600


def dms(s):
    sign = -1 if s.startswith("-") else 1
    d, m, sec = (float(x) for x in s.lstrip("+-").split(":"))
    return sign * (d + m / 60 + sec / 3600)


def precess_b1875_to_j2000(ra_deg, dec_deg):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=FK5(equinox=Time("B1875")))
    j = c.transform_to(FK5(equinox="J2000"))
    return j.ra.deg % 360, j.dec.deg


def norm_abbr(x):
    return "".join(ch for ch in str(x).upper() if ch.isalpha())          # TrA == TRA, SER2 == SER


def vec(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def build_boundaries(stel):
    pieces, ra_all, dec_all = [], [], []
    for e in stel["edges"]:
        p = e.split()
        kind, ra1, dec1, ra2, dec2 = p[1][0], hms(p[2]) * 15, dms(p[3]), hms(p[4]) * 15, dms(p[5])
        if kind == "M":      # meridian: RA fixed
            n = max(1, math.ceil(abs(dec2 - dec1) / STEP_DEG))
            dec = np.linspace(dec1, dec2, n + 1); ra = np.full(n + 1, ra1)
        else:                # parallel: Dec fixed, RA by the short way round
            d_ra = (ra2 - ra1 + 180) % 360 - 180
            n = max(1, math.ceil(abs(d_ra) * math.cos(math.radians(dec1)) / STEP_DEG))
            ra = ra1 + np.linspace(0, d_ra, n + 1); dec = np.full(n + 1, dec1)
        pieces.append((len(ra), norm_abbr(p[-2]), norm_abbr(p[-1]))); ra_all.append(ra % 360); dec_all.append(dec)
    ra_j, dec_j = precess_b1875_to_j2000(np.concatenate(ra_all), np.concatenate(dec_all))
    bounds, per_con, k = [], {}, 0
    for n, ca, cb in pieces:
        pts = np.stack([ra_j[k:k + n], dec_j[k:k + n]], 1); k += n
        bounds.append([round(float(v), 3) for v in pts.reshape(-1)])
        for c in (ca, cb):
            per_con.setdefault(c, []).append(pts)
    return bounds, {c: np.concatenate(v) for c, v in per_con.items()}, (pieces, ra_all, dec_all)


def label_anchors(per_con):
    """For each constellation, the point inside it that is farthest from its own boundary."""
    n = 60000
    i = np.arange(n) + 0.5                                                  # Fibonacci sphere, ~0.9 deg spacing
    dec = np.degrees(np.arcsin(1 - 2 * i / n)); ra = np.degrees(np.pi * (1 + 5 ** 0.5) * i) % 360
    names = np.array([norm_abbr(x) for x in get_constellation(SkyCoord(ra=ra * u.deg, dec=dec * u.deg), short_name=True)])
    cand = vec(ra, dec)
    out = {}
    for c, edge_pts in per_con.items():
        m = names == c
        if not m.any():
            continue
        bv = vec(edge_pts[:, 0], edge_pts[:, 1])
        d = np.degrees(np.arccos(np.clip(cand[m] @ bv.T, -1, 1))).min(axis=1)       # distance to nearest border point
        best = np.argmax(d)
        out[c] = (float(ra[m][best]), float(dec[m][best]))
    return out


# ------------------------------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["esa", "hyg"], required=True)
    ap.add_argument("--hyg", help="path to hygdata_v41.csv (for --source hyg)")
    ap.add_argument("--stellarium", required=True, help="path to Stellarium modern skyculture index.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="validate against astropy's IAU boundary tables")
    a = ap.parse_args()

    stel = json.load(open(a.stellarium))
    figure_hips = {h for c in stel["constellations"] for l in c["lines"] for h in l}
    cat = stars_esa(figure_hips) if a.source == "esa" else stars_hyg(a.hyg, figure_hips)
    missing = figure_hips - set(cat)
    if missing:
        sys.exit(f"no catalogue entry for figure stars HIP {sorted(missing)}")

    stars, index = [], {}
    for hip in sorted(cat, key=lambda h: cat[h][4]):                       # brightest first
        ra, dec, pmra, pmdec, mag, bv = cat[hip]
        index[hip] = len(stars)
        stars.append([round(ra % 360, 5), round(dec, 5), round(mag, 2), round(bv, 2), hip, round(pmra, 1), round(pmdec, 1)])
    print(f"stars: {len(stars)} (<= mag {MAG_LIMIT} plus all {len(figure_hips)} figure stars)")

    names = []
    for key, lst in stel["common_names"].items():
        if key.startswith("HIP "):
            hip = int(key.split()[1])
            if hip in index and stars[index[hip]][2] <= NAME_MAG and lst:
                names.append([hip, lst[0].get("english") or lst[0]["native"]])
    figures = {c["id"].split()[-1]: [[index[h] for h in line] for line in c["lines"]] for c in stel["constellations"]}
    print(f"names: {len(names)}   figures: {len(figures)} constellations")

    bounds, per_con, aux = build_boundaries(stel)
    anchors = label_anchors(per_con)
    labels = []
    for c in stel["constellations"]:
        abbr = c["id"].split()[-1]
        if norm_abbr(abbr) in anchors:
            ra, dec = anchors[norm_abbr(abbr)]
            labels.append([round(ra, 3), round(dec, 3), c["common_name"]["native"], c["common_name"].get("english", ""), abbr])
    print(f"boundaries: {len(bounds)} edges   labels: {len(labels)}")

    src = ("ESA Hipparcos new reduction (van Leeuwen 2007)" if a.source == "esa"
           else "Hipparcos (ESA) via the HYG compilation (CC BY-SA 4.0)")
    out = {"epoch": 2000.0,
           "attribution": f"Stars: {src}. Constellation figures, borders and star names: Stellarium modern sky culture (CC BY-SA 4.0; open source, NOT an agency source, built on the IAU borders).",
           "stars": stars, "names": names, "cl": figures, "labels": labels, "bounds": bounds}
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out)
    if a.check:
        check(stel, stars, figures, labels, aux)


def check(stel, stars, figures, labels, aux):
    pieces, ra_all, dec_all = aux
    print("\n== validation against astropy's independent IAU boundary tables ==")
    bad, tot, detail = 0, 0, []
    for abbr, polys in figures.items():
        idx = sorted({i for p in polys for i in p})
        got = get_constellation(SkyCoord(ra=[stars[i][0] for i in idx] * u.deg, dec=[stars[i][1] for i in idx] * u.deg), short_name=True)
        for i, g in zip(idx, got):
            tot += 1
            if norm_abbr(g) != norm_abbr(abbr):
                bad += 1; detail.append((abbr, str(g), stars[i][4]))
    print(f"figure stars inside their own constellation: {tot - bad}/{tot} (rest sit on a shared border: {detail[:6]})")
    lab = sum(norm_abbr(get_constellation(SkyCoord(ra=l[0] * u.deg, dec=l[1] * u.deg), short_name=True)) != norm_abbr(l[4]) for l in labels)
    print(f"labels outside their own constellation: {lab}/{len(labels)}")
    ok, fails = 0, []
    for j, ((n, ca, cb), e) in enumerate(zip(pieces, stel["edges"])):
        kind = e.split()[1][0]
        ra0, dec0 = ra_all[j][n // 2], dec_all[j][n // 2]
        d = 0.15
        dra = d / max(math.cos(math.radians(dec0)), 0.05)
        offs = [(dra, 0), (-dra, 0)] if kind == "M" else [(0, d), (0, -d)]
        r, dd = precess_b1875_to_j2000(np.array([ra0 + o[0] for o in offs]) % 360, np.clip(np.array([dec0 + o[1] for o in offs]), -89.9, 89.9))
        got = {norm_abbr(x) for x in get_constellation(SkyCoord(ra=r * u.deg, dec=dd * u.deg), short_name=True)}
        if got == {ca, cb}:
            ok += 1
        else:
            fails.append((ca, cb, sorted(got)))
    print(f"boundary edges whose two sides are exactly the two named constellations: {ok}/{len(pieces)}")


if __name__ == "__main__":
    main()
