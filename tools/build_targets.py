#!/usr/bin/env python3
"""Build webapp/static/targets.json: the Space Apps test targets with positions and proper motions from ESA Gaia DR3.

Each target is found with a small cone search in the ESA Gaia archive around its approximate position, and the brightest
(or, for a high proper-motion star, the fastest) source in the cone is taken. The viewer then moves each star from Gaia's
epoch (J2016.0) to the date shown. WISE 0855-0714 is too faint for Gaia: its coordinates are the ones given in the
NASA Space Apps challenge text, with no proper motion.

Usage:  python tools/build_targets.py --out webapp/static/targets.json
Needs:  requests, and access to gea.esac.esa.int (ESA Gaia archive TAP service)
"""
import argparse
import json

import requests

TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
# name, note, approximate J2000 position (deg) used only to find the star, cone radius (deg), how to pick
TARGETS = [
    ("Barnard's Star", "red dwarf with the largest proper motion of any star", 269.4521, 4.6934, 0.06, "fastest"),
    ("Epsilon Indi B", "pair of brown dwarfs (Ba + Bb, one Gaia source) orbiting Epsilon Indi A", 331.07, -56.79, 0.06, "fastest_faint"),
    ("RW Aurigae", "young variable star", 76.9564, 30.4014, 0.01, "brightest"),
    ("R Draconis", "Mira-type variable star", 248.1675, 66.7550, 0.01, "brightest"),
]
WISE0855 = {"name": "WISE 0855-0714", "note": "fast-moving brown dwarf, the coldest known; too faint for Gaia",
            "ra": (8 + 55 / 60 + 10.831 / 3600) * 15, "dec": -(7 + 14 / 60 + 42.53 / 3600), "epoch": None, "pmra": None, "pmdec": None,
            "parallax": None, "g": None, "source": "coordinates from the NASA Space Apps Challenge 2026 text (RA 08h55m10.831s, Dec -7d14m42.53s)"}


def cone(ra, dec, r):
    q = f"""SELECT source_id, ra, dec, pmra, pmdec, parallax, ref_epoch, phot_g_mean_mag
            FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r}))"""
    t = requests.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": q}, timeout=180).json()
    cols = [c["name"] for c in t["metadata"]]
    return [dict(zip(cols, row)) for row in t["data"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for name, note, ra, dec, r, pick in TARGETS:
        rows = [x for x in cone(ra, dec, r) if x["phot_g_mean_mag"] is not None]
        pm = lambda x: (x["pmra"] or 0) ** 2 + (x["pmdec"] or 0) ** 2
        if pick == "fastest":
            s = max(rows, key=pm)
        elif pick == "fastest_faint":                      # the brown dwarfs share Epsilon Indi A's motion but are ~14 mag fainter
            s = max((x for x in rows if x["phot_g_mean_mag"] > 10), key=pm)
        else:
            s = min(rows, key=lambda x: x["phot_g_mean_mag"])
        print(f"{name:16s} Gaia DR3 {s['source_id']}  G={s['phot_g_mean_mag']:.2f}  pm=({s['pmra']:.1f}, {s['pmdec']:.1f}) mas/yr  plx={s['parallax']:.2f}  ({len(rows)} sources in cone)")
        out.append({"name": name, "note": note, "ra": s["ra"], "dec": s["dec"], "epoch": s["ref_epoch"], "pmra": s["pmra"], "pmdec": s["pmdec"],
                    "parallax": s["parallax"], "g": round(s["phot_g_mean_mag"], 3), "gaia_dr3": str(s["source_id"]), "source": "ESA Gaia DR3"})
    out.insert(2, WISE0855)
    json.dump({"source": "ESA Gaia DR3 (gea.esac.esa.int); WISE 0855 from the challenge text", "targets": out}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
