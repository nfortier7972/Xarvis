#!/usr/bin/env python3
"""Audit the viewer's motion against reality. Run:  python tools/verify_motion.py

Every check compares our numbers with something independent: astronomical events whose date and geometry are
published (transits, eclipses, solstices, perihelion), well-known orbital periods, or astropy's own
space-motion code. Exit status 1 if any check fails.
"""
import json
import math
import os
import sys
import warnings

import numpy as np
import requests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import astropy.units as u                                            # noqa: E402
from astropy.coordinates import SkyCoord, get_body_barycentric       # noqa: E402
from astropy.time import Time                                        # noqa: E402
from webapp import server as A                                       # noqa: E402

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


def jd(iso):
    return Time(iso, scale="utc").jd


def helio(body, jds):
    return A.helio_ecl(body, A.utc(np.atleast_1d(jds)))


def geo(body, jds):
    """Geocentric ecliptic vector of a planet or the Sun (AU)."""
    e = helio("earth", jds)
    return (-e) if body == "sun" else helio(body, jds) - e


def lon(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360


def sep_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


A.load_ephemeris()
print(f"Ephemeris under test: {A.EPH_NAME}  (covers JD {A.DE_RANGE[0]:.0f} .. {A.DE_RANGE[1]:.0f}, i.e. 1849-12-26 .. 2150-01-22)\n")

# ------------------------------------------------------------------ 1. Earth's orbit: perihelion / aphelion
print("== Earth's orbit ==")
for label, when, expect_au in [("perihelion 2026", "2026-01-03 17:15", 0.9833), ("aphelion 2026", "2026-07-06 17:30", 1.0167)]:
    j = jd(when) + np.linspace(-6, 6, 12001)
    t = A.utc(j)
    emb = (get_body_barycentric("earth-moon-barycenter", t) - get_body_barycentric("sun", t)).xyz.to_value(u.au)
    d = np.linalg.norm(emb, axis=0)
    k = d.argmin() if "peri" in label else d.argmax()
    d_at = d[6000]
    dt_h = (j[k] - jd(when)) * 24
    # near an extremum the distance is almost flat, so the *time* of the extremum is poorly defined: compare the
    # distance at the published time, and require the extremum itself to fall within a few days
    check(f"Earth-Moon barycentre {label}", abs(d_at - expect_au) < 2e-4 and abs(dt_h) < 72,
          f"model: {d_at:.5f} AU at the published time (published {expect_au} AU); extremum at {dt_h:+.1f} h from it (flat there)")

# ------------------------------------------------------------------ 2. Solstices and equinoxes
print("\n== Sun's position at solstices / equinoxes ==")


def precession_lon_deg(jdv):
    T = (jdv - 2451545.0) / 36525.0
    return (5028.796195 * T + 1.1054348 * T * T) / 3600.0


for name, when, lam in [("March equinox 2024", "2024-03-20 03:06", 0), ("June solstice 2024", "2024-06-20 20:51", 90),
                        ("Sept equinox 2024", "2024-09-22 12:44", 180), ("Dec solstice 2024", "2024-12-21 09:21", 270),
                        ("March equinox 2026", "2026-03-20 14:46", 0), ("Dec solstice 2026", "2026-12-21 20:50", 270)]:
    j = jd(when)
    got = lon(geo("sun", j))[0]
    want = (lam - precession_lon_deg(j)) % 360        # published longitudes are of date; ours are J2000
    diff = (got - want + 180) % 360 - 180
    check(name, abs(diff) < 0.02, f"geocentric Sun longitude {got:.4f} deg (J2000 frame); expected {want:.4f} deg "
                                    f"(= {lam} deg of date minus {precession_lon_deg(j):.4f} deg precession); off by {diff * 3600:+.0f} arcsec")

# ------------------------------------------------------------------ 3. Transits (arcsecond-level geometry)
print("\n== Transits across the Sun (tests Earth + planet + Sun geometry to arcseconds) ==")
for name, start, hours, planet, t_expect, sep_expect in [
        ("Venus transit 2012", "2012-06-05 20:00", 12, "venus", "2012-06-06 01:29", 554.0),
        ("Mercury transit 2019", "2019-11-11 12:00", 8, "mercury", "2019-11-11 15:20", 76.0)]:
    j = jd(start) + np.linspace(0, hours / 24, hours * 600 + 1)
    s = sep_deg(geo("sun", j), geo(planet, j)) * 3600
    k = s.argmin()
    dmin = (j[k] - jd(t_expect)) * 1440
    check(name, abs(dmin) < 10 and abs(s[k] - sep_expect) < 25,
          f"closest approach {s[k]:.0f} arcsec at {Time(j[k], format='jd', scale='utc').iso[:16]} UTC "
          f"(published: ~{sep_expect:.0f} arcsec at {t_expect}); {dmin:+.1f} min, {s[k] - sep_expect:+.0f} arcsec")

# ------------------------------------------------------------------ 4. Moon: eclipses
print("\n== Moon (eclipses) ==")
for name, when, gamma in [("Solar eclipse 2017-08-21", "2017-08-21 18:25:32", 0.4367), ("Solar eclipse 2024-04-08", "2024-04-08 18:17:16", 0.3431),
                          ("Solar eclipse 2019-07-02", "2019-07-02 19:24:08", -0.6466)]:
    j = jd(when) + np.linspace(-0.25, 0.25, 3001)
    t = A.utc(j)
    m = A.geocentric_moon_ecl(t)
    s = sep_deg(geo("sun", j), m) * 1.0
    k = s.argmin()
    rm = np.linalg.norm(m[k]) * A.AU_KM
    expect = math.degrees(math.asin(abs(gamma) * 6378.137 / rm))     # geocentric separation implied by the eclipse's gamma
    check(name, abs((j[k] - jd(when)) * 1440) < 10 and abs(s[k] - expect) < 0.04,
          f"Moon-Sun separation minimum {s[k]:.3f} deg at {Time(j[k], format='jd', scale='utc').iso[11:16]} UTC; "
          f"gamma {gamma} implies {expect:.3f} deg at {when[11:16]} UTC")
for name, when in [("Lunar eclipse 2025-03-14", "2025-03-14 06:58"), ("Lunar eclipse 2022-11-08", "2022-11-08 10:59")]:
    j = jd(when)
    m = A.geocentric_moon_ecl(A.utc(np.array([j])))
    el = sep_deg(geo("sun", j), m)[0]
    check(name, abs(el - 180) < 1.2, f"Moon-Sun elongation {el:.2f} deg (total lunar eclipse is at ~180 deg, within the umbra width)")

# ------------------------------------------------------------------ 5. Orbital periods over a century
print("\n== Orbital periods (time to complete whole orbits inside the DE440s range, 1850-2150) ==")
T0 = jd("2000-01-01")
known = {"Mercury": 87.9691, "Venus": 224.701, "Earth": 365.2564, "Mars": 686.980, "Jupiter": 4332.589, "Saturn": 10759.22,
         "Uranus": 30688.5, "Neptune": 60182.0}
for body, P in known.items():
    half = 145 * 365.25
    n = max(1, int(2 * half / P))                                   # whole orbits that fit in +-145 years
    j = T0 - half + np.arange(0, 2 * half, max(0.25, P / 4000))
    ang = np.unwrap(np.radians(lon(helio(body, j))))
    k = int((ang[-1] - ang[0]) / (2 * np.pi))
    tc = np.interp(ang[0] + 2 * np.pi * np.arange(0, k + 1), ang, j)
    P_model = (tc[-1] - tc[0]) / k
    tol = 1.5e-3 if body in ("Jupiter", "Saturn", "Uranus", "Neptune") else 3e-4      # few orbits fit; Great Inequality moves Jupiter/Saturn ~0.05 %
    check(f"{body} sidereal period", abs(P_model / P - 1) < tol, f"model {P_model:.3f} d over {k} orbit(s) vs published {P} d ({(P_model / P - 1) * 100:+.4f} %)")
j = T0 + np.arange(0, 365.25 * 12, 0.04)
mlon = lon(A.geocentric_moon_ecl(A.utc(j))); slon = lon(geo("sun", j))
for label, series, expect in [("Moon sidereal month", mlon, 27.3217), ("Moon synodic month", (mlon - slon) % 360, 29.5306)]:
    ang = np.unwrap(np.radians(series))
    k = int((ang[-1] - ang[0]) / (2 * np.pi))
    tc = np.interp(ang[0] + 2 * np.pi * np.arange(0, k + 1), ang, j)
    P_model = (tc[-1] - tc[0]) / k
    check(label, abs(P_model - expect) < 0.01, f"model {P_model:.4f} d over {k} months vs published mean {expect} d")

# ------------------------------------------------------------------ 6. Orbit lines
print("\n== Orbit lines ==")
c = A.app.test_client()
for when in ["2000-01-01", "2026-11-14", "2100-06-01", "1900-01-01"]:
    o = c.get(f"/api/orbits?jd={jd(when)}").get_json()["orbits"]
    seam = max(np.linalg.norm(np.array(p[0]) - np.array(p[-1])) for p in o.values())
    pos = {b: helio(b, jd(when))[0] for b in A.PLANETS}
    miss = max(np.linalg.norm(np.array(o[b]) - pos[b], axis=1).min() for b in A.PLANETS)
    check(f"closed orbit lines through the planets, centred {when}", seam < 1e-9 and miss < 2e-3,
          f"seam gap {seam:.1e} AU; worst planet-to-line distance {miss * A.AU_KM:,.0f} km (sample spacing only)")

# ------------------------------------------------------------------ 7. SPHEREx orbit geometry
print("\n== SPHEREx orbit: NASA/JPL Horizons vs the official parameters (IPAC Explanatory Supplement, Table 1) ==")
now = A.now_jd()
real = A.spherex_snapshot(now)
if real["live"]:
    check("Horizons orbit vs official Table 1", abs(real["inc_deg"] - 97.951) < 0.25 and abs(real["period_min"] - 98) < 1.0 and 6900 < real["a_km"] < 7100,
          f"Horizons now: inclination {real['inc_deg']:.3f} deg (Table 1: 97.951), period {real['period_min']:.2f} min (Table 1: 98), radius {real['a_km']:.0f} km (semi-major axis 7,037)")
    worst = 0.0; rows = []
    for d in ["2025-04-01", "2025-08-15", "2026-01-01", "2026-06-01"]:
        s_ = A.spherex_snapshot(jd(d)); n_real = np.array(s_["n"]); n_mod = A.sso_snapshot(jd(d))[1]
        ang = math.degrees(math.acos(np.clip(abs(np.dot(n_real, n_mod)), -1, 1))); worst = max(worst, ang); rows.append(f"{d}: {ang:.1f}")
    check("fallback orbit model vs real orbit plane", worst < 5.0, "model plane (mean local time of ascending node 6:00 AM) differs from the real plane by deg -> " + ", ".join(rows))
else:
    print("[SKIP] Horizons not reachable: SPHEREx checks skipped")
sp = A.sso_snapshot(jd("2026-06-01"))
check("fallback model: inclination / period", abs(sp[4] - 97.951) < 1e-6 and abs(sp[3] - 98.0) < 0.2, f"inclination {sp[4]:.3f} deg, period {sp[3]:.2f} min (official: 97.951 deg, 98 min)")
lt = []
for d in np.linspace(0, 730, 60):
    j = jd("2025-06-01") + d
    n = A.sso_snapshot(j)[1]; sun = -A.earth_helio(j); sun /= np.linalg.norm(sun)
    lt.append(math.degrees(math.acos(abs(np.dot(n, sun)))))
check("fallback model: plane follows the Sun (dawn-dusk, high-beta orbit)", max(lt) < 35,
      f"angle between orbit normal and Sun direction is {min(lt):.0f}-{max(lt):.0f} deg over two years, i.e. the Sun stays {90 - max(lt):.0f}-90 deg out of the orbit plane")
check("launch / mission dates", abs(A.LAUNCH_JD - jd("2025-03-12 03:10:12")) < 1e-5 and not A.spherex_snapshot(jd("2025-03-12 03:00"))["available"] and A.spherex_snapshot(jd("2025-03-12 03:20"))["available"],
      "launch 12 March 2025 03:10:12 UTC (IPAC Explanatory Supplement): reported as not existing before, available after")

# ------------------------------------------------------------------ 7b. Small bodies (SBDB elements, two-body) vs Horizons
print("\n== Asteroids, Trojans, Kuiper belt: the browser's two-body motion from JPL SBDB elements vs JPL Horizons ==")


def two_body(el, jd_):
    a, e, inc, om, w, ma, epoch = el
    n = 0.01720209895 / a ** 1.5
    M = math.radians(ma) + n * (jd_ - epoch); M = (M + math.pi) % (2 * math.pi) - math.pi
    E = M + e * math.sin(M) if e < 0.8 else math.pi
    for _ in range(12):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    I, O, W_ = math.radians(inc), math.radians(om), math.radians(w)
    cO, sO, cW, sW, cI, sI = math.cos(O), math.sin(O), math.cos(W_), math.sin(W_), math.cos(I), math.sin(I)
    P = np.array([cW * cO - sW * sO * cI, cW * sO + sW * cO * cI, sW * sI]); Q = np.array([-sW * cO - cW * sO * cI, -sW * sO + cW * cO * cI, cW * sI])
    return a * (math.cos(E) - e) * P + a * math.sqrt(1 - e * e) * math.sin(E) * Q


def sbdb_elements(des):
    r = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api", params={"sstr": des, "full-prec": "1"}, timeout=60).json()
    el = {x["name"]: float(x["value"]) for x in r["orbit"]["elements"]}
    return [el["a"], el["e"], el["i"], el["om"], el["w"], el["ma"], float(r["orbit"]["epoch"])], r["object"]["fullname"].strip()


def horizons_helio(des, jd_):
    p = {"format": "text", "COMMAND": f"'{des};'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@10'",
         "TLIST": f"'{jd_:.6f}'", "TLIST_TYPE": "'JD'", "TIME_TYPE": "'TDB'", "REF_PLANE": "'ECLIPTIC'", "REF_SYSTEM": "'ICRF'", "OUT_UNITS": "'AU-D'", "VEC_TABLE": "'1'", "CSV_FORMAT": "'YES'"}
    t = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=p, timeout=60).text
    row = t[t.index("$$SOE") + 5:t.index("$$EOE")].strip().split(",")
    return np.array([float(x) for x in row[2:5]])


try:
    import requests
    for des, kind in [("1", "main-belt"), ("4", "main-belt"), ("624", "Jupiter Trojan"), ("588", "Jupiter Trojan"), ("136199", "Kuiper-belt/scattered")]:
        el, name = sbdb_elements(des)
        out = []
        for dyr in (0, 1, 5, -5):
            j_ = el[6] + dyr * 365.25
            err = np.linalg.norm(two_body(el, j_) - horizons_helio(des, j_))
            out.append((dyr, err))
        worst = max(e for _, e in out)
        check(f"{name} ({kind})", worst < 0.1,
              "position error vs Horizons at epoch +0/+1/+5/-5 years: " + ", ".join(f"{e:.4f} AU" for _, e in out) + f"  (object is {el[0]:.1f} AU from the Sun)")
except Exception as exc:
    print(f"[SKIP] small-body check needs JPL SBDB + Horizons ({exc!s:.80})")

# ------------------------------------------------------------------ 8. Stars: proper motion vs astropy
print("\n== Star proper motion (the same maths the browser uses vs astropy's apply_space_motion) ==")
sky = json.load(open(os.path.join(HERE, "..", "webapp", "static", "sky.json")))
S = np.array([[s[0], s[1], s[5], s[6]] for s in sky["stars"]])           # ra, dec (deg), pmra*, pmdec (mas/yr)
ra, dec = np.radians(S[:, 0]), np.radians(S[:, 1])
u0 = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
e_ra = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], 1)
e_de = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], 1)
mas = np.pi / 180 / 3600 / 1000
w = (S[:, 2:3] * e_ra + S[:, 3:4] * e_de) * mas                         # rad / year, equatorial
# Distances only set the speed in astropy's 3-D calculation; use a realistic ~30 km/s tangential speed per star
# (a fixed 100 pc would give fast stars thousands of km/s, where astropy's relativistic terms differ).
mu_arcsec = np.hypot(S[:, 2], S[:, 3]) / 1000.0
dist_pc = np.clip(30.0 / (4.74 * np.maximum(mu_arcsec, 1e-6)), 2.0, 1e4)
for years in [-100000, -10000, -1000, 1000, 10000, 100000]:
    v = u0 + years * w; v /= np.linalg.norm(v, axis=1, keepdims=True)    # the browser's formula
    sc = SkyCoord(ra=S[:, 0] * u.deg, dec=S[:, 1] * u.deg, distance=dist_pc * u.pc, pm_ra_cosdec=S[:, 2] * u.mas / u.yr, pm_dec=S[:, 3] * u.mas / u.yr,
                  radial_velocity=0 * u.km / u.s, obstime=Time(2000.0, format="jyear", scale="tdb"))
    ref = sc.apply_space_motion(new_obstime=Time(2000.0 + years, format="jyear", scale="tdb")).cartesian.xyz.value.T
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    err = np.degrees(np.arccos(np.clip((v * ref).sum(1), -1, 1))) * 3600
    moved = np.degrees(np.arccos(np.clip((v * u0).sum(1), -1, 1)))
    # tolerance: 0.1 arcsec plus 0.01 % of the distance the star travelled (astropy adds tiny relativistic terms)
    check(f"proper motion {years:+,d} yr ({len(S)} stars)", bool(np.all(err < 0.1 + 1e-4 * moved * 3600)),
          f"max difference from astropy {err.max():.4f} arcsec (median {np.median(err):.5f}); stars move up to {moved.max():.2f} deg, median {np.median(moved) * 60:.1f} arcmin")

# What the viewer does NOT model: radial velocity (Hipparcos has none). Estimate its effect with HYG's rv + distance.
hyg = os.environ.get("HYG_CSV")
if hyg and os.path.exists(hyg):
    import csv
    rows = {int(r["hip"]): r for r in csv.DictReader(open(hyg)) if r["hip"]}
    sel = [(i, rows[s[4]]) for i, s in enumerate(sky["stars"]) if s[4] in rows and float(rows[s[4]]["dist"]) < 5000 and float(rows[s[4]]["rv"]) != 0]
    idx = np.array([i for i, _ in sel]); rv = np.array([float(r["rv"]) for _, r in sel]); dist = np.array([float(r["dist"]) for _, r in sel])
    print(f"\n   Effect of ignoring radial velocity ({len(sel)} stars with an RV and a distance):")
    for years in [1000, 10000, 100000]:
        sc = SkyCoord(ra=S[idx, 0] * u.deg, dec=S[idx, 1] * u.deg, distance=dist * u.pc, pm_ra_cosdec=S[idx, 2] * u.mas / u.yr,
                      pm_dec=S[idx, 3] * u.mas / u.yr, radial_velocity=rv * u.km / u.s, obstime=Time(2000.0, format="jyear", scale="tdb"))
        a_ = sc.apply_space_motion(new_obstime=Time(2000.0 + years, format="jyear", scale="tdb")).cartesian.xyz.value.T
        b_ = sc.apply_space_motion(new_obstime=Time(2000.0, format="jyear", scale="tdb")).cartesian.xyz.value.T
        v = u0[idx] + years * w[idx]; v /= np.linalg.norm(v, axis=1, keepdims=True)
        a_ /= np.linalg.norm(a_, axis=1, keepdims=True)
        e = np.degrees(np.arccos(np.clip((v * a_).sum(1), -1, 1))) * 60
        print(f"     {years:>7,d} yr: neglecting RV shifts the median star by {np.median(e):.3f} arcmin, 99th percentile {np.percentile(e, 99):.2f}, worst {e.max():.1f} arcmin")

n_fail = results.count(False)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
