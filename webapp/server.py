"""XARVIS: SPHEREx + solar system viewer (Flask backend). Official sources only.

Run:  python run.py   (from the repository root) then open http://127.0.0.1:5000

Frame for everything sent to the browser: heliocentric ecliptic J2000, AU, times as Julian Dates (UTC).

Data sources (all official)
    Planets, Sun, Moon   NASA/JPL DE440s planetary ephemeris (NAIF SPK kernel, 1849-12-26 .. 2150-01-22), read through
                         astropy + jplephem. Outside that range the browser uses JPL's published mean orbital elements.
    SPHEREx trajectory   NASA/JPL Horizons, spacecraft -163182. Horizons builds it from the US Space Force's Space-Track
                         TLEs from launch to 30 days after the latest one. Outside that range (or if Horizons cannot be
                         reached) an ideal sun-synchronous orbit with the official orbital parameters is used instead
                         (IPAC SPHEREx Explanatory Supplement, Table 1); the position along the orbit is then unknown.
    Stars                ESA Hipparcos (static/sky.json, built by tools/build_sky.py).

API
    /api/meta                       dates, coverage, sources
    /api/track?jd0=&step=&n=        planet and Moon positions on a regular time grid
    /api/spherex?jd=                SPHEREx orbit snapshot (position, orbit plane, period) at one instant
    /api/orbits?jd=                 closed planet orbit paths centred on that instant
    /api/sbdb/lookup, /api/sbident  small-body lookups (NASA/JPL)
    /api/analysis...                SPHEREx image analysis jobs (webapp/analysis.py -> xarvis_pipeline)
"""
import datetime as dt
import math
import re
import threading
import time
import warnings

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

ASTRO = False
try:
    import astropy.units as u
    from astropy.time import Time
    from astropy.utils import iers
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    iers.conf.auto_download = False
    warnings.filterwarnings("ignore")
    ASTRO = True
except ImportError:
    pass

app = Flask(__name__)

from .analysis import bp as analysis_bp          # noqa: E402  SPHEREx image analysis (background jobs)
app.register_blueprint(analysis_bp)

AU_KM = 149_597_870.7
EARTH_R_KM = 6371.0
MU_EARTH = 398_600.4418
OBLIQ = math.radians(84381.406 / 3600)       # IAU 2006 mean obliquity at J2000
J2000 = 2451545.0

PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
PERIOD_DAYS = {"Mercury": 87.969, "Venus": 224.701, "Earth": 365.256, "Mars": 686.980,
               "Jupiter": 4332.59, "Saturn": 10759.22, "Uranus": 30688.5, "Neptune": 60182.0}


def _jd(y, mo, d, h=0, mi=0, s=0):
    return (dt.datetime(y, mo, d, h, mi, s) - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0 + 2440587.5


# SPHEREx facts, all from the IPAC SPHEREx Explanatory Supplement v2.1 (2026-11-10) and the NASA/JPL Horizons entry
LAUNCH_JD = _jd(2025, 3, 12, 3, 10, 12)          # "launched on 12 March 2025 at 03:10:12 UTC"
SCIENCE_START_JD = _jd(2025, 5, 1)               # "On May 1, 2025, the spacecraft began regular science operations"
SCIENCE_END_JD = _jd(2027, 6, 1)                 # "nominal 25-month science mission"
SPHEREX_A_KM = 7037.41                           # Table 1: mean semi-major axis
SPHEREX_INC_DEG = 97.951                         # Table 1: inclination
SPHEREX_MLTAN_H = 6.0                            # Table 1: mean local time of the ascending node, 6:00 AM
HORIZONS_ID = "-163182"                          # SPHEREx in JPL Horizons
HORIZONS_START_JD = _jd(2025, 3, 11)             # Horizons trajectory: 2025-Mar-11 .. latest TLE + 30 d
HORIZONS_AHEAD_DAYS = 28.0                       # stay safely inside "+30 d"


def now_jd():
    return time.time() / 86400.0 + 2440587.5


# ------------------------------------------------------------------ JPL DE440s ---
DE_RANGE = None                  # (start_jd, end_jd) in TDB once the kernel is loaded
EPH_NAME = "JPL DE440s (not loaded)"
_de_lock = threading.Lock()


def load_ephemeris():
    """Load NASA/JPL's DE440s from NAIF (a 32 MB file, downloaded once and cached by astropy)."""
    global DE_RANGE, EPH_NAME
    if not ASTRO:
        EPH_NAME = "unavailable (astropy not installed)"
        return
    with _de_lock:
        if DE_RANGE is not None:
            return
        try:
            solar_system_ephemeris.set("de440s")
            from astropy.coordinates.solar_system import _get_kernel
            segs = _get_kernel("de440s").segments
            DE_RANGE = (max(s.start_jd for s in segs), min(s.end_jd for s in segs))
            EPH_NAME = "JPL DE440s (NASA/JPL NAIF)"
            print(f"[ephemeris] JPL DE440s loaded, covers JD {DE_RANGE[0]:.1f} .. {DE_RANGE[1]:.1f}")
        except Exception as exc:
            EPH_NAME = "JPL DE440s unavailable"
            print(f"[ephemeris] could not load JPL DE440s ({exc!s:.100}). Install jplephem and allow naif.jpl.nasa.gov. "
                  f"The viewer falls back to JPL's mean orbital elements.")


def in_de(jd0, jd1=None):
    jd1 = jd0 if jd1 is None else jd1
    return DE_RANGE is not None and DE_RANGE[0] + 1 <= min(jd0, jd1) and max(jd0, jd1) <= DE_RANGE[1] - 1


def utc(jd):
    far = np.any(np.abs(np.asarray(jd, float) - J2000) > 150 * 365.25)
    return Time(jd, format="jd", scale="tt" if far else "utc")


# -------------------------------------------------------------------- maths ---
def eq_to_ecl(v):
    v = np.asarray(v, float)
    c, s = math.cos(OBLIQ), math.sin(OBLIQ)
    return np.stack([v[..., 0], v[..., 1] * c + v[..., 2] * s, -v[..., 1] * s + v[..., 2] * c], axis=-1)


def normalize(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def close_loop(pts):
    """Close a path sampled over one period centred on "now" (middle sample). Real orbits don't repeat exactly, so
    the ends differ by up to millions of km; the drift is removed with a correction that is zero at the middle sample,
    so the curve still passes exactly through the body's position and the seam sits on the far side of the orbit."""
    pts = np.array(pts, float)
    n = len(pts) - 1
    w = (np.arange(n + 1) - n / 2) / (n / 2)
    pts = pts - 0.5 * (pts[-1] - pts[0]) * w[:, None]
    pts[-1] = pts[0]
    return pts


def helio_ecl(body, t):
    d = get_body_barycentric(body.lower(), t) - get_body_barycentric("sun", t)
    return eq_to_ecl(np.moveaxis(d.xyz.to_value(u.au), 0, -1))


def geocentric_moon_ecl(t):
    d = get_body_barycentric("moon", t) - get_body_barycentric("earth", t)
    return eq_to_ecl(np.moveaxis(d.xyz.to_value(u.au), 0, -1))


def earth_helio(jd):
    if in_de(jd):
        return helio_ecl("earth", utc(jd))
    # outside DE440s only the Sun's direction matters (for the orbit-model plane): JPL's mean elements for Earth
    L = math.radians(100.46457166 + 35999.37244981 * (jd - J2000) / 36525.0)
    return np.array([math.cos(L), math.sin(L), 0.0])


# ------------------------------------------------------------------ SPHEREx ---
_hz_cache = {}
_hz_fail = {"until": 0.0}


def horizons_spherex(jd):
    """SPHEREx geocentric state (position km, velocity km/s) in ecliptic J2000 from NASA/JPL Horizons, or None."""
    if not (HORIZONS_START_JD <= jd <= now_jd() + HORIZONS_AHEAD_DAYS):
        return None
    key = round(jd * 1440)                                        # one answer per minute
    if key in _hz_cache:
        return _hz_cache[key]
    if time.time() < _hz_fail["until"]:
        return None
    params = {"format": "text", "COMMAND": f"'{HORIZONS_ID}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'",
              "CENTER": "'500@399'", "TLIST": f"'{key / 1440.0:.9f}'", "TLIST_TYPE": "'JD'", "TIME_TYPE": "'UT'",
              "REF_PLANE": "'ECLIPTIC'", "REF_SYSTEM": "'ICRF'", "OUT_UNITS": "'KM-S'", "VEC_TABLE": "'2'", "CSV_FORMAT": "'YES'"}
    last = None
    for attempt in range(3):
        try:
            txt = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=params, timeout=30).text
            i, j = txt.index("$$SOE"), txt.index("$$EOE")
            row = txt[i + 5:j].strip().split(",")
            st = np.array([float(x) for x in row[2:8]])
            if len(_hz_cache) > 5000:
                _hz_cache.clear()
            _hz_cache[key] = (st[:3], st[3:])
            return _hz_cache[key]
        except Exception as exc:
            last = exc
            time.sleep(1.5)
    _hz_fail["until"] = time.time() + 60                            # back off for a minute, use the model meanwhile
    print(f"[SPHEREx] Horizons unavailable ({last!s:.80}); using the orbit model for a minute")
    return None


_track_cache = {}


def horizons_spherex_track(jd):
    """SPHEREx geocentric positions (km, ecliptic J2000) every 2 minutes from 60 minutes before to 60 minutes after jd,
    from NASA/JPL Horizons. The viewer interpolates along these, so the drawn spacecraft stays on the real trajectory
    (a single snapshot propagated on a circle drifts ~0.5 deg within an hour)."""
    start = math.floor(jd * 720) / 720 - 30 / 720                 # 2-minute grid
    key = round(start * 720)
    if key in _track_cache:
        return _track_cache[key]
    if time.time() < _hz_fail["until"]:
        return None
    params = {"format": "text", "COMMAND": f"'{HORIZONS_ID}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'",
              "CENTER": "'500@399'", "START_TIME": f"'JD {start:.9f}'", "STOP_TIME": f"'JD {start + 61 / 720:.9f}'", "STEP_SIZE": "'2m'",
              "TIME_TYPE": "'UT'", "REF_PLANE": "'ECLIPTIC'", "REF_SYSTEM": "'ICRF'", "OUT_UNITS": "'KM-S'", "VEC_TABLE": "'1'", "CSV_FORMAT": "'YES'"}
    try:
        txt = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=params, timeout=30).text
        i, j = txt.index("$$SOE"), txt.index("$$EOE")
        rows = [ln.split(",") for ln in txt[i + 5:j].strip().splitlines()]
        out = [[float(r[0]), round(float(r[2]), 3), round(float(r[3]), 3), round(float(r[4]), 3)] for r in rows]
    except Exception:
        return None
    if len(_track_cache) > 2000:
        _track_cache.clear()
    _track_cache[key] = out
    return out


def mean_sun_ra_j2000_deg(jd):
    T = (jd - J2000) / 36525.0
    return (280.46646 + 36000.76983 * T - 1.396971 * T) % 360.0


def sso_snapshot(jd):
    """Ideal sun-synchronous orbit with SPHEREx's published parameters: plane set by the mean local time of the
    ascending node (6:00 AM) and the Sun's mean position; phase along the orbit arbitrary."""
    a = SPHEREX_A_KM
    inc = math.radians(SPHEREX_INC_DEG)
    period_s = 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
    raan = math.radians(mean_sun_ra_j2000_deg(jd) + (SPHEREX_MLTAN_H - 12.0) * 15.0)
    ua = 2 * math.pi * (((jd * 86400.0) % period_s) / period_s)
    cu, su, co, so, ci, si = math.cos(ua), math.sin(ua), math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc)
    r_eq = np.array([cu * co - su * ci * so, cu * so + su * ci * co, su * si])
    n_eq = np.array([so * si, -co * si, ci])
    return eq_to_ecl(r_eq), eq_to_ecl(n_eq), a, period_s / 60.0, SPHEREX_INC_DEG


def spherex_snapshot(jd):
    """Orbit state at one instant: unit vector to SPHEREx, orbit normal, radius, period and where the numbers came from."""
    base = {"jd": jd, "available": jd >= LAUNCH_JD}
    st = horizons_spherex(jd)
    if st is not None:
        r, v = st
        base["jd"] = round(jd * 1440) / 1440.0             # Horizons was asked for this minute: label the state with it
        rn = float(np.linalg.norm(r))
        h = np.cross(r, v)
        sma = -MU_EARTH / (2 * (float(np.dot(v, v)) / 2 - MU_EARTH / rn))      # osculating semi-major axis
        period_min = 2 * math.pi * math.sqrt(sma ** 3 / MU_EARTH) / 60.0
        # The osculating period differs from SPHEREx's true average period (Earth's bulge), which made the drawn
        # spacecraft drift ~0.6 deg within an hour. Measure the real angular rate from a second Horizons state about one
        # orbit later: the angle travelled around the orbit normal, plus one full turn.
        jd2 = round((base["jd"] + period_min / 1440.0) * 1440) / 1440.0
        st2 = horizons_spherex(jd2)
        if st2 is not None:
            u1, u2, nn = normalize(r), normalize(st2[0]), normalize(h)
            dphi = math.atan2(float(np.dot(np.cross(u1, u2), nn)), float(np.dot(u1, u2)))
            rate = (2 * math.pi + dphi) / ((jd2 - base["jd"]) * 1440.0)           # rad per minute
            if abs(rate * period_min / (2 * math.pi) - 1) < 0.02:                  # sanity check
                period_min = 2 * math.pi / rate
        return {**base, "live": True, "u": normalize(r).tolist(), "n": normalize(h).tolist(), "a_km": rn,
                "period_min": period_min, "track": horizons_spherex_track(jd),
                "inc_deg": math.degrees(math.acos(float(np.clip(  # inclination to the Earth's equator (the ecliptic-frame normal is rotated back)
                    (h[1] * math.sin(OBLIQ) + h[2] * math.cos(OBLIQ)) / np.linalg.norm(h), -1, 1)))),
                "source": "NASA/JPL Horizons (from Space-Track TLEs)"}
    u_vec, n_vec, a, period, inc = sso_snapshot(jd)
    if jd < HORIZONS_START_JD:
        why = "before the Horizons trajectory begins (11 Mar 2025)"
    elif jd > now_jd() + HORIZONS_AHEAD_DAYS:
        why = "beyond the 30 days Horizons predicts"
    else:
        why = "Horizons not reachable"
    return {**base, "live": False, "u": normalize(u_vec).tolist(), "n": normalize(n_vec).tolist(), "a_km": a,
            "period_min": period, "inc_deg": inc,
            "source": f"APPROXIMATE orbit model ({why}); official orbit parameters, position along the orbit unknown"}


# ------------------------------------------------------------------- routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about", strict_slashes=False)
@app.route("/about.html")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):                 # API callers get JSON, people get the page
        return jsonify({"error": "not found"}), 404
    return render_template("404.html"), 404


@app.route("/api/meta")
def meta():
    return jsonify({
        "ephemeris": EPH_NAME, "now_jd": now_jd(),
        "de_start_jd": DE_RANGE[0] if DE_RANGE else None, "de_end_jd": DE_RANGE[1] if DE_RANGE else None,
        "launch_jd": LAUNCH_JD, "science_start_jd": SCIENCE_START_JD, "science_end_jd": SCIENCE_END_JD,
        "horizons_start_jd": HORIZONS_START_JD, "horizons_end_jd": now_jd() + HORIZONS_AHEAD_DAYS,
    })


@app.route("/api/track")
def track():
    """Positions of every planet (heliocentric) and the Moon (geocentric) on a regular grid, from JPL DE440s."""
    jd0, step = float(request.args["jd0"]), float(request.args["step"])
    n = int(min(max(int(request.args["n"]), 4), 6000))
    if not ASTRO or not in_de(jd0, jd0 + step * (n - 1)):
        return jsonify({"error": "outside the JPL DE440s ephemeris"}), 422
    t = utc(jd0 + step * np.arange(n))
    pos = np.stack([helio_ecl(b, t) for b in PLANETS], axis=1)
    moon = geocentric_moon_ecl(t)
    return jsonify({"jd0": jd0, "step": step, "n": n, "bodies": PLANETS, "pos": np.round(pos.reshape(-1), 9).tolist(),
                    "moon": np.round(moon.reshape(-1), 9).tolist()})


@app.route("/api/spherex")
def spherex_route():
    return jsonify(spherex_snapshot(float(request.args.get("jd", now_jd()))))


_orbit_cache = {}


@app.route("/api/orbits")
def orbits():
    """Each planet's path over one full period, centred on the requested time and closed (see close_loop)."""
    jd = float(request.args.get("jd", now_jd()))
    if not ASTRO or not in_de(jd):
        return jsonify({"error": "outside the JPL DE440s ephemeris"}), 422
    key = round(jd / 30.0)
    if key not in _orbit_cache:
        out = {}
        for name in PLANETS:
            ts = np.clip(jd + np.linspace(-0.5, 0.5, 361) * PERIOD_DAYS[name], DE_RANGE[0] + 1, DE_RANGE[1] - 1)   # stay inside the kernel
            out[name] = np.round(close_loop(helio_ecl(name, utc(ts))), 7).tolist()
        if len(_orbit_cache) > 40:
            _orbit_cache.clear()
        _orbit_cache[key] = out
    return jsonify({"jd": jd, "orbits": _orbit_cache[key]})



# ------------------------------------------------------------- object lookup (any JPL small body) ---
SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"
_lookup_cache = {}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def horizons_elements(des, kind):
    """Heliocentric ecliptic J2000 osculating elements of a small body at the current time, from NASA/JPL Horizons."""
    cmd = f"{des};" if kind[1:2] == "n" and str(des).isdigit() else f"DES={des};" + (" CAP;" if kind.startswith("c") else "")
    try:
        t = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", timeout=30, params={
            "format": "text", "COMMAND": f"'{cmd}'", "EPHEM_TYPE": "ELEMENTS", "CENTER": "'500@10'", "REF_PLANE": "ECLIPTIC",
            "TLIST": f"{now_jd():.5f}", "TLIST_TYPE": "JD", "TIME_TYPE": "TDB", "CSV_FORMAT": "YES", "OUT_UNITS": "AU-D"}).text
        head = re.search(r"^\s*JDTDB,.*$", t, re.M).group(0)
        body = re.search(r"\$\$SOE\s*(.*?)\$\$EOE", t, re.S).group(1).strip().splitlines()[0]
        cols = [c.strip() for c in head.split(",")]
        vals = [v.strip() for v in body.split(",")]
        el = {c: float(v) for c, v in zip(cols, vals) if c and c != "Calendar Date (TDB)"}
        return el if el.get("EC", 1) < 1 and el.get("A", -1) > 0 else None
    except Exception:
        return None


@app.route("/api/sbdb/lookup")
def sbdb_lookup():
    """Look up one asteroid or comet in NASA/JPL's Small-Body Database and return its orbit in the same row format as
    static/small_bodies.json, so the viewer can track it. Several matches -> the list, so the user can pick one."""
    q = (request.args.get("q") or "").strip()[:60]
    if not q:
        return jsonify({"error": "empty search"}), 400
    if q.lower() in _lookup_cache:
        return jsonify(_lookup_cache[q.lower()])
    try:
        r = requests.get(SBDB_API, params={"sstr": q, "phys-par": "1", "full-prec": "1"}, timeout=25)
        d = r.json()
    except Exception as e:                                   # network problem: say so, the viewer keeps working
        return jsonify({"error": f"NASA/JPL SBDB did not answer ({type(e).__name__})"}), 502
    if str(d.get("code")) == "300":
        out = {"matches": [{"des": m.get("pdes"), "name": m.get("name")} for m in d.get("list", [])][:40]}
    elif "object" not in d:
        out = {"error": d.get("message", "not found")}
    else:
        o, orb = d["object"], d.get("orbit") or {}
        el = {e["name"]: _num(e.get("value")) for e in orb.get("elements", [])}
        phys = {p["name"]: p.get("value") for p in d.get("phys_par", [])}
        epoch = _num(orb.get("epoch"))
        e, a = el.get("e"), el.get("a")
        if e is None or a is None or epoch is None or e >= 1 or a <= 0:
            out = {"error": f"{o.get('fullname', q)} is on an open (hyperbolic) or incomplete orbit, which this viewer cannot draw"}
        else:
            ma = el.get("ma")
            if ma is None and el.get("tp") is not None:      # some comets give the perihelion time instead of the mean anomaly
                ma = (0.01720209895 / a ** 1.5 * (epoch - el["tp"]) * 180 / math.pi) % 360
            kind = o.get("kind") or ""
            # SBDB's elements can be years old (Bennu: 2011), and a two-body advance from there drifts by millions of km.
            # Ask Horizons for osculating elements at the current date instead; keep SBDB's if Horizons does not answer.
            fresh = horizons_elements(o.get("des") or q, kind)
            if fresh:
                a, e, epoch = fresh["A"], fresh["EC"], fresh["JDTDB"]
                el.update({"i": fresh["IN"], "om": fresh["OM"], "w": fresh["W"]}); ma = fresh["MA"]
            flags = ("n" if kind[1:2] == "n" else "p") + ("N" if o.get("neo") else "") + ("P" if o.get("pha") else "")
            h = _num(phys.get("H"))
            row = [a, e, el.get("i"), el.get("om"), el.get("w"), ma, epoch, h, _num(phys.get("diameter")), o.get("fullname", q).strip(), flags]
            out = {"object": {"row": row, "comet": kind.startswith("c"), "class": (o.get("orbit_class") or {}).get("name"),
                              "spkid": o.get("spkid"), "elements": "NASA/JPL Horizons, osculating at today's date" if fresh else "NASA/JPL SBDB",
                              "source": "NASA/JPL Small-Body Database API"}}
    if len(_lookup_cache) > 500:
        _lookup_cache.clear()
    _lookup_cache[q.lower()] = out
    return jsonify(out)



# ------------------------------------------- known asteroids and comets inside a patch of sky (JPL SB Ident) ---
SBIDENT_API = "https://ssd-api.jpl.nasa.gov/sb_ident.api"
_ident_cache = {}


def _sexa(x, hours):
    """degrees -> 'hh-mm-ss.ss' (RA) or '[M]dd-mm-ss.s' (Dec), the formats the SB Ident API wants"""
    neg = x < 0
    v = abs(x) / 15 if hours else abs(x)
    d = int(v); m = int((v - d) * 60); sec = (v - d - m / 60) * 3600
    if sec >= 59.995:
        sec = 0; m += 1
    if m == 60:
        m = 0; d += 1
    return ("M" if neg else "") + f"{d:02d}-{m:02d}-{sec:05.2f}"


@app.route("/api/sbident")
def sbident():
    """Which known asteroids and comets are inside a box of sky at a given time, seen from Earth's centre (NASA/JPL SB Ident,
    fast first pass: positions good to about an arcminute)."""
    try:
        ra, dec, jd = float(request.args["ra"]), float(request.args["dec"]), float(request.args.get("jd", now_jd()))
        hw = min(max(float(request.args.get("hw", 0.25)), 0.01), 5.65)
    except (KeyError, ValueError):
        return jsonify({"error": "need ra, dec (degrees), optional jd and hw (half-width, degrees)"}), 400
    t = dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc) + dt.timedelta(days=jd - 2451545.0)
    key = (round(ra, 3), round(dec, 3), round(jd, 3), round(hw, 3))
    if key in _ident_cache:
        return jsonify(_ident_cache[key])
    params = {"mag-required": "true", "req-elem": "false", "mpc-code": "500", "obs-time": t.strftime("%Y-%m-%d_%H:%M:%S"),
              "fov-ra-center": _sexa(ra, True), "fov-dec-center": _sexa(dec, False), "fov-ra-hwidth": f"{hw:.4f}", "fov-dec-hwidth": f"{hw:.4f}"}
    d, err = None, None
    for attempt in range(3):                                   # JPL occasionally answers with an error page: try up to 3 times
        try:
            d = requests.get(SBIDENT_API, timeout=45, params=params).json()
            break
        except Exception as e:
            err = e
            time.sleep(1.5)
    if d is None:
        return jsonify({"error": f"NASA/JPL SB Ident did not answer ({type(err).__name__})"}), 502
    if "fields_first" not in d:
        return jsonify({"error": d.get("message", "no answer")}), 502
    f = d["fields_first"]; rows = d.get("data_first_pass") or []
    col = {k: f.index(k) for k in f}
    # The first pass is a coarse pre-selection that also returns objects far outside the box: keep only those whose
    # position actually falls inside it (measured on the sky from the box centre).
    c0 = np.array([math.cos(math.radians(dec)) * math.cos(math.radians(ra)), math.cos(math.radians(dec)) * math.sin(math.radians(ra)), math.sin(math.radians(dec))])
    east = np.array([-math.sin(math.radians(ra)), math.cos(math.radians(ra)), 0.0]); north = np.cross(c0, east)
    objs = []
    for x in rows:
        try:
            h, m, sec = (float(p) for p in x[1].split(":"))
            dd = x[2].replace("'", " ").replace('"', " ").split()
            sgn = -1 if dd[0].startswith("-") else 1
            ora = (h + m / 60 + sec / 3600) * 15
            odec = sgn * (abs(float(dd[0])) + float(dd[1]) / 60 + float(dd[2]) / 3600)
        except (ValueError, IndexError):
            continue
        u = np.array([math.cos(math.radians(odec)) * math.cos(math.radians(ora)), math.cos(math.radians(odec)) * math.sin(math.radians(ora)), math.sin(math.radians(odec))])
        if u @ c0 <= 0:
            continue
        xe, yn = (u @ east) / (u @ c0), (u @ north) / (u @ c0)          # gnomonic offsets (radians) east and north of the centre
        if abs(math.degrees(xe)) > hw or abs(math.degrees(yn)) > hw:
            continue
        try:
            v = float(x[col["Visual magnitude (V)"]])
        except (ValueError, TypeError):
            v = None
        objs.append({"name": x[0].strip(), "ra_deg": round(ora, 5), "dec_deg": round(odec, 5), "v": v,
                     "east_arcmin": round(math.degrees(xe) * 60, 2), "north_arcmin": round(math.degrees(yn) * 60, 2),
                     "rate_ra_arcsec_h": _num(x[col['RA rate ("/h)']]), "rate_dec_arcsec_h": _num(x[col['Dec rate ("/h)']])})
    objs.sort(key=lambda o: 99 if o["v"] is None else o["v"])
    out = {"count": len(objs), "objects": objs[:60], "time_utc": t.strftime("%Y-%m-%d %H:%M:%S"), "half_width_deg": hw,
           "source": "NASA/JPL Small-Body Identification API (first pass, ~1 arcmin)"}
    if len(_ident_cache) > 300:
        _ident_cache.clear()
    _ident_cache[key] = out
    return jsonify(out)


def main(port=5000):
    print(f"XARVIS running: viewer http://127.0.0.1:{port}/  ·  about page http://127.0.0.1:{port}/about")
    threading.Thread(target=load_ephemeris, daemon=True).start()       # the first run downloads the 32 MB kernel
    app.run(debug=True, use_reloader=False, port=port, threaded=True)


if __name__ == "__main__":
    main()
