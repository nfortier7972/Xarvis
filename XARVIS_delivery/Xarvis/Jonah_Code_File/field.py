"""Compare what SPHEREx sees around the target with the Gaia DR3 and CatWISE catalogs, and with JPL's list of
known solar-system objects.

  - Every source peak SPHEREx detects in a frame is matched to a catalog source (moved to that frame's date by
    its proper motion). Matched sources check the pointing against Gaia (the most precise catalog there is).
  - Peaks with NO catalog counterpart are the interesting ones. Grouped by sky position over all frames:
        uncatalogued   seen in several survey visits: a real, steady source the catalogs miss (e.g. very red)
        transient      seen in one visit (several exposures) but not in others that covered it
        single epoch   seen only at one instant (both detector arms at once at most): an asteroid passing
                       through, or a glitch -> checked with JPL's Small-Body Identification service
  - JPL SB Ident (https://ssd-api.jpl.nasa.gov/sb_ident.api) lists the known asteroids and comets in a patch of
    sky at a given time. It takes ~20 s per call, so calls run in the background and are cached.

Nothing here changes the target's measurements - it only adds context.
"""

import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

from . import config as cfg
from . import net
from .irsa import gator
from .measure import centroid, significant_peaks, survey_visits

log = logging.getLogger(__name__)

GAIA_EPOCH_MJD = 57388.0            # Gaia DR3 positions are for 2016.0
CATWISE_EPOCH_MJD = 57170.0         # CatWISE2020: ~2015.4
SB_IDENT_URL = 'https://ssd-api.jpl.nasa.gov/sb_ident.api'
FIELD_COLUMNS = {'gaia_dr3_source': 'source_id,ra,dec,pmra,pmdec,parallax,phot_g_mean_mag,phot_rp_mean_mag',
                 'catwise_2020': 'source_name,ra,dec,pmra,pmdec,w1mpro,w2mpro'}


# ----------------------------------------------------------------------------
# The catalogs
# ----------------------------------------------------------------------------

def _gator(catalog, coord, radius_arcsec):
    """One Gator cone search with the columns the field comparison needs (cached; see irsa.gator)."""
    return gator(catalog, coord, radius_arcsec, FIELD_COLUMNS[catalog])


def query_field(coord, radius=None, quiet=False):
    """Catalog sources around `coord` that SPHEREx can see: Gaia DR3 (to RP = FIELD_GAIA_MAX_RP) and CatWISE2020
    (to W1 = FIELD_W1_MAX - it adds the red / dusty sources Gaia misses). One table: source ('Gaia' / 'CatWISE'),
    name, ra, dec, pmra, pmdec (mas/yr, RA*cos(Dec)), epoch_mjd, mag (RP or W1), gaia_match (CatWISE: a Gaia
    source within 3")."""
    radius = radius if radius is not None else cfg.FIELD_RADIUS
    radius_arcsec = radius.to_value(u.arcsec)
    with ThreadPoolExecutor(2) as pool:
        gaia_job = pool.submit(_gator, 'gaia_dr3_source', coord, radius_arcsec)
        wise_job = pool.submit(_gator, 'catwise_2020', coord, radius_arcsec)
        gaia, wise = gaia_job.result(), wise_job.result()

    def column(table, name, fill=0.0):
        return np.ma.filled(np.ma.asarray(table[name], dtype=float), fill) if len(table) else np.array([])

    rp = column(gaia, 'phot_rp_mean_mag', 99.0)
    g = column(gaia, 'phot_g_mean_mag', 99.0)
    rp = np.where(rp < 90, rp, g - 0.5)                       # no RP: estimate from G
    keep = rp <= cfg.FIELD_GAIA_MAX_RP
    gaia_part = Table({'source': ['Gaia'] * int(keep.sum()), 'name': [f'Gaia DR3 {s}' for s in np.array(gaia['source_id'])[keep]],
                       'ra': column(gaia, 'ra')[keep], 'dec': column(gaia, 'dec')[keep],
                       'pmra': column(gaia, 'pmra')[keep], 'pmdec': column(gaia, 'pmdec')[keep],
                       'epoch_mjd': np.full(int(keep.sum()), GAIA_EPOCH_MJD), 'mag': rp[keep],
                       'parallax': column(gaia, 'parallax')[keep]})
    w1 = column(wise, 'w1mpro', 99.0)
    keep = w1 <= cfg.FIELD_W1_MAX
    wise_part = Table({'source': ['CatWISE'] * int(keep.sum()), 'name': [str(s) for s in np.array(wise['source_name'])[keep]],
                       'ra': column(wise, 'ra')[keep], 'dec': column(wise, 'dec')[keep],
                       'pmra': column(wise, 'pmra')[keep] * 1000, 'pmdec': column(wise, 'pmdec')[keep] * 1000,   # "/yr -> mas/yr
                       'epoch_mjd': np.full(int(keep.sum()), CATWISE_EPOCH_MJD), 'mag': w1[keep],
                       'parallax': np.zeros(int(keep.sum()))})
    # CatWISE's proper motions are only good for fast movers; ignore the small (noisy) ones
    small = np.hypot(wise_part['pmra'], wise_part['pmdec']) < 200
    wise_part['pmra'][small] = 0.0
    wise_part['pmdec'][small] = 0.0
    field = vstack([gaia_part, wise_part]) if len(wise_part) else gaia_part
    field['gaia_match'] = np.zeros(len(field), dtype=bool)
    if len(gaia_part) and len(wise_part):
        field['gaia_match'][len(gaia_part):] = _nearest_arcsec(wise_part, gaia_part) < 3.0
    if not quiet:
        log.info(f'Field catalogs within {radius.to_value(u.arcmin):.1f}\': {len(gaia_part)} Gaia DR3 (RP <= {cfg.FIELD_GAIA_MAX_RP}), '
              f'{len(wise_part)} CatWISE (W1 <= {cfg.FIELD_W1_MAX}; {int((~field["gaia_match"][len(gaia_part):]).sum())} '
              f'without a Gaia counterpart)')
    return field


def _nearest_arcsec(a, b):
    """For each source in table `a`, the distance (arcsec) to the nearest one in `b` (small fields: flat sky)."""
    dec0 = np.radians(np.mean(b['dec']))
    ax, ay = np.asarray(a['ra'], float) * np.cos(dec0), np.asarray(a['dec'], float)
    bx, by = np.asarray(b['ra'], float) * np.cos(dec0), np.asarray(b['dec'], float)
    nearest = np.empty(len(ax))
    for start in range(0, len(ax), 500):                # in chunks, to keep memory small
        d = np.hypot(ax[start:start + 500, None] - bx[None, :], ay[start:start + 500, None] - by[None, :])
        nearest[start:start + 500] = d.min(axis=1) * 3600
    return nearest


def field_positions(field, mjd):
    """RA, Dec (deg) of every field source at time `mjd`, moved along its proper motion (straight line)."""
    years = (mjd - np.asarray(field['epoch_mjd'], dtype=float)) / 365.25
    dec = np.asarray(field['dec'], dtype=float) + np.asarray(field['pmdec'], dtype=float) * years / 3.6e6
    ra = np.asarray(field['ra'], dtype=float) + (np.asarray(field['pmra'], dtype=float) * years / 3.6e6
                                                  / np.cos(np.radians(dec)))
    return ra % 360, dec


def describe_target(field, coord, mjd):
    """Print what the catalogs list at the target's position (at time `mjd`): the nearest Gaia and CatWISE source."""
    if not len(field):
        log.info('Catalogs at the target: nothing in Gaia DR3 / CatWISE within the field')
        return
    ra, dec = field_positions(field, mjd)
    separation = SkyCoord(ra, dec, unit='deg').separation(coord).arcsec
    parts = []
    for source, band in (('Gaia', 'RP'), ('CatWISE', 'W1')):
        mine = np.flatnonzero(np.asarray(field['source']) == source)
        if len(mine):
            k = mine[np.argmin(separation[mine])]
            pm = np.hypot(field['pmra'][k], field['pmdec'][k])
            parts.append(f"{field['name'][k]} {separation[k]:.1f}\" away ({band} {field['mag'][k]:.2f}"
                         + (f', proper motion {pm:.0f} mas/yr' if pm > 0 else '') + ')')
    within = int((separation <= cfg.FIELD_MATCH_PIX * cfg.PIXEL_ARCSEC).sum())
    log.info('Catalogs at the target: nearest ' + '; '.join(parts)
          + f'. {within} catalog source(s) within {cfg.FIELD_MATCH_PIX * cfg.PIXEL_ARCSEC:.0f}" (one SPHEREx resolution element)')


# ----------------------------------------------------------------------------
# One frame against the catalogs
# ----------------------------------------------------------------------------

def compare_frame(frame, field):
    """Match the source peaks SPHEREx detects in `frame` (a frame or a zoom: needs 'image', 'bad', 'wcs', 'mjd')
    to the field catalog at the frame's date.
    Returns {'peak_x', 'peak_y', 'peak_sigma', 'peak_ra', 'peak_dec', 'matched' (per peak), 'cat_x', 'cat_y',
    'cat_source', 'cat_gaia_match', 'offset_px' (per matched Gaia peak: measured - catalog, x/y)}."""
    # every real source (instrument noise) is matched; 'peak_sigma' says how far each stands out from the crowd of
    # faint field stars (0 = not at all) - only those count as unexplained sources
    xs, ys, _ = significant_peaks(frame, noise='pixel')
    fx, fy, field_sigma = significant_peaks(frame, noise='field')
    lookup = {(x, y): s for x, y, s in zip(fx, fy, field_sigma)}
    significance = np.array([lookup.get((x, y), 0.0) for x, y in zip(xs, ys)])
    ny, nx = frame['image'].shape
    edge = (xs >= 2) & (xs < nx - 2) & (ys >= 2) & (ys < ny - 2)          # half a PSF from the edge: unreliable
    xs, ys, significance = xs[edge], ys[edge], significance[edge]
    ra, dec = field_positions(field, frame['mjd'])
    cat_x, cat_y = frame['wcs'].world_to_pixel_values(ra, dec)
    inside = (cat_x > -3) & (cat_x < nx + 2) & (cat_y > -3) & (cat_y < ny + 2)
    matched = np.zeros(len(xs), dtype=bool)
    seen = np.zeros(int(inside.sum()), dtype=bool)          # catalog sources SPHEREx detects in this frame
    peak_cat = -np.ones(len(xs), dtype=int)                 # per peak: the catalog row it matched (-1: none)
    offsets = []
    if inside.any() and len(xs):
        d = np.hypot(xs[:, None] - cat_x[None, inside], ys[:, None] - cat_y[None, inside])
        nearest = np.argmin(d, axis=1)
        matched = d[np.arange(len(xs)), nearest] <= cfg.FIELD_MATCH_PIX
        seen = (d <= cfg.FIELD_MATCH_PIX).any(axis=0)
        peak_cat[matched] = np.flatnonzero(inside)[nearest[matched]]
        is_gaia = np.asarray(field['source'])[inside] == 'Gaia'
        clean_image = np.where(frame['bad'], np.nan, frame['image'])
        for i in np.flatnonzero(matched):
            j = nearest[i]
            if is_gaia[j] and significance[i] >= 10:     # well-measured sources only
                c = centroid(clean_image, frame['variance'], xs[i], ys[i]) if 'variance' in frame else None
                if c is not None:
                    offsets.append((c[0] - cat_x[inside][j], c[1] - cat_y[inside][j]))
    peak_ra, peak_dec = frame['wcs'].pixel_to_world_values(xs, ys) if len(xs) else (np.array([]), np.array([]))
    # clean = no flagged pixel within 2 px. Measured (WISE 0855 field): every "unexplained" peak next to a flagged
    # pixel was the leftover glow around a masked cosmic-ray hit / persistence / overflow - not a source.
    bad = frame['bad']
    clean = np.array([not bad[max(int(y) - 2, 0):int(y) + 3, max(int(x) - 2, 0):int(x) + 3].any()
                      for x, y in zip(xs, ys)], dtype=bool)
    return {'peak_x': xs, 'peak_y': ys, 'peak_sigma': significance, 'peak_ra': np.atleast_1d(peak_ra),
            'peak_dec': np.atleast_1d(peak_dec), 'matched': matched, 'peak_clean': clean,
            'cat_x': cat_x[inside], 'cat_y': cat_y[inside], 'cat_source': np.asarray(field['source'])[inside],
            'cat_gaia_match': np.asarray(field['gaia_match'])[inside], 'cat_seen': seen, 'cat_index': np.flatnonzero(inside), 'peak_cat': peak_cat, 'offset_px': np.array(offsets).reshape(-1, 2)}


# ----------------------------------------------------------------------------
# All frames: pointing vs Gaia, and sources no catalog knows
# ----------------------------------------------------------------------------

def field_summary(frames, field, target=None, unknowns=True):
    """Compare every frame with the field catalogs (needs the frames' full pixels: run before make_zooms).
    `target` = the target's expected (ra, dec) per frame, so the target itself isn't listed as unknown.
    Sets frame['field_unmatched'] (number of unexplained peaks) and returns
    {'offset_px', 'candidates': list of dicts (kind, ra, dec, n_frames, frames, mjd, sigma)}."""
    detections = []                                     # (frame index, ra, dec, sigma) of unmatched peaks
    offsets = []
    for i, frame in enumerate(frames):
        result = compare_frame(frame, field)
        offsets.extend(result['offset_px'])
        keep = ~result['matched'] & result['peak_clean'] & (result['peak_sigma'] >= cfg.FIELD_MIN_SIGMA)
        if target is not None:                          # the target itself isn't "unknown"
            tx, ty = frame['wcs'].world_to_pixel_values(*target[i])
            keep &= np.hypot(result['peak_x'] - tx, result['peak_y'] - ty) > cfg.FIELD_MATCH_PIX
        frame['field_unmatched'] = int(keep.sum())
        for k in np.flatnonzero(keep):
            detections.append((i, result['peak_ra'][k], result['peak_dec'][k], result['peak_sigma'][k]))

    offsets = np.array(offsets).reshape(-1, 2)
    if len(offsets):
        median = np.median(offsets, axis=0)
        log.info(f'Field vs Gaia DR3: {len(offsets)} source matches over {len(frames)} frames; SPHEREx positions agree with '
              f'Gaia (moved to each frame\'s date) to {np.hypot(*median) * cfg.PIXEL_ARCSEC:.2f}" on average '
              f'(scatter {np.hypot(*[1.4826 * np.median(np.abs(o - m)) for o, m in zip(offsets.T, median)]) * cfg.PIXEL_ARCSEC:.2f}" per source)')
    if not unknowns:                                    # (the sky map lists the unknown sources instead)
        return {'offset_px': offsets, 'candidates': []}
    candidates = _group(detections, frames)
    if candidates:
        kinds = {}
        for c in candidates:
            kinds[c['kind']] = kinds.get(c['kind'], 0) + 1
        log.info(f'Sources in no catalog (Gaia DR3 / CatWISE): ' + ', '.join(f'{n} {k}' for k, n in kinds.items())
              + '  (see field_candidates in the results; known asteroids are checked with JPL)')
    else:
        log.info('Sources in no catalog (Gaia DR3 / CatWISE): none - every source SPHEREx detects here is a known one')
    return {'offset_px': offsets, 'candidates': candidates}


def _group(detections, frames):
    """Group unmatched detections that are at the same sky position (within FIELD_MATCH_PIX) and classify them."""
    if not detections:
        return []
    index = np.array([d[0] for d in detections])
    sky = SkyCoord([d[1] for d in detections], [d[2] for d in detections], unit='deg')
    sigma = np.array([d[3] for d in detections])
    mjd = np.array([f['mjd'] for f in frames])
    visit = survey_visits(mjd)
    group = -np.ones(len(detections), dtype=int)
    for k in np.argsort(-sigma):                        # brightest first: the group centres
        if group[k] >= 0:
            continue
        close = (sky.separation(sky[k]).arcsec <= cfg.FIELD_MATCH_PIX * cfg.PIXEL_ARCSEC) & (group < 0)
        group[close] = k
    candidates = []
    for g in np.unique(group):
        members = np.flatnonzero(group == g)
        times = np.unique(np.round(mjd[index[members]], 3))
        # how many frames covered this spot at all (not just where it was found)
        covered = []
        for i, f in enumerate(frames):
            x, y = f['wcs'].world_to_pixel(sky[g])
            if 2 <= x < f['image'].shape[1] - 2 and 2 <= y < f['image'].shape[0] - 2:
                covered.append(i)
        visits = np.unique(visit[index[members]])
        visits_covered = np.unique(visit[covered])
        if len(times) == 1:
            kind = 'single epoch'
        elif len(visits) == 1 and len(visits_covered) > 1:
            kind = 'transient'
        elif len(visits) > 1:
            kind = 'uncatalogued'
        else:
            kind = 'one visit (no other visit covers it)'
        if len(times) == 1 and len(members) == 1 and sigma[g] < cfg.FIELD_MIN_SIGMA + 2:
            continue                                    # one faint peak in one frame: most likely noise
        candidates.append({'kind': kind, 'ra': float(sky[g].ra.deg), 'dec': float(sky[g].dec.deg),
                           'n_frames': len(members), 'n_covered': len(covered),
                           'frames': [int(i) for i in index[members]], 'mjd': float(mjd[index[g]]),
                           'sigma': float(sigma[g])})
    candidates.sort(key=lambda c: (c['kind'] != 'transient', c['kind'] != 'single epoch', -c['sigma']))
    return candidates


# ----------------------------------------------------------------------------
# JPL Small-Body Identification: which known asteroids / comets are in the field
# ----------------------------------------------------------------------------

_ident_lock = threading.Lock()
_ident_results = {}

def known_small_bodies(mjd, ra_deg, dec_deg, half_width_deg=None):
    """Known asteroids and comets within a box around (ra, dec) at time `mjd`, from JPL SB Ident (geocentric -
    SPHEREx is in low Earth orbit, a few arcseconds of parallax at most for near-Earth objects). Takes ~20 s;
    cached on disk. Returns a list of {'name', 'ra', 'dec', 'vmag', 'offset_arcsec'}."""
    half = half_width_deg if half_width_deg is not None else cfg.FIELD_RADIUS.to_value(u.deg)
    time = Time(mjd, format='mjd').utc.strftime('%Y-%m-%d_%H:%M:%S')
    centre = SkyCoord(ra_deg, dec_deg, unit='deg')
    params = {'mpc-code': '500', 'obs-time': time,
              'fov-ra-center': centre.ra.to_string(unit=u.hourangle, sep='-', precision=1, pad=True),
              'fov-dec-center': centre.dec.to_string(sep='-', precision=0, alwayssign=True, pad=True),
              'fov-ra-hwidth': f'{half:.4f}', 'fov-dec-hwidth': f'{half:.4f}', 'two-pass': 'true',
              'suppress-first-pass': 'true', 'req-elem': 'false', 'vmag-lim': cfg.SB_IDENT_VMAG_LIMIT}
    # JPL allows 20 SB Ident requests per 5 minutes: net.py spaces them (service 'jpl-sb-ident')
    content = net.get(SB_IDENT_URL, params, service='jpl-sb-ident', label='JPL SB Ident',
                      cache_kind='sb_ident', timeout=120)
    data = json.loads(content)
    fields = data.get('fields_second') or data.get('fields_first') or []
    rows = data.get('data_second_pass') or data.get('data_first_pass') or []
    bodies = []
    for row in rows:
        entry = dict(zip(fields, row))
        try:
            dec_text = entry["Astrometric Dec (dd mm'ss\")"].replace("'", ' ').replace('"', ' ').split()   # +05 34'39.3"
            sky = SkyCoord(entry['Astrometric RA (hh:mm:ss)'], ' '.join(dec_text), unit=(u.hourangle, u.deg))
        except (KeyError, ValueError):
            continue
        def number(key):
            try:
                return float(entry.get(key))
            except (TypeError, ValueError):
                return np.nan
        bodies.append({'name': entry.get('Object name', '?'), 'ra': float(sky.ra.deg), 'dec': float(sky.dec.deg),
                       'vmag': number('Visual magnitude (V)'),
                       'ra_rate': number('RA rate ("/h)'), 'dec_rate': number('Dec rate ("/h)'),   # "/hour, RA*cos(Dec)
                       'offset_arcsec': float(sky.separation(centre).arcsec), 'mjd': float(mjd)})
    return bodies


def identify_async(key, mjd, ra_deg, dec_deg, half_width_deg=None, on_done=None):
    """known_small_bodies in a background thread (it takes ~20 s). The answer is kept under `key`: read it with
    identified(key) - None while it's still running. on_done(bodies) is called from that thread when it finishes."""
    with _ident_lock:
        if key in _ident_results:
            return
        _ident_results[key] = None

    def work():
        try:
            bodies = known_small_bodies(mjd, ra_deg, dec_deg, half_width_deg)
        except Exception as e:                          # never let a lookup crash the viewer
            log.info(f'JPL SB Ident failed: {e}')
            bodies = []
        with _ident_lock:
            _ident_results[key] = bodies
        if on_done:
            on_done(bodies)
    threading.Thread(target=work, daemon=True).start()


def identified(key):
    with _ident_lock:
        return _ident_results.get(key)


def identify_candidates(candidates, max_calls=None):
    """For 'single epoch' / 'transient' candidates: ask JPL which known asteroid or comet was there (in parallel,
    at most `max_calls` lookups). Adds 'known_body' (name, or None) to each candidate checked."""
    max_calls = cfg.SB_IDENT_MAX_CALLS if max_calls is None else max_calls
    todo = [c for c in candidates if c['kind'] in ('single epoch', 'transient')][:max_calls]
    if not todo:
        return candidates
    log.info(f'Asking JPL which known asteroids/comets were at {len(todo)} unexplained source(s) (~20 s)...')
    with ThreadPoolExecutor(len(todo)) as pool:
        answers = list(pool.map(lambda c: _safe_bodies(c['mjd'], c['ra'], c['dec']), todo))
    for candidate, bodies in zip(todo, answers):
        close = [b for b in bodies or [] if b['offset_arcsec'] <= cfg.SB_IDENT_MATCH_ARCSEC]
        candidate['known_body'] = min(close, key=lambda b: b['offset_arcsec'])['name'] if close else None
        log.info(f"  {candidate['kind']} at RA {candidate['ra']:.5f} Dec {candidate['dec']:+.5f} "
              f"(T{candidate['frames'][0] + 1}, {candidate['sigma']:.0f} sigma): "
              + (f"known object {candidate['known_body']}" if candidate['known_body'] else
                 'no known asteroid or comet there' + (' (JPL lookup failed)' if bodies is None else '')))
    return candidates


def _safe_bodies(mjd, ra, dec):
    try:
        return known_small_bodies(mjd, ra, dec, half_width_deg=cfg.SB_IDENT_MATCH_ARCSEC / 3600 * 2)
    except Exception as e:
        log.info(f'JPL SB Ident failed: {e}')
        return None
