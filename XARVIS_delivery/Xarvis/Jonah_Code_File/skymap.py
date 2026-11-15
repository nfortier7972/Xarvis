"""Map EVERY object in the image around a sky position, name what is known, and find what isn't.

Each source SPHEREx detects in each frame is explained, in this order, by:
  stars          ESA Hipparcos (bright stars; HIP numbers) with their common names from Stellarium's "modern"
                 sky culture, ESA Gaia DR3 (everything down to RP ~18), and CatWISE2020 (red / dusty sources
                 Gaia misses) - all moved to the frame's date by their proper motions
  planets / Moon NASA/JPL DE440s planetary ephemeris
  asteroids,     NASA/JPL SB Ident API (which known small bodies were in the field at that time), NASA/JPL SBDB
  comets         API (what each one is) and NASA/JPL Horizons (its exact position as seen FROM SPHEREx)
  artifacts      the SPHEREx FLAGS plane: per the IPAC SPHEREx Explanatory Supplement (v2.1), these flags mark
                 light that is not a source - ghosts, crosstalk, persistence, cosmic rays, saturation spill...
Whatever is left is a NEW OBJECT - not in any of these. Before it is labelled, it is classified from its whole
history in the data (see classify_new):
  static      seen at the same place in several survey visits: a steady source missing from the catalogs
  variable    same place, several visits, but missing in some visits that covered it
  transient   same place, seen in ONE visit only although other visits covered it: appeared, then gone
              (a very slow mover - e.g. a distant solar-system object - can also look like this)
  moving      in different places at different times, on a straight line at a steady rate: an unknown
              asteroid or comet candidate
  single      seen at one instant only: a fast mover, a flare or an unflagged glitch - can't tell
Every detection of every new object (coordinates and UTC time) is written to results/<target>_new_objects.csv.
"""

import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body, solar_system_ephemeris
from astropy.table import Table
from astropy.time import Time

from . import cache
from . import config as cfg
from .field import (GAIA_EPOCH_MJD, compare_frame, field_positions, known_small_bodies, query_field)
from . import net
from .measure import flag_bits, survey_visits

log = logging.getLogger(__name__)

HIP_EPOCH_MJD = 48348.5625                  # Hipparcos (new reduction) positions are for J1991.25
VIZIER_URL = 'https://vizier.cfa.harvard.edu/viz-bin/asu-tsv'
STELLARIUM_URL = 'https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern/index.json'
SBDB_URL = 'https://ssd-api.jpl.nasa.gov/sbdb.api'
PLANETS = ('mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'moon')
# SPHEREx flags that mark light which is NOT an astronomical source (IPAC SPHEREx Explanatory Supplement v2.1)
ARTIFACT_FLAGS = ('GHOST', 'GHOST_FPA', 'GHOST_EXT', 'CROSSTALK', 'PERSIST', 'STREAK', 'SNOWBALL', 'TRANSIENT',
                  'OUTLIER', 'OVERFLOW', 'SUR_ERROR', 'BLOOM', 'HOT', 'COLD', 'NONFUNC', 'MISSING_DATA', 'NONLINEAR')


def _get(kind, key_parts, url, params=None, label='request'):
    """GET through net.py (the service's retries and limits), cached on disk under `kind`."""
    service = 'jpl' if 'jpl.nasa.gov' in url else 'github' if 'github' in url else 'cds'
    return net.get(url, params, service=service, label=label, cache_kind=kind, timeout=60,
                   cache_key=cache.make_key(url, *sorted((params or {}).items()), *key_parts))


# ----------------------------------------------------------------------------
# Stars: Hipparcos + Stellarium names + Gaia DR3 + CatWISE
# ----------------------------------------------------------------------------

def star_names():
    """{HIP number: common name} from Stellarium's 'modern' sky culture (e.g. 32349: 'Sirius')."""
    data = json.loads(_get('reference', ['v1'], STELLARIUM_URL, label='Stellarium star names'))
    names = {}
    for key, entries in data.get('common_names', {}).items():
        match = re.match(r'HIP (\d+)$', key)
        if match and entries:
            names[int(match.group(1))] = entries[0].get('english') or entries[0].get('native')
    return names


def hipparcos(coord, radius):
    """Hipparcos stars (new reduction, VizieR I/311) within `radius`: hip, ra, dec (J1991.25), pmra, pmdec,
    parallax (mas), hp (Hipparcos magnitude)."""
    params = {'-source': 'I/311/hip2', '-c': f'{coord.ra.deg:.6f} {coord.dec.deg:+.6f}',
              '-c.rs': f'{radius.to_value(u.arcsec):.0f}', '-out': 'HIP,RArad,DErad,pmRA,pmDE,Plx,Hpmag',
              '-out.max': '500'}
    text = _get('catalogs', [], VIZIER_URL, params, 'Hipparcos').decode()
    rows = []
    for line in text.splitlines():
        parts = line.split('\t')
        if len(parts) == 7 and parts[0].strip().isdigit():
            try:
                rows.append([int(parts[0])] + [float(p) for p in parts[1:]])
            except ValueError:
                continue
    return Table(rows=rows or None, names=('hip', 'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'hp'))


def catalog_layer(coord, radius=None):
    """All catalog stars around `coord`, one row each, with the best name: Stellarium name (+HIP), HIP number,
    Gaia DR3 id or CatWISE name. Columns as field.query_field, plus 'label' and 'hip' (0 = none)."""
    radius = radius if radius is not None else cfg.FIELD_RADIUS
    field = query_field(coord, radius, quiet=True)
    try:
        hip, names = hipparcos(coord, radius), star_names()
    except RuntimeError as e:
        log.info(f'  (Hipparcos / Stellarium names not available: {e})')
        hip, names = Table(names=('hip', 'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'hp')), {}
    rows = [dict(zip(field.colnames, [r[c] for c in field.colnames]), label=str(r['name']), hip=0) for r in field]
    gaia = [k for k, r in enumerate(rows) if r['source'] == 'Gaia']
    for star in hip:
        # Hipparcos position moved to the Gaia epoch, then the same star found in Gaia (if Gaia has it)
        years = (GAIA_EPOCH_MJD - HIP_EPOCH_MJD) / 365.25
        dec = star['dec'] + star['pmdec'] * years / 3.6e6
        ra = star['ra'] + star['pmra'] * years / 3.6e6 / np.cos(np.radians(dec))
        name = names.get(int(star['hip']))
        label = f"{name} (HIP {star['hip']})" if name else f"HIP {star['hip']}"
        best, best_sep = None, 1.5
        for k in gaia:
            sep = np.hypot((rows[k]['ra'] - ra) * np.cos(np.radians(dec)), rows[k]['dec'] - dec) * 3600
            if sep < best_sep:
                best, best_sep = k, sep
        if best is not None:
            rows[best].update(label=label, hip=int(star['hip']))
        else:                                       # too bright for Gaia: keep Hipparcos's own entry
            rows.append({'source': 'Hipparcos', 'name': f"HIP {star['hip']}", 'ra': float(star['ra']),
                         'dec': float(star['dec']), 'pmra': float(star['pmra']), 'pmdec': float(star['pmdec']),
                         'epoch_mjd': HIP_EPOCH_MJD, 'mag': float(star['hp']), 'parallax': float(star['parallax']),
                         'gaia_match': False, 'label': label, 'hip': int(star['hip'])})
    if not rows:
        field['label'], field['hip'] = np.array([], dtype=str), np.array([], dtype=int)
    table = Table(rows=rows) if rows else field
    n_named = sum(1 for r in rows if r['hip'] and '(' in r['label'])
    log.info(f"Star catalogs: {sum(r['source'] == 'Gaia' for r in rows)} Gaia DR3, {len(hip)} Hipparcos "
          f"({n_named} with a common name), {sum(r['source'] == 'CatWISE' and not r['gaia_match'] for r in rows)} "
          f"CatWISE-only sources within {radius.to_value(u.arcmin):.1f}'")
    return table


def star_to_track(row):
    """A catalog row as a 'star' (stars.positions_at format: J2000 position, motion, parallax) to track."""
    years = (51544.5 - float(row['epoch_mjd'])) / 365.25                # back to J2000
    dec = float(row['dec']) + float(row['pmdec']) * years / 3.6e6
    ra = float(row['ra']) + float(row['pmra']) * years / 3.6e6 / np.cos(np.radians(dec))
    plx = float(row['parallax']) if np.isfinite(row['parallax']) and row['parallax'] > 0 else None
    return {'name': str(row['label']), 'object_type': f"{row['source']} star", 'query': str(row['name']),
            'ra': ra % 360, 'dec': dec, 'pmra': float(row['pmra']), 'pmdec': float(row['pmdec']),
            'parallax': plx, 'rv': None, 'magnitudes': {}, 'spectral_type': '?'}


# ----------------------------------------------------------------------------
# Planets and the Moon (DE440s)
# ----------------------------------------------------------------------------

def planets_in_view(coord, times, radius_deg):
    """Planets / the Moon within the field at any of `times` (MJD), from JPL DE440s (geocentric: SPHEREx's low
    Earth orbit shifts the Moon by up to ~1 degree and planets by seconds of arc). Returns
    {planet: [(mjd, ra, dec), ...]} for the times it is within radius (+ 1.5 degree for the Moon)."""
    found = {}
    try:
        with solar_system_ephemeris.set('de440s'):
            t = Time(np.asarray(times, dtype=float), format='mjd')
            for planet in PLANETS:
                body = get_body(planet, t)
                sep = SkyCoord(body.ra, body.dec).separation(coord).deg
                near = sep <= radius_deg + (1.5 if planet == 'moon' else 0.01)
                if near.any():
                    found[planet] = [(float(m), float(r), float(d))
                                     for m, r, d in zip(t.mjd[near], body.ra.deg[near], body.dec.deg[near])]
    except Exception as e:                          # e.g. no internet for the one-time DE440s download
        log.info(f'  (DE440s planetary ephemeris not available: {e})')
    return found


# ----------------------------------------------------------------------------
# Asteroids and comets: SB Ident -> SBDB -> Horizons (from SPHEREx)
# ----------------------------------------------------------------------------

def sbdb_details(name):
    """What a small body is, from the JPL SBDB API: {'des', 'fullname', 'kind', 'orbit_class'} (or None)."""
    number = re.match(r'^(\d+)\s', name)
    comet = re.match(r'^(\d+[PCDXI]|[PCDXI])/', name)
    paren = re.search(r'\(([^)]+)\)', name)
    query = (name.split('(')[0].strip() if comet else number.group(1) if number else
             paren.group(1) if paren else name.strip())
    try:
        data = json.loads(_get('sbdb', [], SBDB_URL, {'sstr': query}, 'JPL SBDB'))
    except RuntimeError:
        return None
    obj = data.get('object')
    if not obj:
        return None
    kind = {'an': 'asteroid (numbered)', 'au': 'asteroid', 'cn': 'comet (numbered)', 'cu': 'comet'}.get(obj.get('kind'), '?')
    return {'des': obj.get('des', query), 'fullname': obj.get('fullname', name), 'kind': kind,
            'orbit_class': (obj.get('orbit_class') or {}).get('name', '?')}


def time_groups(mjd, max_gap=0.25, max_span=1.0):
    """Frame indices grouped into bursts of observations (gap < max_gap days, span < max_span days)."""
    order = np.argsort(mjd)
    groups, current = [], [order[0]]
    for i in order[1:]:
        if mjd[i] - mjd[current[-1]] > max_gap or mjd[i] - mjd[current[0]] > max_span:
            groups.append(current)
            current = []
        current.append(i)
    groups.append(current)
    return [np.array(g) for g in groups]


def small_bodies(frames, coord, radius_deg, priority=()):
    """Known asteroids / comets in the field at the frames' times.

    One JPL SB Ident call per burst of observations (at its middle time, widened by how far a fast main-belt
    asteroid moves within the burst); each body's position at every frame of the burst follows from its motion
    rates. Bodies that come within the field get their SBDB details and their exact position as seen from
    SPHEREx (Horizons, observer = the spacecraft). Each call takes ~20 s, so by default only ONE is made
    (SKYMAP_MAX_SB_CALLS): for the burst with the most unexplained detections (frame indices in `priority`) -
    the only place a known asteroid changes the answer. If it finds nothing, the run moves on; the viewer's J key
    checks any other frame on demand. Returns (bodies, checked frame indices). Each body: name, des, kind,
    orbit_class, vmag, positions {frame index: (ra, dec)}, precise (from Horizons)."""
    mjd = np.array([f['mjd'] for f in frames])
    groups = time_groups(mjd, max_gap=1.0, max_span=cfg.SKYMAP_BURST_DAYS)
    priority = set(priority)
    groups.sort(key=lambda g: (-len(priority.intersection(g.tolist())), mjd[g[0]]))   # most unexplained first
    todo = groups[:cfg.SKYMAP_MAX_SB_CALLS]
    if not todo:
        return [], set()
    log.info(f'Asking JPL SB Ident which known asteroids / comets were in the field ({len(todo)} call'
          f'{"s" if len(todo) > 1 else ""} covering {sum(len(g) for g in todo)} of {len(frames)} frames, ~20 s; '
          f'cached for re-runs)...')

    def ask(group):
        mid = float(np.median(mjd[group]))
        half = radius_deg + cfg.SKYMAP_MAX_RATE_DEG_PER_DAY * float(np.max(np.abs(mjd[group] - mid)))
        try:
            return known_small_bodies(mid, coord.ra.deg, coord.dec.deg, half_width_deg=min(half, 2.0))
        except Exception as e:
            log.info(f'  JPL SB Ident failed for MJD {mid:.2f}: {e}')
            return None

    with ThreadPoolExecutor(2) as pool:
        answers = list(pool.map(ask, todo))
    bodies, checked = {}, set()
    for group, answer in zip(todo, answers):
        if answer is None:
            continue
        checked.update(int(i) for i in group)
        for b in answer:
            hours = (mjd[group] - b['mjd']) * 24
            rate_ra = np.nan_to_num(b['ra_rate'])
            rate_dec = np.nan_to_num(b['dec_rate'])
            dec = b['dec'] + rate_dec * hours / 3600
            ra = b['ra'] + rate_ra * hours / 3600 / np.cos(np.radians(dec))
            sep = SkyCoord(ra, dec, unit='deg').separation(coord).deg
            inside = sep <= radius_deg + 0.02          # (margin: straight-line motion over a burst)
            if not inside.any():
                continue
            body = bodies.setdefault(b['name'], {'name': b['name'], 'vmag': b['vmag'], 'positions': {}, 'precise': {}})
            for i, r, d in zip(group[inside], ra[inside], dec[inside]):
                body['positions'][int(i)] = (float(r % 360), float(d))
                body['precise'][int(i)] = False
    for body in bodies.values():
        body.update(sbdb_details(body['name']) or {'des': body['name'], 'fullname': body['name'], 'kind': '?',
                                                   'orbit_class': '?'})
        _refine_with_horizons(body, mjd)
    if bodies:
        log.info('Known small bodies in the field: ' + '; '.join(
            f"{b['fullname']} ({b['kind']}, {b['orbit_class']}, V {b['vmag']:.1f}) in {len(b['positions'])} frames"
            for b in bodies.values()))
    else:
        log.info('Known small bodies in the field: none (at the times checked)')
    return list(bodies.values()), checked


def _refine_with_horizons(body, mjd):
    """Exact positions as seen from SPHEREx (JPL Horizons) for the frames a body is in; keeps the SB Ident
    estimate where Horizons can't answer."""
    from .moving import EPOCHS_PER_REQUEST, JD_MINUS_MJD, SPHEREX_OBSERVER, fetch_ephemeris
    frames = sorted(body['positions'])
    epochs, back = np.unique(np.round(mjd[frames], 6), return_inverse=True)
    try:
        ra, dec = [], []
        for start in range(0, len(epochs), EPOCHS_PER_REQUEST):
            table = fetch_ephemeris(body['des'], list(epochs[start:start + EPOCHS_PER_REQUEST] + JD_MINUS_MJD),
                                    SPHEREX_OBSERVER)
            ra += list(table['RA'])
            dec += list(table['DEC'])
    except Exception as e:
        log.info(f"  (Horizons position for {body['name']} not available, using JPL SB Ident's estimate: "
              f"{str(e).splitlines()[0][:80] if str(e) else type(e).__name__})")
        return
    for k, i in enumerate(frames):
        body['positions'][i] = (float(ra[back[k]]), float(dec[back[k]]))
        body['precise'][i] = True


# ----------------------------------------------------------------------------
# Every frame: what each detected source is
# ----------------------------------------------------------------------------

def _artifact_flags(frame, x, y, radius=2):
    """Names of the artifact flags (ARTIFACT_FLAGS) set within `radius` px of (x, y) (0 = that pixel only)."""
    header = frame.get('flags_header')
    if header is None or 'flags' not in frame:
        return []
    x, y = int(round(float(x))), int(round(float(y)))
    box = frame['flags'][max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
    return [name for name in ARTIFACT_FLAGS if flag_bits(header, {name}) and (box & flag_bits(header, {name})).any()]


def map_frames(frames, catalog, bodies, planets):
    """Identify every detected source in every frame (needs the full pixels). Stores frame['map'] - a list of
    entries {kind, label, ra, dec, detected, ...} in sky coordinates, so it can be drawn on any view - and
    returns the unexplained detections [(frame index, ra, dec, significance, mover flags), ...] for classify_new."""
    unexplained = []
    labels = np.asarray(catalog['label'])
    for i, frame in enumerate(frames):
        result = compare_frame(frame, catalog)
        entries = []
        ny, nx = frame['image'].shape
        # catalog stars this frame detects, plus every named / Hipparcos star in view even if not detected
        ra, dec = field_positions(catalog, frame['mjd'])
        cx, cy = frame['wcs'].world_to_pixel_values(ra, dec)
        in_view = (cx > -0.5) & (cx < nx - 0.5) & (cy > -0.5) & (cy < ny - 0.5)
        seen = np.zeros(len(catalog), dtype=bool)
        seen[result['cat_index'][result['cat_seen']]] = True
        for k in np.flatnonzero(in_view & (seen | (np.asarray(catalog['hip']) > 0))):
            row = catalog[k]
            if row['source'] == 'CatWISE' and row['gaia_match']:
                continue                                  # the same star as a Gaia entry
            entries.append({'kind': 'star', 'source': str(row['source']), 'label': str(labels[k]),
                            'ra': float(ra[k]), 'dec': float(dec[k]), 'detected': bool(seen[k]),
                            'named': bool(row['hip']), 'mag': float(row['mag']), 'catalog_row': int(k)})
        # known solar-system objects at this frame's time
        used = np.zeros(len(result['peak_x']), dtype=bool)
        for body in bodies:
            if i not in body['positions']:
                continue
            bra, bdec = body['positions'][i]
            bx, by = frame['wcs'].world_to_pixel_values(bra, bdec)
            limit = cfg.FIELD_MATCH_PIX if body['precise'].get(i) else cfg.SB_IDENT_MATCH_ARCSEC / cfg.PIXEL_ARCSEC
            d = np.hypot(result['peak_x'] - bx, result['peak_y'] - by)
            hit = d <= limit
            used |= hit
            entries.append({'kind': 'small body', 'label': body['fullname'], 'ra': bra, 'dec': bdec,
                            'detected': bool(hit.any()), 'body': body['des'], 'info': f"{body['kind']}, {body['orbit_class']}"})
        for planet, times in planets.items():
            for m, pra, pdec in times:
                if abs(m - frame['mjd']) < 1e-4:
                    entries.append({'kind': 'planet', 'label': planet.capitalize(), 'ra': pra, 'dec': pdec,
                                    'detected': False})
        # detected sources nothing above explains
        found_here = []
        for k in np.flatnonzero(~result['matched'] & ~used):
            x, y = result['peak_x'][k], result['peak_y'][k]
            if not result['peak_clean'][k]:
                flags = _artifact_flags(frame, x, y)
                source = _real_source(frame, x, y)
                if source is not None:                    # flagged pixels nearby, but shaped like a real source
                    mra, mdec, msigma, mover_flags = source
                    if all(np.hypot((mra - r) * np.cos(np.radians(mdec)), mdec - d) * 3600 > cfg.FIELD_MATCH_PIX * cfg.PIXEL_ARCSEC
                           for r, d in found_here):
                        found_here.append((mra, mdec))
                        unexplained.append((i, mra, mdec, msigma, mover_flags))
                    continue
                entries.append({'kind': 'artifact', 'label': 'artifact: ' + (', '.join(flags[:2]) or 'flagged pixels'),
                                'ra': float(result['peak_ra'][k]), 'dec': float(result['peak_dec'][k]), 'detected': True})
            elif result['peak_sigma'][k] >= cfg.FIELD_MIN_SIGMA:
                found_here.append((float(result['peak_ra'][k]), float(result['peak_dec'][k])))
                unexplained.append((i, float(result['peak_ra'][k]), float(result['peak_dec'][k]),
                                    float(result['peak_sigma'][k]), False))
        frame['map'] = entries
    return unexplained


MOVER_FLAGS = {'OUTLIER', 'TRANSIENT'}      # what SPHEREx's pipeline puts on a source that moves during the exposure
# light that LOOKS like a star but is a copy of another one (Explanatory Supplement): on the peak itself = artifact
COPY_FLAGS = ('GHOST', 'GHOST_FPA', 'GHOST_EXT', 'CROSSTALK', 'PERSIST')


def _real_source(frame, x, y):
    """Is a peak with flagged pixels nearby still a real source? Flags NEAR a bright source don't make it an artifact
    (measured: Eros at 500-1000 sigma had a dead pixel / crosstalk flag 2 px away and was wrongly dismissed). So:
      - its brightest pixel carries a ghost / crosstalk / persistence flag, or any other defect flag except
        OUTLIER / TRANSIENT -> artifact (None);
      - otherwise it must be shaped like a real source - at least 4 of its 8 neighbours lit (> 3 sigma) and not a
        lone spike (SPIKE_NEIGHBOR_RATIO, as in measure.bad_pixel_mask) - else a cosmic ray -> None.
    OUTLIER / TRANSIENT on it is SPHEREx's sign of a source changing during the exposure, e.g. moving (measured on
    Eros: 204 of 353 frames). Returns (ra, dec, significance, has OUTLIER/TRANSIENT) or None."""
    from .measure import centroid, robust_scatter
    image = frame['image']
    ny, nx = image.shape
    x0, x1, y0, y1 = max(int(x) - 2, 0), min(int(x) + 3, nx), max(int(y) - 2, 0), min(int(y) + 3, ny)
    box = np.where(np.isfinite(image[y0:y1, x0:x1]), image[y0:y1, x0:x1], -np.inf)
    by, bx = np.unravel_index(np.argmax(box), box.shape)
    cx, cy = x0 + bx, y0 + by
    if not (1 <= cx < nx - 1 and 1 <= cy < ny - 1):
        return None
    finite = image[np.isfinite(image)]
    sky, noise = np.median(finite), max(robust_scatter(finite), 1e-12)
    centre = image[cy, cx] - sky
    around = np.delete((image[cy - 1:cy + 2, cx - 1:cx + 2] - sky).ravel(), 4)
    around = around[np.isfinite(around)]
    if centre < cfg.FIELD_MIN_SIGMA * noise or len(around) < 6:
        return None
    if (around > 3 * noise).sum() < 4 or centre > cfg.SPIKE_NEIGHBOR_RATIO * max(around.max(), noise):
        return None                                      # a lone spike: cosmic ray
    on_peak = _artifact_flags(frame, cx, cy, radius=0)
    if any(f not in MOVER_FLAGS for f in on_peak):
        return None                                      # the peak pixel itself is a ghost / crosstalk / defect
    near = _artifact_flags(frame, cx, cy, radius=1)
    c = centroid(image, frame['variance'], cx, cy)
    if c is None:
        return None
    ra, dec = frame['wcs'].pixel_to_world_values(c[0], c[1])
    return float(ra), float(dec), float(centre / noise), bool(set(near) & MOVER_FLAGS)


# ----------------------------------------------------------------------------
# New objects: classify BEFORE labelling
# ----------------------------------------------------------------------------

def _visits(mjd):
    """Survey visit number of every frame (measure.survey_visits: the one definition)."""
    return survey_visits(mjd)


def _covering(frames, ra, dec, margin=2):
    """Indices of the frames whose image contains (ra, dec), at least `margin` px from the edge."""
    out = []
    for i, f in enumerate(frames):
        x, y = f['wcs'].world_to_pixel_values(ra, dec)
        if margin <= x < f['image'].shape[1] - margin and margin <= y < f['image'].shape[0] - margin:
            out.append(i)
    return out


def classify_new(frames, unexplained, checked=None):
    """Group the unexplained detections into objects and decide what each is (static / variable / transient /
    moving / single - see the module description). Returns a list of new-object dicts (id, kind, description,
    ra, dec, detections [...], track) and adds a 'new' entry to frame['map'] for every detection."""
    if not unexplained:
        return []
    mjd = np.array([f['mjd'] for f in frames])
    visit = _visits(mjd)
    det = [{'frame': i, 'ra': r, 'dec': d, 'sigma': s, 'mjd': mjd[i], 'mover_flags': f}
           for i, r, d, s, f in unexplained]
    sky = SkyCoord([d['ra'] for d in det], [d['dec'] for d in det], unit='deg')
    match = cfg.FIELD_MATCH_PIX * cfg.PIXEL_ARCSEC
    group = -np.ones(len(det), dtype=int)
    for k in np.argsort([-d['sigma'] for d in det]):              # same place: one object
        if group[k] < 0:
            group[(sky.separation(sky[k]).arcsec <= match) & (group < 0)] = k
    objects, singles = [], []
    for g in np.unique(group):
        members = [det[k] for k in np.flatnonzero(group == g)]
        times = np.unique(np.round([m['mjd'] for m in members], 3))
        if len(times) < 2:
            singles.extend(members)
            continue
        ra = float(np.median([m['ra'] for m in members]))
        dec = float(np.median([m['dec'] for m in members]))
        covered = _covering(frames, ra, dec)
        seen_visits = sorted({int(visit[m['frame']]) for m in members})
        covered_visits = sorted({int(visit[i]) for i in covered})
        if len(seen_visits) == 1:
            kind = 'transient' if len(covered_visits) > 1 else 'one visit only'
        else:
            kind = 'static' if seen_visits == covered_visits else 'variable'
        depth = None
        if kind in ('transient', 'variable'):        # missing in some visits: were they deep enough to see it?
            depth = _depth_check(frames, ra, dec, covered, visit, set(seen_visits))
            kind = depth['kind']
        objects.append({'kind': kind, 'ra': ra, 'dec': dec, 'detections': members, 'n_covered': len(covered),
                        'visits_seen': seen_visits, 'visits_covered': covered_visits, 'depth': depth})
    objects += _link_movers(singles, frames, visit)
    linked = {id(m) for o in objects if o['kind'] == 'moving' for m in o['detections']}
    group_of = {id(det[k]): int(group[k]) for k in range(len(det))}
    leftovers = {}                                                # seen at one instant: per place (both arms together)
    for m in singles:
        if id(m) not in linked:
            leftovers.setdefault(group_of[id(m)], []).append(m)
    glitches = 0
    for members in leftovers.values():
        best = max(members, key=lambda m: m['sigma'])
        if best['sigma'] < cfg.FIELD_MIN_SIGMA + 2:               # one faint peak at one instant: most likely noise
            continue
        arms = {frames[m['frame']]['detector'] <= 3 for m in members}
        if len(arms) == 1:
            # SPHEREx's other detector arm saw the same sky at the same instant: a real object shows up there too,
            # a cosmic-ray hit doesn't (it hits one detector). Seen by one arm while the other covered it: a glitch.
            short = frames[best['frame']]['detector'] <= 3
            other_arm = [i for i in _covering(frames, best['ra'], best['dec'])
                         if abs(mjd[i] - best['mjd']) * 86400 < 5 and (frames[i]['detector'] <= 3) != short]
            if other_arm:
                glitches += 1
                continue
        objects.append({'kind': 'single', 'ra': best['ra'], 'dec': best['dec'], 'detections': members,
                        'both_arms': len(arms) == 2, 'n_covered': len(_covering(frames, best['ra'], best['dec'])),
                        'visits_seen': [int(visit[best['frame']])], 'visits_covered': []})
    if glitches:
        log.info(f'  ({glitches} one-instant detection(s) dropped as probable cosmic rays: seen by one detector arm only, '
              f'while the other arm imaged the same spot at the same instant and saw nothing)')
    for obj in objects:
        obj['jpl_checked'] = all(m['frame'] in (checked or ()) for m in obj['detections'])

    order = {'moving': 0, 'transient': 1, 'variable': 2, 'single': 3, 'too shallow': 4, 'static': 5, 'one visit only': 6}
    objects.sort(key=lambda o: (order.get(o['kind'], 9), -max(m['sigma'] for m in o['detections'])))
    for n, obj in enumerate(objects, 1):
        obj['id'] = f'NEW-{n}'
        obj['description'] = _describe(obj)
        if obj['kind'] == 'moving':
            t0, t1 = obj['fit']['mjd'].min(), obj['fit']['mjd'].max()
            path_t = np.array([t0 - 0.5, t1 + 0.5])
            ra, dec = _linear(obj['fit'], path_t)
            obj['track'] = {'kind': 'path', 'name': f"{obj['id']} (unknown mover)", 'mjd': path_t.tolist(),
                            'ra': ra.tolist(), 'dec': dec.tolist()}
        else:
            obj['track'] = {'kind': 'coords', 'coords': (obj['ra'], obj['dec']), 'name': obj['id']}
        for m in obj['detections']:
            frames[m['frame']].setdefault('map', []).append(
                {'kind': 'new', 'label': f"{obj['id']} ({obj['kind']})", 'ra': m['ra'], 'dec': m['dec'],
                 'detected': True, 'new_id': obj['id'], 'info': obj['description']})
    return objects


def _linear(fit, mjd):
    """Position on a fitted straight-line path at time(s) mjd."""
    dt = np.asarray(mjd, dtype=float) - fit['t0']
    dec = fit['dec0'] + fit['rate_dec'] * dt
    ra = fit['ra0'] + fit['rate_ra'] * dt / np.cos(np.radians(dec))
    return ra % 360, dec


def _depth_check(frames, ra, dec, covered, visit, seen_visits):
    """Forced photometry at (ra, dec) in every frame that covered it. A source missing from some visits is only
    'transient' if those visits were deep enough to see it: its brightness in the visits it was seen must be
    >= 5x the noise of the other visits combined, and the other visits must measure nothing there (< 3 sigma).
    If they measure something after all it is there but fainter ('variable', or 'static' when about as bright);
    if they were too shallow: 'too shallow' (can't tell). (The visits see it at different wavelengths, so
    'about as bright' is judged loosely.) Returns {'kind', 'seen_mjy', 'other_mjy', 'other_err', 'depth_sigma',
    'other_text'}."""
    from .measure import aperture_flux
    seen, other = [], []
    for i in covered:
        x, y = (float(v) for v in frames[i]['wcs'].world_to_pixel_values(ra, dec))
        flux, err, n_bad = aperture_flux(frames[i], x, y)
        if n_bad == 0 and np.isfinite(flux) and np.isfinite(err) and err > 0:
            (seen if visit[i] in seen_visits else other).append((flux, err))
    if not seen or not other:
        return {'kind': 'one visit only' if len(seen_visits) == 1 else 'variable', 'depth_sigma': 0.0,
                'other_text': 'no usable measurement'}
    seen_flux = float(np.median([f for f, _ in seen]))
    weights = np.array([1 / e ** 2 for _, e in other])
    other_flux = float(np.sum(weights * np.array([f for f, _ in other])) / weights.sum())
    other_err = float(1 / np.sqrt(weights.sum()))
    depth = seen_flux / other_err if other_err > 0 else 0.0
    result = {'seen_mjy': seen_flux, 'other_mjy': other_flux, 'other_err': other_err, 'depth_sigma': depth,
              'other_text': f'{other_flux:.2f} +/- {other_err:.2f} mJy (vs {seen_flux:.2f} mJy when seen)'}
    if depth < 5:
        result['kind'] = 'too shallow'
    elif other_flux < 3 * other_err:
        result['kind'] = 'transient' if len(seen_visits) == 1 else 'variable'
    elif other_flux > 0.5 * seen_flux:
        result['kind'] = 'static'                       # it IS there in the other visits, about as bright
    else:
        result['kind'] = 'variable'                     # there, but clearly fainter
    return result


def _link_movers(singles, frames, visit):
    """Detections seen once each that line up: same visit, at least 3 different times, on one straight line at a
    steady rate (within FIELD_MATCH_PIX). Returns 'moving' objects (with a least-squares fit of the path)."""
    tol = cfg.FIELD_MATCH_PIX * cfg.PIXEL_ARCSEC / 3600
    pool = list(singles)
    movers = []
    while True:
        best = None
        for a in range(len(pool)):
            for b in range(len(pool)):
                pa, pb = pool[a], pool[b]
                dt = pb['mjd'] - pa['mjd']
                if dt <= 0.002 or visit[pa['frame']] != visit[pb['frame']]:
                    continue
                cos = np.cos(np.radians(pa['dec']))
                rate_ra = ((pb['ra'] - pa['ra'] + 180) % 360 - 180) * cos / dt
                rate_dec = (pb['dec'] - pa['dec']) / dt
                if np.hypot(rate_ra, rate_dec) > cfg.SKYMAP_MAX_RATE_DEG_PER_DAY * 4:
                    continue
                support = []
                for c, pc in enumerate(pool):
                    if visit[pc['frame']] != visit[pa['frame']]:
                        continue
                    t = pc['mjd'] - pa['mjd']
                    pdec = pa['dec'] + rate_dec * t
                    pra = pa['ra'] + rate_ra * t / np.cos(np.radians(pdec))
                    if np.hypot(((pc['ra'] - pra + 180) % 360 - 180) * np.cos(np.radians(pdec)), pc['dec'] - pdec) <= tol:
                        support.append(c)
                times = np.unique(np.round([pool[c]['mjd'] for c in support], 3))
                if len(times) >= 3 and (best is None or len(support) > len(best)):
                    best = support
        if best is None:
            return movers
        members = [pool[c] for c in best]
        t = np.array([m['mjd'] for m in members])
        t0 = float(t.mean())
        dec0 = np.array([m['dec'] for m in members])
        ra_off = ((np.array([m['ra'] for m in members]) - members[0]['ra'] + 180) % 360 - 180) * np.cos(np.radians(dec0))
        rate_ra, ra0_off = np.polyfit(t - t0, ra_off, 1)
        rate_dec, dec0_fit = np.polyfit(t - t0, dec0, 1)
        fit = {'t0': t0, 'ra0': members[0]['ra'] + ra0_off / np.cos(np.radians(dec0_fit)), 'dec0': dec0_fit,
               'rate_ra': rate_ra, 'rate_dec': rate_dec, 'mjd': t}
        ra, dec = _linear(fit, t0)
        movers.append({'kind': 'moving', 'ra': float(ra), 'dec': float(dec), 'detections': members, 'fit': fit,
                       'rate_arcsec_per_hour': float(np.hypot(rate_ra, rate_dec) * 3600 / 24),
                       'visits_seen': [int(visit[members[0]['frame']])], 'visits_covered': [], 'n_covered': len(members)})
        pool = [p for c, p in enumerate(pool) if c not in set(best)]


def _describe(obj):
    n = len(obj['detections'])
    times = len(np.unique(np.round([m['mjd'] for m in obj['detections']], 3)))
    known = ('not a known asteroid or comet (JPL SB Ident)' if obj.get('jpl_checked') else
             'NOT checked against known asteroids (press J on its frame in the viewer)')
    if obj['kind'] == 'moving':
        return (f"moving {obj['rate_arcsec_per_hour']:.1f}\"/hour on a straight line, {n} detections at {times} times - "
                f"{known}: unknown solar-system object candidate")
    depth = obj.get('depth') or {}
    if obj['kind'] == 'transient':
        return (f"seen in 1 survey visit ({n} detections at {times} times); the other visit(s) were deep enough to see "
                f"it ({depth.get('depth_sigma', 0):.0f}x their noise) and measure {depth.get('other_text', 'nothing')} "
                f"there: likely TRANSIENT (or a very slow mover)")
    if obj['kind'] == 'variable':
        return (f"same place in {len(obj['visits_seen'])} of {len(obj['visits_covered'])} visits that covered it "
                f"({n} detections); {depth.get('other_text', 'fainter')} in the others: likely VARIABLE, not in the "
                f"catalogs")
    if obj['kind'] == 'too shallow':
        return (f"seen in {len(obj['visits_seen'])} visit(s) ({n} detections), not in {len(obj['visits_covered']) - len(obj['visits_seen'])} "
                f"other(s) - but those were too shallow to see it (it would be only {depth.get('depth_sigma', 0):.1f}x "
                f"their noise): can't tell whether it is transient")
    if obj['kind'] == 'static':
        return (f"same place in all {len(obj['visits_seen'])} visits ({n} detections): steady source that is not in "
                f"Gaia DR3 / Hipparcos / CatWISE - likely NOT transient")
    if obj['kind'] == 'single':
        sigma = max(m['sigma'] for m in obj['detections'])
        arms = ('by BOTH detector arms (so not a cosmic ray)' if obj.get('both_arms') else
                'by one detector arm (the other did not cover it)')
        if any(m.get('mover_flags') for m in obj['detections']):
            return (f"seen at one instant only, {arms}, {sigma:.0f} sigma; SPHEREx's pipeline flagged it as changing "
                    f"DURING the exposure (OUTLIER/TRANSIENT): likely a MOVING object - {known}")
        return f"seen at one instant only, {arms}, {sigma:.0f} sigma: a fast mover or a flare - {known}"
    return f"{n} detections in one visit; no other visit covers this spot: can't tell if it is transient"


# ----------------------------------------------------------------------------
# Everything together, and the output table
# ----------------------------------------------------------------------------

def map_sky(frames, coord):
    """Map every object in the frames around `coord` (needs the full pixels: before make_zooms).
    Returns {'catalog', 'bodies', 'planets', 'new_objects'}; frame['map'] holds each frame's objects."""
    radius = cfg.ZOOM_SIZE_PIX / 2 * np.sqrt(2) * cfg.PIXEL_ARCSEC / 3600 + 0.01        # covers the cutouts
    catalog = catalog_layer(coord)
    mjd = np.array([f['mjd'] for f in frames])
    planets = planets_in_view(coord, mjd, radius)
    if planets:
        log.info('Planets / Moon near the field (DE440s): ' + ', '.join(p.capitalize() for p in planets))
    # first pass without asteroids: which frames have something unexplained? (they are checked with JPL first)
    unexplained = map_frames(frames, catalog, [], planets)
    bodies, checked = small_bodies(frames, coord, radius, priority={u_[0] for u_ in unexplained})
    if bodies:
        unexplained = map_frames(frames, catalog, bodies, planets)
    new = classify_new(frames, unexplained, checked)
    if new:
        kinds = {}
        for o in new:
            kinds[o['kind']] = kinds.get(o['kind'], 0) + 1
        log.info(f"NEW OBJECTS (in no catalog, not a known asteroid/comet/planet, not flagged as an artifact): "
              + ', '.join(f'{n} {k}' for k, n in kinds.items()))
        for o in new[:15]:
            first = min(o['detections'], key=lambda m: m['mjd'])
            log.info(f"  {o['id']:7s} RA {o['ra']:10.5f}  Dec {o['dec']:+9.5f}  first seen "
                  f"{Time(first['mjd'], format='mjd').utc.iso[:19]} UTC  - {o['description']}")
        if len(new) > 15:
            log.info(f'  ... and {len(new) - 15} more (all in the new-objects file)')
    else:
        log.info('NEW OBJECTS: none - every source SPHEREx detects here is a known star, asteroid, comet or planet')
    unchecked = len(frames) - len(checked)
    if unchecked and any(o['kind'] in ('single', 'moving') and not o.get('jpl_checked') for o in new):
        log.info(f'  (note: {unchecked} frames were not checked against known asteroids (one JPL call per run); '
              f'"single" / "moving" objects there could still be known ones - press J on that frame in the viewer)')
    return {'catalog': catalog, 'bodies': bodies, 'planets': planets, 'new_objects': new, 'checked': checked}


def new_objects_table(new_objects, frames):
    """One row per detection of every new object: id, class, RA/Dec (deg and sexagesimal), UTC time, MJD, frame,
    detector, wavelength, significance."""
    rows = []
    for obj in new_objects:
        for m in sorted(obj['detections'], key=lambda d: d['mjd']):
            f = frames[m['frame']]
            sky = SkyCoord(m['ra'], m['dec'], unit='deg')
            rows.append({'object': obj['id'], 'class': obj['kind'], 'ra_deg': round(m['ra'], 6),
                         'dec_deg': round(m['dec'], 6),
                         'ra_hms': sky.ra.to_string(u.hourangle, sep=':', precision=2, pad=True),
                         'dec_dms': sky.dec.to_string(sep=':', precision=1, alwayssign=True, pad=True),
                         'time_utc': Time(m['mjd'], format='mjd').utc.isot[:23], 'mjd': round(m['mjd'], 6),
                         'frame': f"T{m['frame'] + 1}", 'detector': f['detector'],
                         'wavelength_um': round(f['wavelength_um'], 4), 'significance': round(m['sigma'], 1),
                         'description': obj['description']})
    return Table(rows=rows) if rows else Table(names=('object', 'class', 'ra_deg', 'dec_deg', 'time_utc', 'mjd'))
