"""The website's way into the pipeline: analyse(request, folder, on_progress) runs one analysis with no windows and
writes what the browser's time-lapse needs into `folder`:

    summary.json        the target, and every frame's time, detector, wavelength, signal-to-noise and flags
    frame_<n>.png       every frame (n from 1, in time order)

`request` is what Test_Robert_Code_Folder/app.py makes of the analysis popup's payload (see web_request):
    {'ra_deg', 'dec_deg', 'patch_px', 'max_images' (None: all), 'star' (None, or a Gaia DR3 star with its motion)}

Every PNG is the frame's original pixels resampled onto ONE grid centred on the target, north up and east left
(SPHEREx images come in many rotations; on one grid they line up, so the frames can be played as a time-lapse). One
PNG pixel is one SPHEREx pixel (6.15"); the browser scales it up without smoothing, so the real pixels stay visible.

Like the desktop viewer, every run of something fixed on the sky compares each frame with the catalogs (Gaia DR3,
CatWISE2020, Hipparcos, JPL's known asteroids and comets, the planets) and maps the sky (skymap.py): every object
is named, and what no catalog explains is flagged as a NEW object (static / variable / transient / moving /
single). Each mapped object carries how to track it; a request with a 'track' spec runs that (main.run_track):
a catalog star, a known asteroid (JPL Horizons path), a sky position, or a new object's fitted path.

What a web run does differently from the desktop (through api.settings_override, for the duration of the run only):
no windows, the patch size the popup asks for, the sky map for every fixed target, and - with a limit on the number
of images - whole exposures spread over all the survey dates rather than the first ones. Runs are one at a time
(api's lock): the settings are module-wide.
"""

import json
import logging
import os
import re
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')      # the pipeline imports matplotlib.pyplot: never open a window here

import numpy as np
from astropy.time import Time

from . import config as cfg

log = logging.getLogger(__name__)

PATCH_PX_LIMITS = (10, 400)                 # pixels per side the website may ask for (400 px = 41 arcmin)
MAX_IMAGES_LIMIT = 2000                     # more than this is not played in a browser
DEFAULT_MAX_IMAGES = 24
GAIA_EPOCH_YEAR = 2016.0                    # ESA Gaia DR3 reference epoch


def results_folder():
    """Where web runs are kept: <results>/jobs (next to the pipeline's own results files)."""
    from .results import results_dir
    return results_dir() / 'jobs'


# ----------------------------------------------------------------------------
# The request
# ----------------------------------------------------------------------------

def _number(value, low, high, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number') from None
    if not (low <= number <= high):                         # (also refuses NaN)
        raise ValueError(f'{name} must be between {low} and {high}')
    return number


def web_request(payload):
    """The popup's payload (static/js/analysis.js: analysisPayload, plus 'max_images') -> the few values a run needs.
    Raises ValueError with a message for the user if something is missing or out of range. The result is also the
    run's identity: the same request gives the same job, so finished results are reused.

        target.ra_deg, target.dec_deg     where to point (geocentric J2000, degrees)
        patch.half_width_deg              -> patch_px, pixels per side (limited to PATCH_PX_LIMITS)
        max_images                        a number, or null / 'all' for every image
        named_target                      with pmra/pmdec: a Gaia DR3 star, followed along its motion (epoch J2016.0)
    Not needed by the pipeline: time (the browser uses it to pick the frame to start on), spherex (the cutouts are
    made by IRSA from SPHEREx's own pointing), and the rest of target (derived from ra/dec)."""
    if not isinstance(payload, dict):
        raise ValueError('the request must be a JSON object')
    target, patch = payload.get('target'), payload.get('patch')
    if not isinstance(target, dict) or not isinstance(patch, dict):
        raise ValueError('target and patch are required')
    request = {'ra_deg': round(_number(target.get('ra_deg'), 0, 360, 'target.ra_deg'), 6),
               'dec_deg': round(_number(target.get('dec_deg'), -90, 90, 'target.dec_deg'), 6)}
    half_width = _number(patch.get('half_width_deg'), 1e-4, 10, 'patch.half_width_deg')
    wanted = int(round(2 * half_width * 3600 / cfg.PIXEL_ARCSEC))
    request['patch_px'] = int(min(max(wanted, PATCH_PX_LIMITS[0]), PATCH_PX_LIMITS[1]))
    max_images = payload.get('max_images', DEFAULT_MAX_IMAGES)
    if max_images in (None, 'all'):
        request['max_images'] = None
    else:
        request['max_images'] = int(_number(max_images, 1, MAX_IMAGES_LIMIT, 'max_images'))
    request['star'] = None
    if payload.get('track') is not None:                  # an object picked in the images: track it instead
        request['track'] = track_request(payload['track'])
        return request
    named = payload.get('named_target')
    if isinstance(named, dict) and named.get('pmra_mas_yr') is not None and named.get('pmdec_mas_yr') is not None:
        parallax = named.get('parallax_mas')
        request['star'] = {
            'name': str(named.get('name') or 'target')[:80],
            'gaia_dr3': str(named['gaia_dr3'])[:40] if named.get('gaia_dr3') else None,
            'ra': _number(named.get('ra_deg_at_epoch'), 0, 360, 'named_target.ra_deg_at_epoch'),
            'dec': _number(named.get('dec_deg_at_epoch'), -90, 90, 'named_target.dec_deg_at_epoch'),
            'epoch': _number(named.get('epoch') or GAIA_EPOCH_YEAR, 1900, 2100, 'named_target.epoch'),
            'pmra': _number(named['pmra_mas_yr'], -30000, 30000, 'named_target.pmra_mas_yr'),
            'pmdec': _number(named['pmdec_mas_yr'], -30000, 30000, 'named_target.pmdec_mas_yr'),
            'parallax': None if parallax is None else _number(parallax, -100, 2000, 'named_target.parallax_mas'),
        }
    return request


DATE = re.compile(r'\d{4}-\d{2}-\d{2}')


def _text(value, name, limit=80):
    if not isinstance(value, (str, int, float)) or not str(value).strip():
        raise ValueError(f'{name} is missing')
    return str(value).strip()[:limit]


def track_request(spec):
    """A mapped object's track spec (as the summary gave it to the browser: plots.track_spec) checked and reduced to
    plain values - the browser sends it back to track that object. ValueError if it is not one."""
    if not isinstance(spec, dict):
        raise ValueError('track must be an object')
    kind, name = spec.get('kind'), _text(spec.get('name', 'object'), 'track.name')
    if kind == 'star':
        star = spec.get('star')
        if not isinstance(star, dict):
            raise ValueError('track.star is missing')
        parallax = star.get('parallax')
        return {'kind': 'star', 'name': name, 'star': {
            'name': _text(star.get('name', name), 'track.star.name'),
            'object_type': _text(star.get('object_type', 'catalog star'), 'track.star.object_type'),
            'query': _text(star.get('query', name), 'track.star.query'),
            'ra': _number(star.get('ra'), 0, 360, 'track.star.ra'), 'dec': _number(star.get('dec'), -90, 90, 'track.star.dec'),
            'pmra': _number(star.get('pmra') or 0, -30000, 30000, 'track.star.pmra'),
            'pmdec': _number(star.get('pmdec') or 0, -30000, 30000, 'track.star.pmdec'),
            'parallax': None if parallax is None else _number(parallax, -100, 2000, 'track.star.parallax')}}
    if kind == 'asteroid':
        window = spec.get('window')
        if not (isinstance(window, (list, tuple)) and len(window) == 2 and all(DATE.fullmatch(str(w)) for w in window)):
            raise ValueError('track.window must be two dates (YYYY-MM-DD)')
        return {'kind': 'asteroid', 'name': name, 'id': _text(spec.get('id'), 'track.id', 40),
                'window': [str(w) for w in window]}
    if kind == 'coords':
        coords = spec.get('coords')
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            raise ValueError('track.coords must be [ra, dec]')
        return {'kind': 'coords', 'name': name,
                'coords': [_number(coords[0], 0, 360, 'track.coords'), _number(coords[1], -90, 90, 'track.coords')]}
    if kind == 'path':
        mjd, ra, dec = (spec.get(k) for k in ('mjd', 'ra', 'dec'))
        if not all(isinstance(v, (list, tuple)) for v in (mjd, ra, dec)) or not (2 <= len(mjd) == len(ra) == len(dec) <= 500):
            raise ValueError('track.mjd / ra / dec must be lists of the same length (2 to 500)')
        return {'kind': 'path', 'name': name, 'mjd': [_number(v, 60000, 70000, 'track.mjd') for v in mjd],
                'ra': [_number(v, 0, 360, 'track.ra') for v in ra], 'dec': [_number(v, -90, 90, 'track.dec') for v in dec]}
    raise ValueError("track.kind must be 'star', 'asteroid', 'coords' or 'path'")


def pipeline_star(request):
    """The request's target as the pipeline's `star` (see stars.positions_at). A Gaia star carries the epoch of
    its position ('epoch_mjd'), so it is moved from there - not from J2000, the epoch SIMBAD positions have."""
    star = request.get('star')
    if star:
        return {'name': star['name'], 'object_type': 'Gaia DR3 star', 'query': star.get('gaia_dr3') or star['name'],
                'ra': star['ra'], 'dec': star['dec'], 'pmra': star['pmra'], 'pmdec': star['pmdec'],
                'parallax': star['parallax'], 'rv': None, 'magnitudes': {}, 'spectral_type': '?',
                'epoch_mjd': float(Time(star['epoch'], format='jyear').mjd)}
    ra, dec = request['ra_deg'], request['dec_deg']
    return {'name': f'RA {ra:.5f} Dec {dec:+.5f}', 'object_type': 'sky position', 'query': f'{ra} {dec}',
            'ra': ra, 'dec': dec, 'pmra': 0.0, 'pmdec': 0.0, 'parallax': None, 'rv': None, 'magnitudes': {},
            'spectral_type': '?'}


def zoom_px(patch_px):
    """Pixels per side the pipeline must keep of every frame so that the north-up patch is filled whatever the
    frame's rotation: the patch's diagonal (a frame rotated 45 degrees), plus a margin for the resampling."""
    return int(np.ceil(patch_px * np.sqrt(2))) + 4


def run_settings(request):
    """config.json values for this web run only (see the module docstring)."""
    zoom = zoom_px(request['patch_px'])
    cutout_arcmin = max(cfg.CUTOUT_SIZE.to_value('arcmin'), (zoom + 4) * cfg.PIXEL_ARCSEC / 60)
    # the catalogs (Gaia DR3, CatWISE) must cover everything the sky map looks at - the whole zoom, to its corners;
    # a star beyond FIELD_RADIUS has no catalog entry and would be flagged NEW
    field_arcmin = max(cfg.FIELD_RADIUS.to_value('arcmin'), zoom / 2 * np.sqrt(2) * cfg.PIXEL_ARCSEC / 60 + 0.5)
    return {'ZOOM_SIZE_PIX': zoom, 'CUTOUT_SIZE': cutout_arcmin, 'FIELD_RADIUS': field_arcmin, 'SPREAD_IMAGES': True, 'FIELD_CHECK': True,
            'SKYMAP_FOR_ALL_TARGETS': True, 'LIVE_VIEW': False, 'ZOOM_ONLY': True,
            'SAVE_RESULTS': False,
            'MEASURE_WORKERS': 0}      # measure in the server process: no worker processes started from a web server      # (the desktop's results list is for desktop runs; this run is kept as summary.json)


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------

def north_up_grid(size_pix):
    """(east, north) offsets from the target in arcsec for a size_pix x size_pix image: row 0 is the north edge,
    column 0 the east edge (north up, east left, like the popup's patch map)."""
    offsets = (np.arange(size_pix) - (size_pix - 1) / 2) * cfg.PIXEL_ARCSEC
    north, east = np.meshgrid(-offsets, -offsets, indexing='ij')
    return east, north


def resampled(zoom, size_pix):
    """The zoom's pixels on the north-up grid (sky level removed, bad / missing pixels NaN)."""
    from .spectrumanalyser import sample_frame
    east, north = north_up_grid(size_pix)
    return sample_frame(zoom, east, north)[0]


def flags_in_aperture(zoom):
    """Names of the SPHEREx pixel flags set inside the measuring circle around the target."""
    header, flags = zoom.get('flags_header'), zoom.get('flags')
    if header is None or flags is None:
        return []
    x, y = zoom['catalog_xy']
    yy, xx = np.indices(flags.shape)
    inside = np.hypot(xx - x, yy - y) <= cfg.APERTURE_RADIUS_PIX
    if not inside.any():
        return []
    present = int(np.bitwise_or.reduce(np.asarray(flags)[inside].astype(np.int64)))
    return sorted({str(header[key]) for key in header['MSKN*'] if present & (1 << int(key[4:]))})


def north_up_zoom(zoom, size_pix):
    """The zoom resampled onto the north-up grid, as a zoom of its own (pixels + a north-up TAN WCS centred where
    the grid is), so the backend's display code (visual.render) and sky positions work on it unchanged."""
    from astropy.wcs import WCS
    ra0, dec0 = (float(v) for v in zoom['wcs'].pixel_to_world_values(*zoom['catalog_xy']))   # the grid's centre
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.crpix = [(size_pix + 1) / 2, (size_pix + 1) / 2]        # FITS counts from 1: the middle pixel
    wcs.wcs.cdelt = [-cfg.PIXEL_ARCSEC / 3600, cfg.PIXEL_ARCSEC / 3600]   # east left, north up
    keep = ('mjd', 'detector', 'wavelength_um', 'object_found', 'object_ra', 'object_dec', 'expected_ra',
            'expected_dec')
    return {**{key: zoom[key] for key in keep if key in zoom},
            'image': resampled(zoom, size_pix)[::-1],              # rows from south to north, like a FITS image
            'wcs': wcs}


def reference_position(movement, index):
    """The blue dashed circle of the desktop viewer: movement['reference'] (the object at T1 for a star), or None."""
    reference = (movement or {}).get('reference')
    if reference is None:
        return None
    reference = reference[index] if getattr(reference, 'shape', ()) else reference
    return float(reference.ra.deg), float(reference.dec.deg)


def png_xy(north, factor, ra, dec):
    """Sky positions -> pixel positions in the PNG (x right, y DOWN, in PNG pixels) and whether each is inside."""
    ny, nx = north['image'].shape
    x, y = (np.atleast_1d(np.asarray(v, dtype=float)) for v in north['wcs'].world_to_pixel_values(ra, dec))
    inside = (x >= -0.5) & (x < nx - 0.5) & (y >= -0.5) & (y < ny - 0.5)
    return (x + 0.5) * factor, (ny - (y + 0.5)) * factor, inside


def target_marks(zoom, north, factor, blue):
    """The desktop viewer's circles for the analysed object, in PNG pixels: red = where it was found in this frame,
    blue dashed = the reference position (movement['reference'], drawn only inside the image). Radii in PNG pixels."""
    marks = {'red': None, 'blue': None, 'radius': cfg.APERTURE_RADIUS_PIX * factor,
             'blue_radius': (cfg.APERTURE_RADIUS_PIX + 0.6) * factor}
    red = (zoom.get('object_ra'), zoom.get('object_dec'))
    if all(v is not None and np.isfinite(v) for v in red):
        x, y, _ = png_xy(north, factor, *red)
        marks['red'] = [round(float(x[0]), 1), round(float(y[0]), 1)]
    if blue is not None:
        x, y, inside = png_xy(north, factor, *blue)
        if inside[0]:
            marks['blue'] = [round(float(x[0]), 1), round(float(y[0]), 1)]
    return marks


def object_id(entry):
    """One mapped object's id, the same in every frame (plots._object_key: catalog row, JPL designation, NEW id)."""
    kind = entry['kind']
    key = {'star': entry.get('catalog_row'), 'small body': entry.get('body'), 'new': entry.get('new_id')}.get(kind)
    return f"{kind}:{entry['label'] if key is None else key}"


MAP_COUNTS = (('Hipparcos', 'Hipparcos'), ('Gaia', 'Gaia DR3 (+)'), ('CatWISE', 'CatWISE (x)'),
              ('small body', 'known asteroid/comet'), ('planet', 'planet'), ('artifact', 'flagged artifact'),
              ('new', 'NEW (yellow)'))


def map_marks(zoom, north, factor):
    """The sky map of one frame (frame['map'], skymap.map_frames) in PNG pixels, plus the desktop viewer's summary
    line for it ('mapped: 3 Hipparcos, 41 Gaia DR3 (+) ...'). None, None if the frame has no sky map."""
    entries = zoom.get('map')
    if entries is None:
        return None, None
    marks, counts = [], {}
    if entries:
        xs, ys, inside = png_xy(north, factor, [e['ra'] for e in entries], [e['dec'] for e in entries])
        for x, y, ok, entry in zip(xs, ys, inside, entries):
            if not ok:
                continue
            key = entry.get('source', 'Gaia') if entry['kind'] == 'star' else entry['kind']
            counts[key] = counts.get(key, 0) + 1
            marks.append({'id': object_id(entry), 'kind': entry['kind'], 'source': entry.get('source'),
                          'label': entry['label'], 'named': bool(entry.get('named')),
                          'detected': bool(entry.get('detected', True)), 'x': round(float(x), 1), 'y': round(float(y), 1)})
    parts = [f'{counts[key]} {label}' for key, label in MAP_COUNTS if counts.get(key)]
    return marks, 'mapped: ' + (', '.join(parts) or 'nothing detected')


def write_frames(zooms, folder, size_pix, movement=None):
    """frame_<n>.png for every zoom: the desktop viewer's pretty image (visual.render - colour from the simultaneous
    frame on the other detector arm, sharpened and upscaled DISPLAY_UPSCALE times, the target drawn at the same
    brightness and in the same colour in every frame), on the north-up grid so the frames line up. Returns one dict
    per frame with what the browser draws on top, in PNG pixels: the red / blue circles ('marks') and the sky map
    ('objects', 'map_text') - drawn there, not into the PNG, so they can be clicked and switched off."""
    from PIL import Image
    from .visual import anchor_colour, find_partners, render, target_xy
    factor = int(cfg.DISPLAY_UPSCALE)
    north = [north_up_zoom(zoom, size_pix) for zoom in zooms]
    partners = find_partners(north)
    colour = anchor_colour(north, partners)
    shown = []
    for i, (zoom, flat) in enumerate(zip(zooms, north)):
        partner = north[partners[i]] if partners[i] is not None else None
        rgb, _ = render(flat, partner, factor=factor, anchor={'xy': target_xy(flat), 'colour': colour})
        rgb = (np.clip(np.flipud(rgb), 0, 1) * 255).astype(np.uint8)     # PNG rows run top to bottom
        Image.fromarray(rgb).save(Path(folder) / f'frame_{i + 1}.png')
        wavelengths = sorted([zoom['wavelength_um'], partner['wavelength_um']]) if partner is not None else None
        objects, map_text = map_marks(zoom, flat, factor)
        shown.append({'png': f'frame_{i + 1}.png', 'flags_in_aperture': flags_in_aperture(zoom),
                      'colour_um': wavelengths,                        # [short, long]: what the colours compare
                      'marks': target_marks(zoom, flat, factor, reference_position(movement, i)),
                      'objects': objects, 'map_text': map_text})
    return shown


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

FRAME_FIELDS = ('obs_id', 'mjd', 'detector', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr',
                'n_bad_in_aperture', 'saturated', 'track', 'astrometry_ok',
                'expected_ra', 'expected_dec')          # where the target was in that image (the patch centre)


def frame_title(frames, index, movement, brightness):
    """The desktop viewer's title for frame T(index+1) (plots.draw_zoom), as three lines:
    'T5 of 21   2025-11-03 12:10   +186.81 d   detector 4', the wavelength and how far it moved since T1, and
    its brightness (flux, and against its own spectrum) - with the same flags (MOVED, BRIGHTNESS CHANGE ...)."""
    from .plots import fmt_angle
    movement, brightness = movement or {}, brightness or {}
    frame = frames[index]

    def at(result, key, default=np.nan):
        values = result.get(key)
        return values[index] if values is not None and len(values) > index else default

    days = frame['mjd'] - frames[0]['mjd']
    n_bad = frame.get('n_bad_in_aperture') or 0
    flag_label = movement.get('flag_label', 'MOVED')
    moved, changed = bool(at(movement, 'flagged', False)), bool(at(brightness, 'flagged', False))
    unreliable = f'   [{n_bad} bad pixel(s) in circle: not used]' if n_bad and not frame.get('saturated') else ''
    offset = at(movement, 'offset_arcsec')
    travelled = f'moved {fmt_angle(offset)}' if np.isfinite(offset) else 'movement: not measured'
    if 'notes' in movement:                         # an asteroid: signal-to-noise and distance from its ephemeris
        third = movement['notes'][index]
    else:                                           # a star or a sky position: brightness against its own spectrum
        deviation = at(brightness, 'deviation')
        versus = f'{deviation * 100:+.1f}%' if np.isfinite(deviation) else 'n/a (too few similar-wavelength frames)'
        third = f"vs spectrum {versus}{'  << BRIGHTNESS CHANGE' if changed else ''}"
    if frame.get('saturated'):
        deviation = at(brightness, 'deviation') if brightness.get('source') == 'glow' else np.nan
        third_line = ('SATURATED: position from its outer glow, ' +
                      (f'glow vs spectrum {deviation * 100:+.1f}%{"  << BRIGHTNESS CHANGE" if changed else ""}'
                       if np.isfinite(deviation) else 'brightness not measurable'))
    else:
        third_line = f"flux {frame['flux_mjy']:.2f} ± {frame['flux_err_mjy']:.2f} mJy   {third}"
    date = str(frame.get('date_obs') or Time(frame['mjd'], format='mjd').utc.isot)[:16].replace('T', ' ')
    return [f"T{index + 1} of {len(frames)}   {date}   +{days:.2f} d   detector {frame['detector']}{unreliable}",
            f"λ = {frame['wavelength_um']:.3f} µm   {travelled}{f'  << {flag_label}' if moved else ''}",
            third_line]


def mapped_objects(results, shown):
    """Every object the sky map found INSIDE the images shown (shown = write_frames' output; the sky map also covers
    the margin cut for rotated frames): {id: {label, kind, source, info, frames (T numbers), detected_in,
    track (how to track it: plots.track_spec) or reason (why it can't be)}}."""
    from .api import jsonable
    from .plots import track_spec
    tracked, objects = results['tracked'], {}
    for n, (zoom, frame) in enumerate(zip(results['zooms'], shown), start=1):
        inside = {o['id'] for o in frame.get('objects') or []}
        for entry in zoom.get('map') or []:
            oid = object_id(entry)
            if oid not in inside:
                continue
            obj = objects.get(oid)
            if obj is None:
                try:
                    spec = jsonable(track_spec(entry, tracked, zoom['mjd']))
                except Exception as e:                      # (never stops the summary)
                    spec = {'kind': 'none', 'reason': f'cannot be tracked ({type(e).__name__})'}
                obj = objects[oid] = {
                    'label': entry['label'], 'kind': entry['kind'], 'source': entry.get('source'),
                    'info': entry.get('info'), 'named': bool(entry.get('named')), 'frames': [], 'detected_in': [],
                    'track': spec if spec and spec.get('kind') != 'none' else None,
                    'reason': spec.get('reason') if spec and spec.get('kind') == 'none' else None}
            obj['frames'].append(n)
            if entry.get('detected', True):
                obj['detected_in'].append(n)
    return objects


def database_comparison(results, log_lines, objects=None):
    """What the frames were compared with, and what came of it (the desktop's field check and sky map). With
    `objects` (mapped_objects), the NEW objects and known asteroids are the ones inside the images shown; the
    number of NEW objects found only in the margin around them is 'new_outside'."""
    from .api import jsonable
    from astropy.time import Time as _Time
    sky, field, table = results.get('skymap'), results.get('field'), results['tracked'].get('field')
    out = {'sky_map': sky is not None, 'field_check': field is not None}
    skipped = [' '.join(str(line).split()).strip('()') for line in log_lines
               if ('skipped' in line and ('Sky map' in line or 'Field' in line)) or 'not available' in line]
    if skipped:                                         # (the reason, without the network library's details)
        out['notes'] = [re.sub(r'\s*\((HTTPSConnectionPool|Max retries).*$', ')', note)[:200] for note in skipped]
    if field is not None:
        offsets = np.asarray(field.get('offset_px', []), dtype=float).reshape(-1, 2)
        out['pointing'] = {'matches': int(len(offsets)),
                           'median_offset_arcsec': jsonable(float(np.median(np.hypot(*offsets.T))) * cfg.PIXEL_ARCSEC)
                           if len(offsets) else None}
    catalog = sky['catalog'] if sky is not None else table
    if catalog is not None and len(catalog):
        sources = np.asarray(catalog['source']).astype(str)
        out['catalog_stars'] = {name: int(np.sum(sources == name)) for name in ('Gaia', 'CatWISE') if np.any(sources == name)}
        if 'hip' in catalog.colnames:
            out['catalog_stars']['Hipparcos'] = int(np.sum(np.asarray(catalog['hip']) > 0))
    if sky is not None:
        shown = (lambda oid: True) if objects is None else (lambda oid: oid in objects)
        out['known_small_bodies'] = [{'name': b['fullname'], 'kind': b['kind'], 'orbit_class': b['orbit_class'],
                                      'frames': sorted(int(i) + 1 for i in b['positions'])} for b in sky['bodies']
                                     if shown(f"small body:{b.get('des')}") or 'des' not in b]
        out['planets'] = sorted(p.capitalize() for p in sky['planets'])
        out['new_objects'] = [{
            'id': o['id'], 'class': o['kind'], 'description': o.get('description'),
            'ra_deg': jsonable(o['ra']), 'dec_deg': jsonable(o['dec']), 'n_detections': len(o['detections']),
            'first_seen_utc': _Time(min(m['mjd'] for m in o['detections']), format='mjd').utc.strftime('%Y-%m-%d %H:%M'),
            'object': f"new:{o['id']}"} for o in sky['new_objects'] if shown(f"new:{o['id']}")]
        out['new_outside'] = len(sky['new_objects']) - len(out['new_objects'])
    return out


def summary_of(results, request, shown, log_lines=()):
    """The JSON the browser gets when the run is done (also saved as summary.json)."""
    from .api import jsonable
    tracked, frames = results['tracked'], results['frames']
    movement, brightness = results.get('movement'), results.get('brightness')
    objects = mapped_objects(results, shown)
    rows = []
    for n, (frame, extra) in enumerate(zip(frames, shown), start=1):
        row = {'n': n, 'name': f'T{n}', 'title': frame_title(frames, n - 1, movement, brightness),
               'time_utc': Time(frame['mjd'], format='mjd').utc.strftime('%Y-%m-%d %H:%M:%S')}
        row.update({key: jsonable(frame.get(key)) for key in FRAME_FIELDS})
        row.update(extra)
        rows.append(row)
    return {'target': {'name': tracked.get('name'), 'ra_deg': request['ra_deg'], 'dec_deg': request['dec_deg'],
                       'moving': 'ephemeris' in tracked,
                       'gaia_dr3': (request.get('star') or {}).get('gaia_dr3')},
            'request': request,
            'patch_px': request['patch_px'], 'pixel_arcsec': cfg.PIXEL_ARCSEC,
            'images_available': jsonable(tracked.get('images_available')), 'n_frames': len(rows),
            'frames': rows,
            'image_px': request['patch_px'] * int(cfg.DISPLAY_UPSCALE),
            'objects': objects,
            'database': database_comparison(results, log_lines, objects),
            'tracking': (request.get('track') or {}).get('name'),
            'circles': {'red': 'object found in this frame',
                        'blue': (movement or {}).get('reference_label', 'object at T1')},
            'source': 'NASA/IPAC IRSA SPHEREx spectral images (' + ', '.join(cfg.COLLECTIONS) + ')'}


REJECTION_NOTES = ('skipped', 'too close to the edge', 'could not be fitted')


def explained(error, log_lines):
    """The pipeline's error, with the reasons it logged for rejecting images (otherwise only in the server's
    terminal) - e.g. 'None of the images could be measured at this position' says nothing about why."""
    notes = [' '.join(line.split()) for line in log_lines if any(key in line for key in REJECTION_NOTES)]
    message = str(error).strip().rstrip('.')
    return message + ('. ' + ' '.join(dict.fromkeys(notes)) if notes else '.')


def run(request, pipeline):
    """The pipeline run for one web request: the request's target (fixed, or a Gaia star moving at its proper
    motion), or - with a 'track' spec - the object picked in an earlier run's images (main.run_track's choices)."""
    spec, n = request.get('track'), request['max_images']
    if spec is None:
        return pipeline.run_fixed_target(pipeline_star(request), {'mode': 'web'}, n, show=False)
    if spec['kind'] == 'star':                      # a catalog star (J2000 position, motion, parallax)
        star = {**spec['star'], 'rv': None, 'magnitudes': {}, 'spectral_type': '?'}
        return pipeline.run_fixed_target(star, {'mode': 'catalog star'}, n, show=False)
    if spec['kind'] == 'coords':
        ra, dec = spec['coords']
        return pipeline.run_fixed_target(pipeline_star({'ra_deg': ra, 'dec_deg': dec, 'star': None}),
                                         {'mode': 'web'}, n, show=False)
    if spec['kind'] == 'asteroid':                  # JPL Horizons path, as seen from SPHEREx
        return pipeline.run_asteroid(spec['id'], tuple(spec['window']), n, show=False)
    return pipeline.run_path(spec['name'], spec['mjd'], spec['ra'], spec['dec'], n, show=False)   # a NEW mover


def analyse(request, folder, on_progress=None):
    """Run the pipeline for one web request and write summary.json + frame_<n>.png into `folder`.
    on_progress(done, total, message): called while the images download (done and total are 0 until the number
    of images is known). Returns the summary. Raises RuntimeError / ValueError with a message for the user."""
    from . import main as pipeline
    from .api import _run_lock, progress_events, settings_override
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report = on_progress or (lambda done, total, message: None)
    log_lines = []                                  # the pipeline's own messages: they say why images were rejected

    state = {'done': 0, 'total': 0, 'loading': True}

    def hook(frame, done, total):
        state.update(done=done, total=total, loading=done < total)
        report(done, total, f'Downloading and measuring SPHEREx images: {done} of {total}')

    def message(line):                              # after the downloads: the pipeline's own steps as progress
        log_lines.append(line)
        text = ' '.join(str(line).split())
        if text and not state['loading'] and state['total']:
            report(state['done'], state['total'], text[:160])

    with _run_lock, settings_override(run_settings(request)), progress_events(message):
        report(0, 0, 'Searching the SPHEREx archive at NASA/IPAC IRSA...')
        pipeline.PROGRESS_HOOK = hook
        try:
            results = run(request, pipeline)
        except RuntimeError as e:
            raise RuntimeError(explained(e, log_lines)) from e
        finally:
            pipeline.PROGRESS_HOOK = None
        n = len(results['frames'])
        report(n, n, 'Making the images...')
        shown = write_frames(results['zooms'], folder, request['patch_px'], results.get('movement'))
        summary = summary_of(results, request, shown, log_lines)
    (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return summary
