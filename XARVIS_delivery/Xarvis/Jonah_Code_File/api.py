"""Entry points for a web app (or any other program): run an analysis and get plain, JSON-ready data back.

    from Jonah_Code_File import api
    result = api.analyze({'coords': '08h55m10.8s -07d14m42s'}, on_progress=print)    # a dict, json.dumps-able
    result, raw = api.analyze({'asteroid': '433'}, settings={'MAX_IMAGES': 40}, return_raw=True)
    png = api.frame_png(raw, 0)                 # the frame T1 as a PNG image (bytes), display-only colour
    objects = api.frame_objects(raw, 0)         # what the sky map found in T1, in that image's pixel coordinates

target: one of {'asteroid': id, 'window': [start, end]}, {'star': name}, {'coords': '...' or [ra, dec]}, or the
settings a results file stores ({'mode': ..., ...}; see results.known_objects).
settings: config.json values to use for this analysis only (e.g. {'MAX_IMAGES': 40, 'USE_PSF': False}).
on_progress(message): called with every progress message (the same lines the terminal shows).
isolated=True: run in a separate process - analyses with different settings can then run at the same time
(settings are module-wide in one process, so in-process analyses run one at a time).

Nothing here opens a window: images are made with visual.render (display-only) and returned as PNG bytes.
"""

import io
import logging
import math
import threading
from contextlib import contextmanager
from datetime import date, datetime

import numpy as np

from . import config as cfg

log = logging.getLogger('Jonah_Code_File')
_run_lock = threading.Lock()                # in one process, one analysis at a time (settings are module-wide)


# ----------------------------------------------------------------------------
# Settings for one analysis
# ----------------------------------------------------------------------------

def _converted(name, value):
    """A JSON-style value in the Python type config.py uses for `name` (units, tuples, sets)."""
    current = getattr(cfg, name)
    if hasattr(current, 'unit') and not hasattr(value, 'unit'):
        return value * current.unit
    if isinstance(current, set) and isinstance(value, (list, tuple, set)):
        return set(value)
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


@contextmanager
def settings_override(settings=None):
    """Use these config values (names as in config.json) for the duration of the block, then restore."""
    settings = settings or {}
    unknown = [name for name in settings if not (name.isupper() and hasattr(cfg, name))]
    if unknown:
        raise KeyError(f'Unknown setting(s): {", ".join(unknown)} (see config.json)')
    saved = {name: getattr(cfg, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(cfg, name, _converted(name, value))
        yield
    finally:
        for name, value in saved.items():
            setattr(cfg, name, value)


@contextmanager
def progress_events(on_progress):
    """Send every progress message to on_progress(message) while the block runs."""
    if on_progress is None:
        yield
        return

    class _Forward(logging.Handler):
        def emit(self, record):
            try:
                on_progress(record.getMessage())
            except Exception:                   # a broken listener must not stop the analysis
                pass

    handler = _Forward(level=logging.INFO)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

def _run(target, show=False):
    from .main import run, run_settings
    target = dict(target)
    if 'mode' in target:
        return run_settings(target, show=show)
    allowed = {'asteroid', 'star', 'coords', 'window', 'max_images', 'ra_deg', 'dec_deg'}
    unknown = set(target) - allowed
    if unknown:
        raise KeyError(f'Unknown target field(s): {", ".join(sorted(unknown))}')
    if 'window' in target:
        target['window'] = tuple(target['window'])
    if isinstance(target.get('coords'), list):
        target['coords'] = tuple(target['coords'])
    return run(show=show, **target)


def analyze(target, settings=None, on_progress=None, isolated=False, return_raw=False):
    """Run an analysis without opening any window and return its JSON-ready summary (see to_json).
    return_raw=True: return (summary, raw results) - the raw results are needed for frame_png / frame_objects
    (not available with isolated=True: they stay in the other process)."""
    if isolated:
        if return_raw:
            raise ValueError('return_raw is not available with isolated=True')
        return _analyze_isolated(target, settings, on_progress)
    with _run_lock, settings_override(settings), progress_events(on_progress):
        raw = _run(target)
        summary = to_json(raw)
    return (summary, raw) if return_raw else summary


def _isolated_job(target, settings, queue):
    """(In the separate process.) Runs the analysis, sending progress messages through `queue`."""
    with settings_override(settings), progress_events(queue.put if queue is not None else None):
        return to_json(_run(target))


def _analyze_isolated(target, settings, on_progress):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    manager = multiprocessing.Manager() if on_progress else None
    queue = manager.Queue() if manager else None
    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_isolated_job, target, settings, queue)
            while True:
                try:
                    return future.result(timeout=0.2) if queue is None else _drain(future, queue, on_progress)
                except TimeoutError:
                    continue
    finally:
        if manager:
            manager.shutdown()


def _drain(future, queue, on_progress):
    """Forward progress messages until the job finishes; then return its result."""
    import queue as queue_module
    while True:
        try:
            on_progress(queue.get(timeout=0.2))
        except queue_module.Empty:
            if future.done():
                while not queue.empty():
                    on_progress(queue.get())
                return future.result()


# ----------------------------------------------------------------------------
# Plain data out
# ----------------------------------------------------------------------------

def jsonable(value):
    """Any value made of numpy / astropy / Python types -> plain JSON types (NaN / inf -> None)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, 'ra') and hasattr(value, 'dec') and getattr(value, 'isscalar', True):   # SkyCoord
        try:
            return {'ra_deg': float(value.ra.deg), 'dec_deg': float(value.dec.deg)}
        except (TypeError, AttributeError):
            pass
    if hasattr(value, 'unit') and hasattr(value, 'value'):                                  # Quantity
        return {'value': jsonable(np.asarray(value.value).tolist()), 'unit': str(value.unit)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


FRAME_FIELDS = ('obs_id', 'date_obs', 'mjd', 'detector', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr',
                'aperture_correction', 'n_bad_in_aperture', 'object_found', 'object_ra', 'object_dec',
                'object_err_arcsec', 'expected_ra', 'expected_dec', 'track', 'track_sep_px', 'position_method',
                'psf_chi2', 'saturated', 'glow', 'alignment_px', 'rotation_deg', 'astrometry_ok', 'moving_flags')
SUMMARY_ARRAYS = ('offset_arcsec', 'd_ra_arcsec', 'd_dec_arcsec', 'error_arcsec', 'residual_arcsec', 'flagged',
                  'detected', 'deviation', 'reference_mjy', 'used')


def to_json(results):
    """The JSON-ready summary of main.run(...)'s results: target, per-frame measurements, movement and brightness
    per frame, the spectrum check, and the sky map's objects (NEW ones with every detection's time and place).
    Pixels are not included (see frame_png)."""
    tracked, frames = results['tracked'], results['frames']
    movement, brightness = results.get('movement') or {}, results.get('brightness') or {}
    out = {
        'target': tracked.get('name'),
        'kind': 'moving' if 'ephemeris' in tracked else 'fixed',
        'n_frames': len(frames),
        'frames': [{key: jsonable(f.get(key)) for key in FRAME_FIELDS} | {'T': i + 1} for i, f in enumerate(frames)],
        'movement': {key: jsonable(movement[key]) for key in SUMMARY_ARRAYS if key in movement}
                    | {key: jsonable(movement.get(key)) for key in ('reference_label', 'flag_label', 'notes')
                       if key in movement},
        'brightness': {key: jsonable(brightness[key]) for key in SUMMARY_ARRAYS + ('scatter', 'source')
                       if key in brightness},
    }
    spectrum = results.get('spectrum')
    if spectrum:
        out['spectrum'] = jsonable({k: v for k, v in spectrum.items() if k not in ('figure',)})
    if results.get('catalog_motion'):
        out['catalog_motion'] = jsonable(results['catalog_motion'])
    sky = results.get('skymap')
    if sky:
        from astropy.time import Time
        out['new_objects'] = [{
            'id': o['id'], 'class': o['kind'], 'description': o['description'], 'ra_deg': o['ra'], 'dec_deg': o['dec'],
            'track': jsonable(o.get('track')),
            'detections': [{'frame': f"T{m['frame'] + 1}", 'ra_deg': m['ra'], 'dec_deg': m['dec'], 'mjd': m['mjd'],
                            'time_utc': Time(m['mjd'], format='mjd').utc.isot[:23], 'significance': m['sigma']}
                           for m in sorted(o['detections'], key=lambda d: d['mjd'])],
        } for o in sky['new_objects']]
        out['known_small_bodies'] = [{'name': b['fullname'], 'kind': b['kind'], 'orbit_class': b['orbit_class'],
                                      'frames': sorted(f'T{i + 1}' for i in b['positions'])} for b in sky['bodies']]
    return jsonable(out)


# ----------------------------------------------------------------------------
# Images for a browser
# ----------------------------------------------------------------------------

def frame_png(raw, index, pretty=True, wide=False):
    """Frame T(index+1) as a PNG image (bytes): display-only colour, sharpening and upscaling (visual.render),
    with the target drawn at the same brightness and colour in every frame - or the plain original pixels in
    grayscale with pretty=False. wide=True: the wide view (downloaded on demand). North is not necessarily up:
    the image is in the frame's own pixel grid (frame_objects gives positions in the same grid)."""
    import matplotlib.image as mpimg
    from .plots import load_wide
    from .visual import anchors, find_partners, render, target_xy
    zooms = raw['zooms']
    zoom = zooms[index]
    if pretty:
        partners = find_partners(zooms)
        colour = anchors(zooms, partners)[0]['colour']
        partner = zooms[partners[index]] if partners[index] is not None else None
        factor = None
        if wide:
            zoom = load_wide(zoom)
            partner = load_wide(partner) if partner is not None else None
            factor = cfg.WIDE_VIEW_UPSCALE
        rgb, _ = render(zoom, partner, factor=factor, anchor={'xy': target_xy(zoom), 'colour': colour})
        image = np.flipud(rgb)                  # PNG rows run top to bottom; the frame's run bottom to top
        buffer = io.BytesIO()
        mpimg.imsave(buffer, image, format='png')
    else:
        from .measure import robust_scatter
        if wide:
            zoom = load_wide(zoom)
        pixels = zoom['image'][np.isfinite(zoom['image'])]
        sky, noise = np.median(pixels), robust_scatter(pixels)
        buffer = io.BytesIO()
        mpimg.imsave(buffer, np.flipud(zoom['image']), format='png', cmap='gray', vmin=sky - 3 * noise,
                     vmax=sky + 15 * noise)
    return buffer.getvalue()


def frame_objects(raw, index):
    """What the sky map found in frame T(index+1), with positions in that frame's (zoom) pixel grid - x to the
    right, y DOWN from the top, in original pixels (multiply by the upscale factor for frame_png's pixels).
    Plus the target: where it was expected and where it was found."""
    zoom = raw['zooms'][index]
    ny = zoom['image'].shape[0]
    wcs = zoom['wcs']

    def place(ra, dec):
        x, y = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        return {'x': x + 0.5, 'y': ny - (y + 0.5), 'ra_deg': float(ra), 'dec_deg': float(dec)}

    objects = [dict(place(e['ra'], e['dec']), kind=e['kind'], label=e['label'], detected=e.get('detected'),
                    info=e.get('info'), new_id=e.get('new_id')) for e in zoom.get('map', [])]
    target = {'expected': place(zoom['expected_ra'], zoom['expected_dec'])}
    if zoom.get('object_found'):
        target['found'] = place(zoom['object_ra'], zoom['object_dec'])
    return jsonable({'frame': f'T{index + 1}', 'image_size': [zoom['image'].shape[1], ny], 'target': target,
                     'objects': objects})
