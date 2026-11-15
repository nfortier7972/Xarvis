"""Headless entry point for the website: analyse(request, folder, on_progress) runs the SPHEREx pipeline with no windows
and writes everything the browser needs into `folder`:

    summary.json        target, every frame's measurements and flags, movement, spectrum (102 channels), new objects
    frame_<n>.png       every frame (original pixels), in time order
    stack_d<d>.png      inverse-variance stack per detector array
    diff_d<d>.png       last survey visit minus the first, per detector array

The request is what the website's analysis popup sends (see webapp/analysis.py):
    {'mode': 'coordinates', 'ra': .., 'dec': ..}                         a fixed point on the sky
    {'mode': 'gaia_star', 'target': {name, ra, dec, pmra, pmdec, parallax, ...}}   a star moved by its Gaia motion
    {'mode': 'asteroid', 'asteroid': '433', 'window': ['2026-02-15', '2026-06-15']}
plus 'patch_px' (pixels per side, default config.ZOOM_SIZE_PIX) and 'max_images'.

Settings are module globals in config.py, so runs are serialised (one at a time per process) and per-request values
are applied only for the duration of the run.
"""

import contextlib
import json
import math
import threading
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                       # never open windows from the web server

import numpy as np
from astropy import units as u

from . import config as cfg
from . import main as pipeline
from .render import frame_pngs, stacks_and_differences
from .spectrumanalyser import SpectrumAnalyzer
from .stars import gaia_star

_run_lock = threading.Lock()


@contextlib.contextmanager
def _settings(**values):
    """Temporarily change config.py values (restored afterwards, even on errors)."""
    old = {k: getattr(cfg, k) for k in values}
    try:
        for k, v in values.items():
            setattr(cfg, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(cfg, k, v)


def _plain(value):
    """JSON-safe copy: numpy scalars -> numbers, NaN/inf -> None, arrays -> lists; anything else -> str."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if not str(k).startswith('_')}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()] if value.ndim and value.size < 5000 else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, u.Quantity):
        return _plain(value.value)
    return None                                     # WCS, headers, tables, SkyCoord ...: not needed by the browser


FRAME_FIELDS = ('obs_id', 'date_obs', 'mjd', 'detector', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr',
                'n_bad_in_aperture', 'object_ra', 'object_dec', 'object_err_arcsec', 'expected_ra', 'expected_dec',
                'track', 'track_sep_px', 'saturated', 'astrometry_ok', 'psf_fwhm_arcsec', 'rotation_deg', 'moving_flags')


def _summary(results, request, frame_info, images):
    frames, zooms = results['frames'], results['zooms']
    rows = []
    for n, (frame, zoom, info) in enumerate(zip(frames, zooms, frame_info), start=1):
        row = {'T': n, **{k: _plain(frame.get(k)) for k in FRAME_FIELDS if k in frame}, **info}
        rows.append(row)
    spectrum = None
    try:
        s = SpectrumAnalyzer(frames).band_spectrum()
        spectrum = _plain({k: s[k] for k in s})
    except Exception as e:                           # too few measured frames: no spectrum, not an error
        spectrum = {'error': str(e)}
    sky = results.get('skymap') or {}
    new_objects = []
    for obj in (sky.get('new_objects') or []) if isinstance(sky, dict) else []:
        new_objects.append(_plain({k: obj.get(k) for k in ('name', 'kind', 'ra', 'dec', 'mjd', 'detections', 'rate')
                                   if isinstance(obj, dict) and k in obj}))
    tracked = results['tracked']
    return {
        'request': request,
        'target': {'name': tracked.get('name'), 'ra': _plain(tracked['coord'].ra.deg) if 'coord' in tracked else None,
                   'dec': _plain(tracked['coord'].dec.deg) if 'coord' in tracked else None},
        'pixel_arcsec': cfg.PIXEL_ARCSEC, 'patch_px': cfg.ZOOM_SIZE_PIX,
        'frames': rows, 'images': images,
        'movement': _plain(results.get('movement')), 'brightness': _plain(results.get('brightness')),
        'spectrum': spectrum, 'new_objects': new_objects,
        'source': 'NASA/IPAC IRSA SPHEREx spectral images (collections ' + ', '.join(cfg.COLLECTIONS) + ')',
    }


def analyse(request, folder, on_progress=None):
    """Run the pipeline for one request (see the module docstring) and write summary.json + PNGs into `folder`.
    on_progress(done, total, message) is called while images are downloaded and measured. Returns the summary."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mode = request.get('mode')
    patch = int(min(max(int(request.get('patch_px') or cfg.ZOOM_SIZE_PIX), 10), 400))
    max_images = request.get('max_images')
    max_images = int(max_images) if max_images else None

    def hook(frame, done, total):
        if on_progress:
            on_progress(done, total, f'Measured {done} of {total} SPHEREx images')

    with _run_lock, _settings(ZOOM_SIZE_PIX=patch, CUTOUT_SIZE=(patch + 4) * cfg.PIXEL_ARCSEC * u.arcsec,
                              LIVE_VIEW=False, SAVE_RESULTS=True, SPREAD_IMAGES=True):
        pipeline.PROGRESS_HOOK = hook
        try:
            if on_progress:
                on_progress(0, 0, 'Searching the SPHEREx archive at IRSA...')
            if mode == 'coordinates':
                results = pipeline.run_coords((float(request['ra']), float(request['dec'])), max_images, show=False)
            elif mode == 'gaia_star':
                star = gaia_star(request['target'])
                results = pipeline.run_fixed_target(star, {'mode': 'gaia star', 'star': star['name']}, max_images, show=False)
            elif mode == 'asteroid':
                window = tuple(request.get('window') or cfg.ASTEROID_WINDOW)
                results = pipeline.run_asteroid(str(request['asteroid']), window, max_images, show=False)
            else:
                raise ValueError(f'unknown mode {mode!r}')
        finally:
            pipeline.PROGRESS_HOOK = None
        if on_progress:
            on_progress(len(results['frames']), len(results['frames']), 'Making images...')
        info = frame_pngs(results['zooms'], folder, patch)
        images = stacks_and_differences(results['zooms'], folder, patch)
        summary = _summary(results, request, info, images)
    (folder / 'summary.json').write_text(json.dumps(summary))
    return summary
