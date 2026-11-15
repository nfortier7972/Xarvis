"""The website's way into the pipeline: analyse(request, folder, on_progress) runs one analysis with no windows and
writes what the browser's time-lapse needs into `folder`:

    summary.json        the target, and every frame's time, detector, wavelength, signal-to-noise and flags
    frame_<n>.png       every frame (n from 1, in time order)

`request` is what Test_Robert_Code_Folder/app.py makes of the analysis popup's payload (see web_request):
    {'ra_deg', 'dec_deg', 'patch_px', 'max_images' (None: all), 'star' (None, or a Gaia DR3 star with its motion)}

Every PNG is the frame's original pixels resampled onto ONE grid centred on the target, north up and east left
(SPHEREx images come in many rotations; on one grid they line up, so the frames can be played as a time-lapse). One
PNG pixel is one SPHEREx pixel (6.15"); the browser scales it up without smoothing, so the real pixels stay visible.

What a web run does differently from the desktop (through api.settings_override, for the duration of the run only):
no windows, no sky map / field comparison (about a minute of catalog and JPL lookups the popup does not show), the
patch size the popup asks for, and - with a limit on the number of images - images spread over all the survey dates
rather than the first ones. Runs are one at a time (api's lock): the settings are module-wide.
"""

import json
import logging
import os
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
NO_DATA_RGB = (24, 30, 46)                  # pixels with no (good) data: dark blue-grey, not black


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
    return {'ZOOM_SIZE_PIX': zoom, 'CUTOUT_SIZE': cutout_arcmin, 'SPREAD_IMAGES': True, 'FIELD_CHECK': False,
            'SKYMAP_FOR_ALL_TARGETS': False, 'LIVE_VIEW': False, 'ZOOM_ONLY': True,
            'SAVE_RESULTS': False}      # (the desktop's results list is for desktop runs; this run is kept as summary.json)


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


def to_rgb(image):
    """Display-only: sky -> black, brightest -> white, with an asinh stretch (faint things stay visible next to
    bright ones). Each frame is stretched by its own noise: the 102 wavelengths differ too much for one scale."""
    from .measure import robust_scatter
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    rgb = np.empty(image.shape + (3,), dtype=np.uint8)
    rgb[...] = NO_DATA_RGB
    if finite.any():
        values = image[finite]
        sky = np.median(values)
        soft = 3 * max(float(robust_scatter(values)), 1e-12)
        top = max(float(np.percentile(values, 99.8)) - sky, soft)
        level = np.arcsinh(np.clip(image - sky, 0, None) / soft) / np.arcsinh(top / soft)
        grey = (np.clip(np.nan_to_num(level), 0, 1) * 255).astype(np.uint8)
        rgb[finite] = grey[finite][:, None]
    return rgb


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


def write_frames(zooms, folder, size_pix):
    """frame_<n>.png for every zoom. Returns one dict of display facts per frame."""
    from PIL import Image
    shown = []
    for n, zoom in enumerate(zooms, start=1):
        Image.fromarray(to_rgb(resampled(zoom, size_pix))).save(Path(folder) / f'frame_{n}.png')
        shown.append({'png': f'frame_{n}.png', 'flags_in_aperture': flags_in_aperture(zoom)})
    return shown


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

FRAME_FIELDS = ('obs_id', 'mjd', 'detector', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr',
                'n_bad_in_aperture', 'saturated', 'track', 'astrometry_ok')


def summary_of(results, request, shown):
    """The JSON the browser gets when the run is done (also saved as summary.json)."""
    from .api import jsonable
    tracked, frames = results['tracked'], results['frames']
    rows = []
    for n, (frame, extra) in enumerate(zip(frames, shown), start=1):
        row = {'n': n, 'time_utc': Time(frame['mjd'], format='mjd').utc.strftime('%Y-%m-%d %H:%M:%S')}
        row.update({key: jsonable(frame.get(key)) for key in FRAME_FIELDS})
        row.update(extra)
        rows.append(row)
    return {'target': {'name': tracked.get('name'), 'ra_deg': request['ra_deg'], 'dec_deg': request['dec_deg'],
                       'gaia_dr3': (request.get('star') or {}).get('gaia_dr3')},
            'request': request,
            'patch_px': request['patch_px'], 'pixel_arcsec': cfg.PIXEL_ARCSEC,
            'images_available': jsonable(tracked.get('images_available')), 'n_frames': len(rows),
            'frames': rows,
            'source': 'NASA/IPAC IRSA SPHEREx spectral images (' + ', '.join(cfg.COLLECTIONS) + ')'}


def analyse(request, folder, on_progress=None):
    """Run the pipeline for one web request and write summary.json + frame_<n>.png into `folder`.
    on_progress(done, total, message): called while the images download (done and total are 0 until the number
    of images is known). Returns the summary. Raises RuntimeError / ValueError with a message for the user."""
    from . import main as pipeline
    from .api import _run_lock, settings_override
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report = on_progress or (lambda done, total, message: None)

    def hook(frame, done, total):
        report(done, total, f'Downloading and measuring SPHEREx images: {done} of {total}')

    with _run_lock, settings_override(run_settings(request)):
        report(0, 0, 'Searching the SPHEREx archive at NASA/IPAC IRSA...')
        pipeline.PROGRESS_HOOK = hook
        try:
            results = pipeline.run_fixed_target(pipeline_star(request), {'mode': 'web'},
                                                request['max_images'], show=False)
        finally:
            pipeline.PROGRESS_HOOK = None
        n = len(results['frames'])
        report(n, n, 'Making the images...')
        summary = summary_of(results, request, write_frames(results['zooms'], folder, request['patch_px']))
    (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return summary
