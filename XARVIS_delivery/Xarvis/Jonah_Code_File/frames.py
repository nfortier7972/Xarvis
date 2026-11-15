"""Download and measure every SPHEREx image of the object, and zoom on the ORIGINAL pixels for display.

Each image is downloaded once (a small cutout) and everything is measured from it right away: wavelength,
brightness, position. Frames are placed using ONLY SPHEREx's pointing: IRSA's coordinate solution (WCS) for
each image, fitted to thousands of catalog stars over the whole 2040 x 2040 detector. Nothing inside our
small cutout is used to line frames up, so real motion or brightness changes in the sky are preserved.
"""

import atexit
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import Any, TypedDict

import numpy as np

from astropy import units as u
from astropy.wcs import WCS

from . import config as cfg
from .irsa import download_cutout, open_cutout
from .measure import (bad_pixel_mask, fit_wings, flag_bits, glow_brightness, locate_object, measure_flux,
                      saturated_core, survey_visits, track_counts)
from .psf import detector_position, encircled, psf_fit

log = logging.getLogger(__name__)


class Frame(TypedDict, total=False):
    """One measured SPHEREx image (the dict every step passes around). Keys by stage:
    make_frame: image, variance, flags, bad (2-D arrays: MJy/sr, (MJy/sr)^2, FLAGS bits, True = ignored),
        saturated, wcs, catalog_xy (object's expected pixel), mjd, date_obs, wavelength_um, detector, detector_xy,
        rotation_deg, obs_id, access_url, astrometry_ok, psf_fwhm_arcsec, flags_header, psf, psf_oversample
    measure_cutout: expected_ra/dec, flux_mjy, flux_err_mjy (aperture-corrected when the PSF is known),
        aperture_correction, n_bad_in_aperture, snr, object_found, object_ra/dec, object_err_arcsec,
        track ('ok' | 'off' | 'undetected' | 'no astrometry'), track_sep_px, position_method
        ('psf' | 'centroid' | 'wings' | None), centroid_ra/dec, psf_chi2, glow, moving_flags
    later: alignment_px (check_alignment), field_unmatched, map (skymap.map_frames)"""
    image: np.ndarray
    variance: np.ndarray
    flags: np.ndarray
    bad: np.ndarray
    wcs: Any
    catalog_xy: tuple
    mjd: float
    wavelength_um: float
    detector: int
    flux_mjy: float
    flux_err_mjy: float
    snr: float
    object_found: bool
    object_ra: float
    object_dec: float
    track: str
    position_method: str


def wavelength_at(hdulist, header, x, y):
    """SPHEREx's wavelength (um) at cutout pixel (x, y): it changes across the detector. Alternate WCS 'W' is a
    WAVE-TAB lookup (FITS -TAB): the pixel's detector position, interpolated in a 9 x 14 grid of wavelengths
    (extension WCS-WAVE). Computed directly - building astropy's WCS for it cost ~19 ms per frame; other
    layouts fall back to astropy."""
    if header.get('CTYPE1W') != 'WAVE-TAB' or 'WCS-WAVE' not in hdulist:
        spectral_wcs = WCS(header=header, fobj=hdulist, key='W')
        spectral_wcs.sip = None
        return float(spectral_wcs.pixel_to_world(x, y)[0].to_value(u.um))
    row = hdulist['WCS-WAVE'].data[0]
    grid_x, grid_y = np.asarray(row['X'], dtype=float), np.asarray(row['Y'], dtype=float)
    values = np.asarray(row['VALUES'], dtype=float)                 # (len(Y), len(X), 2): wavelength, bandwidth
    # FITS intermediate coordinate, 1-based pixel: CRVAL + CDELT * (p - CRPIX)
    psi_x = header.get('CRVAL1W', 1.0) + header.get('CDELT1W', 1.0) * (x + 1 - header['CRPIX1W'])
    psi_y = header.get('CRVAL2W', 1.0) + header.get('CDELT2W', 1.0) * (y + 1 - header['CRPIX2W'])
    fx = float(np.interp(psi_x, grid_x, np.arange(len(grid_x))))      # fractional grid index (linear in between)
    fy = float(np.interp(psi_y, grid_y, np.arange(len(grid_y))))
    x0, y0 = min(int(fx), len(grid_x) - 2), min(int(fy), len(grid_y) - 2)
    tx, ty = fx - x0, fy - y0
    w = values[..., 0]
    return float(w[y0, x0] * (1 - tx) * (1 - ty) + w[y0, x0 + 1] * tx * (1 - ty)
                 + w[y0 + 1, x0] * (1 - tx) * ty + w[y0 + 1, x0 + 1] * tx * ty)


def _position_angle(ra1, dec1, ra2, dec2):
    """Position angle (deg, east of north) from point 1 to point 2 (degrees in)."""
    ra1, dec1, ra2, dec2 = np.radians([ra1, dec1, ra2, dec2])
    pa = np.arctan2(np.sin(ra2 - ra1) * np.cos(dec2),
                    np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(ra2 - ra1))
    return float(np.degrees(pa) % 360)


def make_frame(hdulist, coord, row, moving=False):
    """One frame as a plain dict (see Frame): the cutout image plus everything later steps need.
    coord = (ra, dec) in degrees (or a SkyCoord). moving=True: the object is moving (see bad_pixel_mask)."""
    header = hdulist['IMAGE'].header
    wcs = WCS(header)       # SPHEREx's pointing for this image
    ra, dec = (coord.ra.deg, coord.dec.deg) if hasattr(coord, 'ra') else coord
    x, y = (float(v) for v in wcs.world_to_pixel_values(ra, dec))

    # wavelength at the object (it changes across the detector), where the object landed on the full detector,
    # and how the detector was rotated on the sky: together these describe SPHEREx's pointing
    wavelength = wavelength_at(hdulist, header, x, y)
    detector_x, detector_y = detector_position(header, x, y)
    (ra0, ra1), (dec0, dec1) = wcs.pixel_to_world_values([x, x], [y, y + 1])
    rotation_deg = _position_angle(ra0, dec0, ra1, dec1)

    image = hdulist['IMAGE'].data.astype(float) - hdulist['ZODI'].data          # MJy/sr, zodi subtracted
    flags = hdulist['FLAGS'].data.astype(np.int64)
    psf = hdulist['PSF'] if 'PSF' in hdulist else None
    return {
        'image': image,
        'variance': hdulist['VARIANCE'].data.astype(float),
        'flags': flags,
        'bad': bad_pixel_mask(image, flags, hdulist['FLAGS'].header, moving),  # ignored in all measurements
        'saturated': saturated_core(image, flags, hdulist['FLAGS'].header, float(x), float(y)),
        'wcs': wcs,
        'catalog_xy': (float(x), float(y)),                # the object's catalog position in this cutout
        'mjd': float(header['MJD-AVG']),
        'date_obs': header['DATE-OBS'],
        'wavelength_um': wavelength,
        'detector': int(header['DETECTOR']),
        'detector_xy': (float(detector_x), float(detector_y)),
        'rotation_deg': float(rotation_deg),
        'obs_id': str(row['obs_id']),
        'access_url': str(row['access_url']),              # (for the wide view, fetched on demand)
        # image metadata used to judge the tracking: was SPHEREx's pointing verified against the stars
        # (astrometry.net solution), and how sharp the image is
        'astrometry_ok': bool(header.get('ASTRONETSOLVED', True)),
        'psf_fwhm_arcsec': float(header.get('PSF_FWHM', np.nan)),
        'flags_header': hdulist['FLAGS'].header,
        # SPHEREx's own PSF at this spot (see psf.py) - None for cutouts cached before it was kept
        'psf': np.asarray(psf.data, dtype=float) if psf is not None else None,
        'psf_oversample': int(psf.header.get('OVERSAMP', 5)) if psf is not None else None,
    }


def _coord_of(row, coord):
    """The object's (ra, dec) in degrees for this image: `coord`, or the row's own position (an asteroid's
    ephemeris, or a nearby star moved by its proper motion)."""
    if coord is None:
        return float(row['target_ra']), float(row['target_dec'])
    return (float(coord.ra.deg), float(coord.dec.deg)) if hasattr(coord, 'ra') else tuple(coord)


def load_frame(row, coord=None, moving=None):
    """Download one image's cutout and measure the object in it (runs in parallel, one per image).

    coord = the object's sky position. None: its position at this exposure is in the row ('target_ra' /
    'target_dec' - an asteroid's ephemeris, or a nearby star moved by its proper motion), and the cutout is
    centered there. moving = treat it as a fast-moving object for the bad-pixel flags (see bad_pixel_mask);
    default: True when coord is None (an asteroid).
    The measuring runs in a worker process when MEASURE_WORKERS > 0 (several CPU cores at once), otherwise in
    this process one frame at a time (see _measuring_turn)."""
    from astropy.coordinates import SkyCoord
    moving = (coord is None) if moving is None else moving
    radec = _coord_of(row, coord)
    info = {'obs_id': str(row['obs_id']), 'access_url': str(row['access_url'])}
    download = download_cutout(info['access_url'], SkyCoord(*radec, unit='deg'), cfg.CUTOUT_SIZE)  # many at once
    pool = _measuring_pool()
    if pool is not None:
        return pool.submit(measure_cutout, download, radec, info, moving).result()
    with _measuring_turn():                                       # ...but measured one at a time in this process
        return measure_cutout(download, radec, info, moving)


def measure_cutout(download, radec, info, moving):
    """Parse a downloaded cutout and measure the object in it. Plain inputs only (bytes, numbers, a dict), so it
    can run in another process. radec = where the object should be (degrees)."""
    frame = make_frame(open_cutout(*download), radec, info, moving)
    frame['expected_ra'], frame['expected_dec'] = radec                   # where it should be
    frame['flux_mjy'], frame['flux_err_mjy'], frame['n_bad_in_aperture'] = measure_flux(frame, radec)
    frame['snr'] = frame['flux_mjy'] / frame['flux_err_mjy'] if frame['flux_err_mjy'] > 0 else np.nan
    frame.update(locate_object(frame))
    frame['position_method'] = 'centroid' if frame.get('track') in ('ok', 'off', 'no astrometry') else None
    if cfg.USE_PSF and frame['psf'] is not None:
        _apply_psf(frame)
    if frame['saturated']:                  # too bright: the core is useless, the outer glow isn't
        frame['flux_mjy'] = frame['flux_err_mjy'] = frame['snr'] = np.nan     # brightness: not measurable
        fit = fit_wings(frame, *frame['catalog_xy'])
        frame['position_method'] = None
        frame['object_found'] = fit is not None
        if fit is not None:
            sky = frame['wcs'].pixel_to_world(fit[0], fit[1])
            separation = float(np.hypot(fit[0] - frame['catalog_xy'][0], fit[1] - frame['catalog_xy'][1]))
            track = ('no astrometry' if not frame['astrometry_ok'] else
                     'ok' if separation <= cfg.TRACK_TOLERANCE_PIX else 'off')
            frame.update(object_ra=float(sky.ra.deg), object_dec=float(sky.dec.deg), position_method='wings',
                         object_err_arcsec=float(np.hypot(fit[2], fit[3]) * cfg.PIXEL_ARCSEC),
                         track=track, track_sep_px=separation,
                         glow=glow_brightness(frame, fit[0], fit[1]))   # RELATIVE brightness only
    if moving and frame['object_found']:    # SPHEREx's own sign of a moving source
        x, y = frame['catalog_xy']
        yy, xx = np.indices(frame['flags'].shape)
        near = np.hypot(xx - x, yy - y) <= cfg.APERTURE_RADIUS_PIX
        bits = flag_bits(frame['flags_header'], {'OUTLIER', 'TRANSIENT'})
        frame['moving_flags'] = int(((frame['flags'][near] & bits) != 0).sum())
    return frame


def _apply_psf(frame):
    """With SPHEREx's PSF known (psf.py): divide the flux by the fraction of a point source's light that the
    measuring circle catches (so wavelengths with blurrier images compare fairly), and refine a centroid position
    by fitting the PSF. The centroid is kept in centroid_ra / centroid_dec for comparison."""
    fraction = encircled(frame, *frame['catalog_xy'])
    if fraction:
        frame['aperture_correction'] = fraction
        frame['flux_mjy'] /= fraction
        frame['flux_err_mjy'] /= fraction
    if frame.get('position_method') != 'centroid':
        return
    x, y = (float(v) for v in frame['wcs'].world_to_pixel_values(frame['object_ra'], frame['object_dec']))
    fit = psf_fit(frame, x, y)
    if fit is None:
        return
    ra, dec = (float(v) for v in frame['wcs'].pixel_to_world_values(fit[0], fit[1]))
    separation = float(np.hypot(fit[0] - frame['catalog_xy'][0], fit[1] - frame['catalog_xy'][1]))
    if frame['track'] in ('ok', 'off'):          # the tolerance check, now with the better position
        frame['track'] = 'ok' if separation <= cfg.TRACK_TOLERANCE_PIX else 'off'
    frame.update(centroid_ra=frame['object_ra'], centroid_dec=frame['object_dec'], object_ra=ra, object_dec=dec,
                 object_err_arcsec=float(np.hypot(fit[2], fit[3]) * cfg.PIXEL_ARCSEC), track_sep_px=separation,
                 position_method='psf', psf_chi2=fit[4])


# ----------------------------------------------------------------------------
# Measuring on several CPU cores (MEASURE_WORKERS processes)
# ----------------------------------------------------------------------------

_process_pool = None
_pool_settings = None
_pool_lock = threading.Lock()


def _settings_snapshot():
    return {name: getattr(cfg, name) for name in dir(cfg) if name.isupper()}


def _init_worker(settings):
    """A worker process starts from config.json; give it the parent's settings as they are now (they may have
    been changed at run time, e.g. by api.analyze(settings=...))."""
    for name, value in settings.items():
        setattr(cfg, name, value)


def _measuring_pool():
    """The worker processes for measuring (None: MEASURE_WORKERS = 0, measure in this process). Made once and
    reused; remade if the settings changed since."""
    global _process_pool, _pool_settings
    if not cfg.MEASURE_WORKERS:
        return None
    with _pool_lock:
        settings = _settings_snapshot()
        if _process_pool is None or repr(settings) != repr(_pool_settings):
            if _process_pool is not None:
                _process_pool.shutdown(wait=False, cancel_futures=True)
            _process_pool = ProcessPoolExecutor(max_workers=int(cfg.MEASURE_WORKERS), initializer=_init_worker,
                                                initargs=(settings,))
            _pool_settings = settings
        return _process_pool


@atexit.register
def _close_pool():
    if _process_pool is not None:
        _process_pool.shutdown(wait=False, cancel_futures=True)


# Measuring a frame is Python/numpy work, and Python threads can't do that in parallel (only one thread runs
# Python code at a time - the "GIL"). Letting 16 threads try anyway gains nothing (measured: same loading time),
# and it starves the main thread: a live-view redraw that takes 0.16 s on its own took 6-9 s. So frames are
# measured one at a time, and a live display can pause the measuring (not the downloads) while it redraws.
_compute_slot = threading.Lock()
_measuring_allowed = threading.Event()
_measuring_allowed.set()


@contextmanager
def _measuring_turn():
    """A worker's turn to parse + measure one frame. Python's lock isn't first-come-first-served, so a waiting
    display could keep losing it to the workers; instead, while a display is waiting, a worker that gets the
    lock hands it straight back and waits."""
    while True:
        _measuring_allowed.wait()
        _compute_slot.acquire()
        if _measuring_allowed.is_set():
            break
        _compute_slot.release()                 # a display is waiting: give way
    try:
        yield
    finally:
        _compute_slot.release()


@contextmanager
def pause_measuring():
    """Pause the measuring threads for a moment - e.g. while a live display redraws - so the display gets the
    CPU (downloads carry on). Waits at most for the frame being measured right now to finish (~40 ms)."""
    _measuring_allowed.clear()                  # workers stop taking new frames...
    try:
        with _compute_slot:                     # ...and the current one finishes
            yield
    finally:
        _measuring_allowed.set()


def load_all_frames(coord, images, max_images=cfg.MAX_IMAGES, on_frame=None, moving=None):
    """Download and measure every image (or the first `max_images` in time), in parallel. Sorted by time.
    coord = the star's sky position, or None for a moving object (see load_frame).

    on_frame: optional function called as on_frame(frame, done, total) the moment each frame is measured
    (in arrival order, not time order), and as on_frame(None, done, total) about 4 times a second while
    waiting - so a live display can fill in, and stay responsive, while the rest is still downloading.
    It is always called from this (the calling) thread."""
    rows = list(images if max_images is None else images[:max_images])
    frames, failures = [], []
    start = time.time()
    log.info(f'Downloading and measuring {len(rows)} images ({cfg.MAX_CONCURRENT_DOWNLOADS} at a time)...')

    with ThreadPoolExecutor(max_workers=cfg.MAX_CONCURRENT_DOWNLOADS) as pool:
        pending = {pool.submit(load_frame, row, coord, moving) for row in rows}
        done = 0
        while pending:
            finished, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in finished:
                done += 1
                try:
                    frame = future.result()
                except Exception as e:                      # one bad image shouldn't stop the whole run
                    first = next((line.strip() for line in str(e).splitlines() if line.strip()), '')
                    failures.append(f'{type(e).__name__}: {first}' if first else type(e).__name__)
                    frame = None
                if frame is not None:
                    frames.append(frame)
                    if on_frame and frame['object_found']:
                        on_frame(frame, done, len(rows))
                if done % 50 == 0 or done == len(rows):
                    elapsed = time.time() - start
                    log.info(f'  {done}/{len(rows)} images  ({done / elapsed:.1f}/s, '
                          f'{(len(rows) - done) * elapsed / done:.0f} s left)')
            if on_frame:
                on_frame(None, done, len(rows))             # nothing new: just keep the display responsive

    if failures:
        log.info(f'  skipped {len(failures)} images that could not be used, e.g.: {failures[0][:100]}')
    saturated = [f for f in frames if f['saturated']]
    usable = [f for f in frames if f['object_found'] and (np.isfinite(f['flux_mjy']) or f['saturated'])]
    lost_saturated = sum(1 for f in saturated if not f['object_found'])
    lost_edge = len(frames) - len(usable) - lost_saturated
    if lost_edge:
        log.info(f'  {lost_edge} images had the object too close to the edge to measure')
    if saturated:
        log.info(f'  {len(saturated)} frames are SATURATED (object too bright for SPHEREx): position fitted to its '
              f'unsaturated outer glow, brightness not measurable' + (f'; {lost_saturated} could not be fitted' if lost_saturated else ''))
    unreliable = sum(f['n_bad_in_aperture'] > 0 and not f['saturated'] for f in usable)
    if unreliable:
        log.info(f'  {unreliable} frames have a bad pixel (glitch/hot pixel) inside the measurement circle: '
              f'shown, but not used for flags or statistics')
    log.info(f'  tracking: {track_counts(usable)}  (only "ok" = a source peak within {cfg.TRACK_TOLERANCE_PIX} px of the '
          f'expected position, with no flagged pixel deciding it - is used for positions)')
    # time order; the 6 detectors expose simultaneously (same timestamp), so break ties by detector -
    # otherwise T-numbers would depend on which download happened to finish first
    usable.sort(key=lambda f: (f['mjd'], f['detector']))
    return usable


PIXEL_ARRAYS = ('image', 'variance', 'flags', 'bad')     # the per-pixel arrays of a frame


def zoom_original(frame, size_pix=cfg.ZOOM_SIZE_PIX, release_original=False):
    """Crop the display zoom from the original pixels (nothing resampled), centered on the object.
    Near the edge of the detector the zoom is cut off rather than padded.

    Returns a new dict: the frame's measurements (time, wavelength, flux, position ...) plus the cropped
    pixel arrays, the cropped coordinates (`wcs`) and the object's catalog position inside the zoom.

    release_original=True: the full-size pixel arrays are deleted from `frame` once the crop is made, so
    only the zoom stays in memory (the frame keeps its measurements). The crop is copied for this, because
    a plain slice is only a view that keeps the whole original array alive. Do it once per frame, and only
    when nothing needs the full-size pixels any more."""
    x, y = frame['catalog_xy']
    cx, cy = int(np.round(x)), int(np.round(y))
    ny, nx = frame['image'].shape
    x0, y0 = max(cx - size_pix // 2, 0), max(cy - size_pix // 2, 0)
    x1, y1 = min(cx - size_pix // 2 + size_pix, nx), min(cy - size_pix // 2 + size_pix, ny)

    crop = {key: frame[key][y0:y1, x0:x1] for key in PIXEL_ARRAYS}
    if release_original:
        crop = {key: pixels.copy() for key, pixels in crop.items()}

    zoom = dict(frame)                                      # measurements carry over unchanged
    zoom.update(crop, wcs=frame['wcs'][y0:y1, x0:x1], catalog_xy=(x - x0, y - y0))
    if release_original:
        for key in PIXEL_ARRAYS:
            del frame[key]
    return zoom


def pixel_sky_positions(frame):
    """RA/Dec (deg) of every pixel center, from that frame's own pointing - for pixel-by-pixel comparisons
    between frames (which can be rotated relative to each other)."""
    yy, xx = np.indices(frame['image'].shape)
    sky = frame['wcs'].pixel_to_world(xx, yy)
    return sky.ra.deg, sky.dec.deg


def pointing_summary(frames):
    """How SPHEREx's pointing varied over the dataset."""
    days = np.array([f['mjd'] for f in frames]) - frames[0]['mjd']
    visits = 1 + int(survey_visits(days).max())             # survey passes, separated by months
    x1, y1 = frames[0]['detector_xy']
    shifts = [np.hypot(f['detector_xy'][0] - x1, f['detector_xy'][1] - y1) for f in frames]
    rotations = sorted({int(np.round(f['rotation_deg'])) % 360 for f in frames})
    detectors = len({f['detector'] for f in frames})
    log.info(f'\n{len(frames)} frames over {days[-1]:.0f} days in {visits} survey visits, {detectors} detectors '
          f'({min(f["wavelength_um"] for f in frames):.2f} - {max(f["wavelength_um"] for f in frames):.2f} um)')
    log.info(f'SPHEREx pointing: the object fell up to {max(shifts):.0f} px ({max(shifts) * cfg.PIXEL_ARCSEC / 60:.0f}\') '
          f'from its T1 detector position; detector orientations used: {rotations} deg')
