"""Download and measure every SPHEREx image of the object, and zoom on the ORIGINAL pixels for display.

Each image is downloaded once (a small cutout) and everything is measured from it right away: wavelength,
brightness, position. Frames are placed using ONLY SPHEREx's pointing: IRSA's coordinate solution (WCS) for
each image, fitted to thousands of catalog stars over the whole 2040 x 2040 detector. Nothing inside our
small cutout is used to line frames up, so real motion or brightness changes in the sky are preserved.
"""

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from . import config as cfg
from .irsa import download_cutout, open_cutout
from .measure import (bad_pixel_mask, fit_wings, flag_bits, glow_brightness, locate_object, measure_flux,
                      saturated_core, track_counts)


def make_frame(hdulist, coord, row, moving=False):
    """One frame as a plain dict: the cutout image plus everything later steps need.
    moving=True: the object is moving (see bad_pixel_mask)."""
    header = hdulist['IMAGE'].header
    wcs = WCS(header)       # SPHEREx's pointing for this image

    # wavelength at the object: SPHEREx's wavelength changes across the detector.
    # The spectral WCS is alternate WCS "W" and uses the WCS-WAVE lookup table.
    spectral_wcs = WCS(header=header, fobj=hdulist, key='W')
    spectral_wcs.sip = None
    x, y = wcs.world_to_pixel(coord)
    wavelength, _ = spectral_wcs.pixel_to_world(x, y)

    # where the object landed on the full detector (alternate WCS "A" = 0-based detector pixels)
    # and how the detector was rotated on the sky: together these describe SPHEREx's pointing
    detector_x, detector_y = WCS(header, key='A').pixel_to_world_values(x, y)
    rotation_deg = wcs.pixel_to_world(x, y).position_angle(wcs.pixel_to_world(x, y + 1)).deg

    image = hdulist['IMAGE'].data.astype(float) - hdulist['ZODI'].data          # MJy/sr, zodi subtracted
    flags = hdulist['FLAGS'].data.astype(np.int64)
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
        'wavelength_um': float(wavelength.to_value(u.um)),
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
    }


def load_frame(row, coord=None, moving=None):
    """Download one image's cutout and measure the object in it (runs in parallel, one per image).

    coord = the object's sky position. None: its position at this exposure is in the row ('target_ra' /
    'target_dec' - an asteroid's ephemeris, or a nearby star moved by its proper motion), and the cutout is
    centered there. moving = treat it as a fast-moving object for the bad-pixel flags (see bad_pixel_mask);
    default: True when coord is None (an asteroid)."""
    moving = (coord is None) if moving is None else moving
    if coord is None:
        coord = SkyCoord(float(row['target_ra']), float(row['target_dec']), unit='deg')
    download = download_cutout(str(row['access_url']), coord, cfg.CUTOUT_SIZE)   # many downloads at once...
    with _measuring_turn():                                                    # ...but parsed + measured one at a time
        frame = make_frame(open_cutout(*download), coord, row, moving)
        frame['expected_ra'], frame['expected_dec'] = float(coord.ra.deg), float(coord.dec.deg)   # where it should be
        frame['flux_mjy'], frame['flux_err_mjy'], frame['n_bad_in_aperture'] = measure_flux(frame, coord)
        frame['snr'] = frame['flux_mjy'] / frame['flux_err_mjy'] if frame['flux_err_mjy'] > 0 else np.nan
        frame.update(locate_object(frame))
        frame['position_method'] = 'centroid' if frame.get('track') in ('ok', 'off', 'no astrometry') else None
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
        if moving and frame['object_found']:    # SPHEREx's own sign of a moving source (see config.py)
            x, y = frame['catalog_xy']
            yy, xx = np.indices(frame['flags'].shape)
            near = np.hypot(xx - x, yy - y) <= cfg.APERTURE_RADIUS_PIX
            bits = flag_bits(frame['flags_header'], {'OUTLIER', 'TRANSIENT'})
            frame['moving_flags'] = int(((frame['flags'][near] & bits) != 0).sum())
    return frame


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
    print(f'Downloading and measuring {len(rows)} images ({cfg.MAX_CONCURRENT_DOWNLOADS} at a time)...')

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
                    failures.append(str(e).splitlines()[0] if str(e) else type(e).__name__)
                    frame = None
                if frame is not None:
                    frames.append(frame)
                    if on_frame and frame['object_found']:
                        on_frame(frame, done, len(rows))
                if done % 50 == 0 or done == len(rows):
                    elapsed = time.time() - start
                    print(f'  {done}/{len(rows)} images  ({done / elapsed:.1f}/s, '
                          f'{(len(rows) - done) * elapsed / done:.0f} s left)')
            if on_frame:
                on_frame(None, done, len(rows))             # nothing new: just keep the display responsive

    if failures:
        print(f'  skipped {len(failures)} images that could not be used, e.g.: {failures[0][:100]}')
    saturated = [f for f in frames if f['saturated']]
    usable = [f for f in frames if f['object_found'] and (np.isfinite(f['flux_mjy']) or f['saturated'])]
    lost_saturated = sum(1 for f in saturated if not f['object_found'])
    lost_edge = len(frames) - len(usable) - lost_saturated
    if lost_edge:
        print(f'  {lost_edge} images had the object too close to the edge to measure')
    if saturated:
        print(f'  {len(saturated)} frames are SATURATED (object too bright for SPHEREx): position fitted to its '
              f'unsaturated outer glow, brightness not measurable' + (f'; {lost_saturated} could not be fitted' if lost_saturated else ''))
    unreliable = sum(f['n_bad_in_aperture'] > 0 and not f['saturated'] for f in usable)
    if unreliable:
        print(f'  {unreliable} frames have a bad pixel (glitch/hot pixel) inside the measurement circle: '
              f'shown, but not used for flags or statistics')
    print(f'  tracking: {track_counts(usable)}  (only "ok" = a source peak within {cfg.TRACK_TOLERANCE_PIX} px of the '
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
    visits = 1 + int(np.sum(np.diff(days) > 30))           # survey passes, separated by months
    x1, y1 = frames[0]['detector_xy']
    shifts = [np.hypot(f['detector_xy'][0] - x1, f['detector_xy'][1] - y1) for f in frames]
    rotations = sorted({int(np.round(f['rotation_deg'])) % 360 for f in frames})
    detectors = len({f['detector'] for f in frames})
    print(f'\n{len(frames)} frames over {days[-1]:.0f} days in {visits} survey visits, {detectors} detectors '
          f'({min(f["wavelength_um"] for f in frames):.2f} - {max(f["wavelength_um"] for f in frames):.2f} um)')
    print(f'SPHEREx pointing: the object fell up to {max(shifts):.0f} px ({max(shifts) * cfg.PIXEL_ARCSEC / 60:.0f}\') '
          f'from its T1 detector position; detector orientations used: {rotations} deg')
