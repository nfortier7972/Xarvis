"""Measurements: brightness, position, alignment check, movement, brightness changes, spectrum check."""

import logging
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_area

from . import config as cfg

log = logging.getLogger(__name__)


def reliable_frames(frames):
    """True for frames with no bad pixel inside the measurement circle. Unreliable frames are still shown,
    but can't trigger a flag or influence the statistics."""
    return np.array([f['n_bad_in_aperture'] == 0 for f in frames])


def usable_frames(frames):
    """The frames the object's POSITION can be trusted in: reliable (no bad pixel in the circle) and detected
    (signal-to-noise >= DETECTION_SNR at its expected position) - or saturated, with the position fitted to the
    star's unsaturated outer glow (see fit_wings)."""
    snr = np.array([f.get('snr', np.inf) for f in frames], dtype=float)
    detected = reliable_frames(frames) & np.isfinite(snr) & (snr >= cfg.DETECTION_SNR)
    from_wings = np.array([f.get('position_method') == 'wings' for f in frames], dtype=bool)
    on_track = np.array([f.get('track', 'ok') == 'ok' for f in frames], dtype=bool)   # see locate_object
    return (detected | from_wings) & on_track


def survey_visits(mjd, gap_days=None):
    """Survey visit number (0, 1, 2 ...) of each time (MJD, any order): SPHEREx observes a patch of sky for a few
    days to weeks, then returns months later - a gap longer than VISIT_GAP_DAYS starts a new visit. The one
    definition of a visit used everywhere."""
    gap_days = cfg.VISIT_GAP_DAYS if gap_days is None else gap_days
    mjd = np.asarray(mjd, dtype=float)
    if not mjd.size:
        return np.zeros(0, dtype=int)
    order = np.argsort(mjd)
    visit = np.empty(len(mjd), dtype=int)
    visit[order] = np.r_[0, np.cumsum(np.diff(mjd[order]) > gap_days)]
    return visit


def robust_scatter(values):
    """Typical spread (1 sigma equivalent) that ignores outliers: 1.4826 x median absolute deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return 1.4826 * np.median(np.abs(values - np.median(values))) if values.size else np.nan


# ----------------------------------------------------------------------------
# Bad pixels
# ----------------------------------------------------------------------------

def flag_bits(flags_header, names):
    """The FLAGS bit mask for the given flag names (bit numbers are read from the header: MSKNnn = name)."""
    bits = 0
    for key in flags_header['MSKN*']:
        if flags_header[key] in names:
            bits |= 1 << int(key[4:])
    return bits


def saturated_core(image, flags, flags_header, x, y, radius=1.5):
    """True if the pixels within `radius` of (x, y) are saturated (saturation flags, or no value at all)."""
    yy, xx = np.indices(image.shape)
    core = np.hypot(xx - x, yy - y) <= radius
    saturated = ((flags & flag_bits(flags_header, cfg.SATURATION_FLAG_NAMES)) != 0) | ~np.isfinite(image)
    return bool(saturated[core].any())


def fit_wings(frame, x0, y0):
    """Position of a SATURATED star from its unsaturated outer glow.

    The flat-topped, bloomed core is useless, so: take the good pixels between WING_RADII_PIX from the guess
    (x0, y0) and fit a symmetric (Gaussian) profile plus a flat background, by brute-force search over centre
    and width with the brightness and background solved exactly at each step. A symmetric model's centre isn't
    pulled by which pixels happen to be missing (a plain centroid would be).

    Returns (x, y, err_x, err_y) or None.
    For very bright stars (e.g. R Dra) the good pixels may be too few within WING_RADII_PIX: then a wider ring
    (out to WING_OUTER_MAX_PIX) is used.
    """
    image, good = frame['image'], ~frame['bad'] & np.isfinite(frame['image'])
    yy, xx = np.indices(image.shape)
    r = np.hypot(xx - x0, yy - y0)
    for outer in (cfg.WING_RADII_PIX[1], cfg.WING_OUTER_MAX_PIX):
        use = good & (r >= cfg.WING_RADII_PIX[0]) & (r <= outer)
        if use.sum() >= 25:
            break
    else:
        return None
    px, py, values = xx[use].astype(float), yy[use].astype(float), image[use].astype(float)

    def search(xs, ys, widths):
        X, Y, W = (a.ravel() for a in np.meshgrid(xs, ys, widths, indexing='ij'))
        model = np.exp(-((px[None, :] - X[:, None]) ** 2 + (py[None, :] - Y[:, None]) ** 2) / (2 * W[:, None] ** 2))
        m_mean, v_mean = model.mean(axis=1), values.mean()
        dm = model - m_mean[:, None]
        amplitude = (dm * (values - v_mean)).sum(axis=1) / np.maximum((dm ** 2).sum(axis=1), 1e-30)
        background = v_mean - amplitude * m_mean
        chi2 = ((values[None, :] - amplitude[:, None] * model - background[:, None]) ** 2).sum(axis=1)
        chi2[amplitude <= 0] = np.inf
        return X, Y, W, chi2, amplitude

    X, Y, W, chi2, _ = search(x0 + np.arange(-2, 2.01, 0.1), y0 + np.arange(-2, 2.01, 0.1),
                              np.array([0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0]))
    best = int(np.argmin(chi2))
    if not np.isfinite(chi2[best]):
        return None
    steps = np.arange(-0.15, 0.1501, 0.01)
    X, Y, W, chi2, amplitude = search(X[best] + steps, Y[best] + steps, W[best] * np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    best = int(np.argmin(chi2))
    x, y, width = X[best], Y[best], W[best]

    # uncertainty from how fast the fit gets worse away from the best centre (chi2 curvature), scaled by the
    # actual scatter of the pixels around the fit
    noise2 = chi2[best] / max(use.sum() - 4, 1)
    errors = []
    for along, other, value in ((X, Y, x), (Y, X, y)):
        line = (np.isclose(other, y if along is X else x)) & np.isclose(W, width)
        if line.sum() >= 5:
            curvature = np.polyfit(along[line] - value, chi2[line], 2)[0]
            errors.append(np.sqrt(noise2 / curvature) if curvature > 0 else np.nan)
        else:
            errors.append(np.nan)
    if abs(x - x0) >= 1.99 or abs(y - y0) >= 1.99:     # ran into the edge of the search: not trustworthy
        return None
    return float(x), float(y), float(errors[0]), float(errors[1])


def glow_brightness(frame, x, y):
    """RELATIVE brightness of a saturated star (MJy/sr): the median of its good pixels in a ring GLOW_RING_PIX out
    from (x, y), minus the sky (sigma-clipped median further out). Proportional to the star's brightness at a given
    wavelength, so it allows comparing frames even though the core is saturated. Tested on Barnard's Star (steady):
    it scatters ~9% per frame (GLOW_PRECISION) - a Gaussian-profile fit scattered 34%, so this simpler measure is
    used. NaN if too few good pixels."""
    image, good = frame['image'], ~frame['bad'] & np.isfinite(frame['image'])
    yy, xx = np.indices(image.shape)
    r = np.hypot(xx - x, yy - y)
    ring = good & (r >= cfg.GLOW_RING_PIX[0]) & (r <= cfg.GLOW_RING_PIX[1])
    sky = good & (r >= cfg.GLOW_SKY_PIX[0]) & (r <= cfg.GLOW_SKY_PIX[1])
    if ring.sum() < 10 or sky.sum() < 20:
        return np.nan
    return float(np.median(image[ring]) - sigma_clipped_stats(image[sky], sigma=3.0)[1])


def bad_pixel_mask(image, flags, flags_header, moving=False):
    """True where a pixel can't be trusted: flagged bad by SPHEREx, not a number, or a lone spike that the
    pipeline missed (one pixel far brighter than all 8 of its neighbors - real sources span several pixels).
    moving=True: the object is moving, so the pipeline's OUTLIER / TRANSIENT flags (which it puts on a moving
    source) are not counted as bad."""
    bad_names = cfg.BAD_FLAG_NAMES_MOVING if moving else cfg.BAD_FLAG_NAMES
    bad = ((flags & flag_bits(flags_header, bad_names)) != 0) | ~np.isfinite(image)

    background = np.nanmedian(image)
    noise = robust_scatter(image)
    padded = np.pad(image - background, 1, constant_values=np.nan)
    ny, nx = image.shape
    neighbors = np.stack([padded[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    brightest_neighbor = np.nanmax(neighbors, axis=0).clip(min=noise)
    above = image - background
    spikes = (above > cfg.SPIKE_MIN_SIGMA * noise) & (above > cfg.SPIKE_NEIGHBOR_RATIO * brightest_neighbor)
    return bad | spikes


# ----------------------------------------------------------------------------
# Brightness and position in one frame (run while downloading, one call per image)
# ----------------------------------------------------------------------------

def measure_flux(frame, coord):
    """Brightness (mJy): sum inside the measurement circle minus the local background, measured in the
    ring around it (sigma-clipped so neighboring stars in the ring are ignored). Bad pixels are left out;
    returns (flux, error, number of bad pixels inside the circle)."""
    ra, dec = (coord.ra.deg, coord.dec.deg) if hasattr(coord, 'ra') else coord     # SkyCoord or (ra, dec) degrees
    x, y = frame['wcs'].world_to_pixel_values(ra, dec)
    return aperture_flux(frame, float(x), float(y))


def aperture_flux(frame, x, y):
    """measure_flux at pixel position (x, y) of the frame."""
    yy, xx = np.indices(frame['image'].shape)
    r = np.hypot(xx - x, yy - y)
    image, variance, good = frame['image'], frame['variance'], ~frame['bad']
    circle = r <= cfg.APERTURE_RADIUS_PIX
    in_aperture = circle & good
    in_annulus = (r >= cfg.ANNULUS_PIX[0]) & (r <= cfg.ANNULUS_PIX[1]) & good
    n_bad = int((circle & ~good).sum())
    if in_annulus.sum() < 5 or circle.sum() < 9 or in_aperture.sum() == 0:
        return np.nan, np.nan, n_bad

    _, background, background_std = sigma_clipped_stats(image[in_annulus], sigma=3.0)
    n = in_aperture.sum()
    flux = np.sum(image[in_aperture] - background)
    flux_var = np.sum(variance[in_aperture]) + (n * 1.2533 * background_std / np.sqrt(in_annulus.sum())) ** 2
    to_mjy = (proj_plane_pixel_area(frame['wcs']) * u.deg**2).to_value(u.sr) * 1e9   # MJy/sr * pixels -> mJy
    return float(flux * to_mjy), float(np.sqrt(flux_var) * to_mjy), n_bad


def centroid(image, variance, x, y, half=1):
    """Brightness-weighted center of the (2*half+1)^2 pixels around (x, y), with its uncertainty from the
    pixel noise. Returns (cx, cy, err_x, err_y) in pixels, or None near the edge / with no signal."""
    cx, cy = int(np.round(float(x))), int(np.round(float(y)))
    if not (half <= cx < image.shape[1] - half and half <= cy < image.shape[0] - half):
        return None
    box = image[cy - half:cy + half + 1, cx - half:cx + half + 1]
    box_var = np.nan_to_num(variance[cy - half:cy + half + 1, cx - half:cx + half + 1])
    weights = np.nan_to_num(box - np.nanmedian(image)).clip(min=0)
    total = weights.sum()
    if total <= 0:
        return None
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    ox, oy = (weights * dx).sum() / total, (weights * dy).sum() / total
    err_x = np.sqrt((box_var * (dx - ox) ** 2).sum()) / total
    err_y = np.sqrt((box_var * (dy - oy) ** 2).sum()) / total
    return cx + ox, cy + oy, err_x, err_y


def _box_mean(image, with_count=False):
    """3x3 mean of every pixel over its finite neighbours (NaN where none); optionally also that count."""
    padded = np.pad(image, 1, constant_values=np.nan)
    ny, nx = image.shape
    stack = np.stack([padded[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    count = np.isfinite(stack).sum(axis=0)
    mean = np.where(count > 0, np.nansum(stack, axis=0) / np.maximum(count, 1), np.nan)
    return (mean, count) if with_count else mean


def significant_peaks(frame, noise='pixel'):
    """Every real source peak in the frame: local maxima of the 3x3-smoothed, bad-pixel-free image that stand
    TRACK_MIN_SIGMA above the sky. Smoothing over ~one PSF means a single noisy pixel can't make a peak, and
    bad pixels (every artifact flag in BAD_FLAG_NAMES: ghosts, crosstalk, persistence, cosmic rays ...) are
    removed BEFORE looking, so a flagged artifact can never be picked as the object.

    noise='pixel': significance against the instrument noise (the pipeline's VARIANCE plane) - every real source,
    however faint. noise='field': against the scatter of the whole image, which in crowded fields is dominated by
    faint unresolved stars - only sources standing out from that crowd (for listing unknown sources).
    (Measured, crowded Cygnus field at 4.5-5 um: a SNR 10-24 star reached only ~3 'field' sigma, but ~10 'pixel'.)
    Returns (xs, ys, significance) arrays (pixel positions of the peak pixels)."""
    clean = np.where(frame['bad'], np.nan, frame['image'])
    smooth, count = _box_mean(clean, with_count=True)
    finite = smooth[np.isfinite(smooth)]
    if finite.size < 20:
        return np.array([]), np.array([]), np.array([])
    sky = np.median(finite)
    if noise == 'pixel' and 'variance' in frame:
        variance = np.where(np.isfinite(clean), frame['variance'], np.nan)
        noise = np.sqrt(np.clip(_box_mean(variance), 0, None) / np.maximum(count, 1))
        noise = np.where(np.isfinite(noise) & (noise > 0), noise, np.inf)
    else:
        noise = max(robust_scatter(finite), 1e-12)
    padded = np.pad(np.nan_to_num(smooth, nan=-np.inf), 1, constant_values=-np.inf)
    ny, nx = smooth.shape
    neighbours = np.stack([padded[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    significance = (smooth - sky) / noise
    peak = np.isfinite(smooth) & (smooth >= neighbours.max(axis=0)) & (significance >= cfg.TRACK_MIN_SIGMA)
    ys, xs = np.nonzero(peak)
    return xs.astype(float), ys.astype(float), significance[ys, xs]


def locate_object(frame):
    """Where the object actually is in this frame, and whether we're sure it's the right one.

    Of all significant source peaks (see significant_peaks), the NEAREST to where the object should be is taken -
    not the brightest one: a brighter star or asteroid passing within the search radius can't steal the track
    (that happened to Eros in one frame: a brighter neighbour 2.75 px away was picked, 16.9" off). Then the
    brightness-weighted centre around it gives the position, converted to RA/Dec with this frame's own pointing.

    frame['track'] says how far to trust it:
      'ok'            a peak within TRACK_TOLERANCE_PIX of the expected position: the object
      'off'           the nearest peak is further than that (within SEARCH_RADIUS_PIX): probably something else
      'undetected'    no significant peak near the expected position (position = expected, for display only)
      'no astrometry' SPHEREx's pointing for this image wasn't verified (header ASTRONETSOLVED = False)
    Only 'ok' frames are used for positions and movement (usable_frames)."""
    x, y = frame['catalog_xy']
    clean = np.where(frame['bad'], np.nan, frame['image'])         # bad pixels can't pull the position
    expected = {'object_ra': frame['expected_ra'], 'object_dec': frame['expected_dec'], 'object_err_arcsec': np.nan}
    if not (0 <= x < clean.shape[1] and 0 <= y < clean.shape[0]):
        return {'object_found': False, 'track': 'undetected'}
    xs, ys, significance = significant_peaks(frame)
    distance = np.hypot(xs - x, ys - y)
    near = distance <= cfg.SEARCH_RADIUS_PIX
    if not near.any():
        return {'object_found': True, 'track': 'undetected', 'track_sep_px': np.nan, **expected}
    best = np.flatnonzero(near)[np.argmin(distance[near])]
    # the smoothed peak picks WHICH source; its brightest original pixel (within 1 px) centres the centroid
    px, py = int(xs[best]), int(ys[best])
    box = clean[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
    if np.isfinite(box).any():
        by, bx = np.unravel_index(np.nanargmax(box), box.shape)
        px, py = max(px - 1, 0) + bx, max(py - 1, 0) + by
    c = centroid(clean, frame['variance'], px, py)
    if c is None:
        return {'object_found': False, 'track': 'undetected'}
    separation = float(np.hypot(c[0] - x, c[1] - y))
    sky = frame['wcs'].pixel_to_world(c[0], c[1])
    if not frame.get('astrometry_ok', True):
        track = 'no astrometry'
    else:
        track = 'ok' if separation <= cfg.TRACK_TOLERANCE_PIX else 'off'
    return {'object_found': True, 'track': track, 'track_sep_px': separation,
            'object_ra': float(sky.ra.deg), 'object_dec': float(sky.dec.deg),
            'object_err_arcsec': float(np.hypot(c[2], c[3]) * cfg.PIXEL_ARCSEC)}


def track_counts(frames):
    """How many frames have each track status, as text."""
    counts = {}
    for f in frames:
        counts[f.get('track', 'ok')] = counts.get(f.get('track', 'ok'), 0) + 1
    return ', '.join(f'{n} {status}' for status, n in sorted(counts.items(), key=lambda kv: -kv[1]))


# ----------------------------------------------------------------------------
# Alignment check (verifies, never adjusts)
# ----------------------------------------------------------------------------

def check_alignment(frames, field_stars):
    """In every frame, compare where moderately bright catalog stars actually appear with where SPHEREx's
    pointing puts them (median over many stars, so one star with real motion or a saturated halo can't fake
    a misalignment). Stores each frame's offset as frame['alignment_px']; nothing is moved."""
    ks = np.ma.filled(field_stars['k_m'], 99.0)
    # moderately bright: not saturated, but bright enough for a precise centre (wider range in sparse fields)
    chosen = (ks > 9) & (ks < 12)
    if chosen.sum() < 8:
        chosen = (ks > 8.5) & (ks < 13.5)
    stars = SkyCoord(field_stars['ra'], field_stars['dec'], unit='deg')[chosen]

    for frame in frames:
        xs, ys = frame['wcs'].world_to_pixel(stars)
        clean = np.where(frame['bad'], np.nan, frame['image'])
        ny, nx = clean.shape
        offsets = []
        for x, y in zip(xs, ys):
            if 3 <= x < nx - 3 and 3 <= y < ny - 3:
                c = centroid(clean, frame['variance'], x, y)
                if c:
                    offsets.append((c[0] - x, c[1] - y))
        frame['alignment_px'] = float(np.hypot(*np.median(offsets, axis=0))) if len(offsets) >= 3 else np.nan

    alignment = np.array([f['alignment_px'] for f in frames])
    if np.isnan(alignment).all():
        log.info(f'Alignment check: not possible here - too few catalog reference stars in the zoom ({len(stars)}; a sparse '
              f'field). The pointing is still IRSA\'s, fitted over the whole detector.')
        return alignment
    bad = np.flatnonzero(alignment > cfg.ALIGNMENT_TOLERANCE_PIX)
    log.info(f"Alignment check ({'OK' if not len(bad) else f'WARNING: {len(bad)} frames above tolerance'}): "
          f"{len(stars)} catalog stars appear where SPHEREx's pointing puts them, to within "
          f'{np.nanmedian(alignment):.2f} px typical, {np.nanmax(alignment):.2f} px worst '
          f'({np.nanmedian(alignment) * cfg.PIXEL_ARCSEC:.2f}" / {np.nanmax(alignment) * cfg.PIXEL_ARCSEC:.2f}")'
          + (f"; unchecked (too few stars): {int(np.isnan(alignment).sum())}" if np.isnan(alignment).any() else ''))
    return alignment


# ----------------------------------------------------------------------------
# Movement of the object relative to T1
# ----------------------------------------------------------------------------

def track_object(frames):
    """Offset of the object from its T1 position in every frame.

    Flagged as movement if the offset is bigger than MOVEMENT_SIGMA x the typical frame-to-frame scatter of
    the offsets AND 3x that frame's own measurement noise + alignment offset (so neither a noisy frame nor a
    poorly aligned one can fake it). Only frames where the object is detected (signal-to-noise >= DETECTION_SNR)
    and has no bad pixel in the circle are used: in the others its 'position' is just the brightest noise."""
    usable = usable_frames(frames)
    if not usable.any():
        log.info(f'Movement: the object is not detected (signal-to-noise >= {cfg.DETECTION_SNR}) in any single frame, '
              f'so its position cannot be measured frame by frame')
        usable = reliable_frames(frames)
    first = int(np.flatnonzero(usable)[0]) if usable.any() else 0
    t1 = SkyCoord(frames[first]['object_ra'], frames[first]['object_dec'], unit='deg')
    positions = SkyCoord([f['object_ra'] for f in frames], [f['object_dec'] for f in frames], unit='deg')
    east, north = t1.spherical_offsets_to(positions)              # east = RA offset on the sky (RA * cos Dec)
    d_ra, d_dec = east.to_value(u.arcsec), north.to_value(u.arcsec)
    offset = np.hypot(d_ra, d_dec)
    error = np.array([f['object_err_arcsec'] for f in frames])
    alignment = np.nan_to_num([f.get('alignment_px', 0.0) for f in frames]) * cfg.PIXEL_ARCSEC

    # Typical scatter per detector: each detector covers different wavelengths, where the star's image is
    # sharper or broader, so positions are measured more or less precisely (worst at ~5 um). Measured from
    # differences between frames taken within a day of each other, so a star's real (slow) motion over months
    # doesn't inflate it and hide itself.
    detector = np.array([f['detector'] for f in frames])
    mjd = np.array([f['mjd'] for f in frames])
    scatter = np.full(len(frames), np.nan)
    for d in np.unique(detector):
        same = np.flatnonzero((detector == d) & usable)
        close = np.diff(mjd[same]) < 1.0
        steps = np.r_[np.diff(d_ra[same])[close], np.diff(d_dec[same])[close]] / np.sqrt(2)
        scatter[detector == d] = robust_scatter(steps) if len(steps) >= 6 else robust_scatter(np.r_[d_ra[same], d_dec[same]])
    flagged = usable & (offset > cfg.MOVEMENT_SIGMA * scatter) & (offset > 3 * np.hypot(error, alignment))
    largest = int(np.argmax(np.where(usable, offset, -1)))

    per_detector = ', '.join(f'D{d} {scatter[detector == d][0]:.2f}"' for d in np.unique(detector))
    log.info(f'Movement vs T1: typical scatter per axis {per_detector}; largest offset {offset[largest]:.2f}" '
          f'({offset[largest] / cfg.PIXEL_ARCSEC:.2f} px) in T{largest + 1}. '
          + (f'FLAGGED ({int(flagged.sum())} frames): {", ".join(f"T{i + 1}" for i in np.flatnonzero(flagged)[:12])}'
             f'{" ..." if flagged.sum() > 12 else ""}' if flagged.any()
             else f'No significant movement (nothing above {cfg.MOVEMENT_SIGMA}x its detector\'s scatter)'))
    return {'reference': t1, 'reference_label': 'object at T1',     # the blue circle in the plots
            'd_ra_arcsec': d_ra, 'd_dec_arcsec': d_dec, 'offset_arcsec': offset,
            'error_arcsec': error, 'scatter_arcsec': scatter, 'flagged': flagged}


def track_moving(frames):
    """A MOVING object (an asteroid): how far it has travelled since it was first detected, and how well it
    was found where the ephemeris said it would be.

    A frame counts as detected above DETECTION_SNR. 'residual' = distance between where the object was found
    and where JPL Horizons predicted it; it is flagged ('OFF EPHEMERIS') above RESIDUAL_LIMIT_PX - usually a
    star or glitch pulled the position, not a real deviation. Frames are in time order."""
    snr = np.array([f['snr'] for f in frames])
    track = np.array([f.get('track', 'ok') for f in frames])
    bright = np.isfinite(snr) & (snr >= cfg.DETECTION_SNR)
    detected = bright & (track == 'ok')                 # bright at the prediction AND a source peak right there
    if not detected.any():
        raise RuntimeError(f'The asteroid is not detected (signal-to-noise {cfg.DETECTION_SNR}) in any of the '
                           f'{len(frames)} frames - it may be too faint for SPHEREx at this distance')
    expected = SkyCoord([f['expected_ra'] for f in frames], [f['expected_dec'] for f in frames], unit='deg')
    found = SkyCoord([f['object_ra'] for f in frames], [f['object_dec'] for f in frames], unit='deg')

    east, north = expected.spherical_offsets_to(found)
    residual = np.hypot(east.to_value(u.arcsec), north.to_value(u.arcsec))
    start = found[np.flatnonzero(detected)[0]]                      # first detection
    d_ra, d_dec = (a.to_value(u.arcsec) for a in start.spherical_offsets_to(found))
    offset = start.separation(found).arcsec
    # off ephemeris: light at the predicted spot, but the nearest source peak is elsewhere (or too far)
    flagged = (detected & (residual > cfg.RESIDUAL_LIMIT_PX * cfg.PIXEL_ARCSEC)) | (bright & (track == 'off'))

    days = np.array([f['mjd'] for f in frames]) - frames[0]['mjd']
    steps = found[detected][1:].separation(found[detected][:-1]).arcsec
    span = np.ptp(days[detected])
    log.info(f'Asteroid detected in {int(detected.sum())} of {len(frames)} frames (signal-to-noise >= {cfg.DETECTION_SNR}). '
          f'Found within {np.median(residual[detected]):.1f}" (median) / {np.max(residual[detected]):.1f}" (worst) of its '
          f'ephemeris position ({np.median(residual[detected]) / cfg.PIXEL_ARCSEC:.2f} / '
          f'{np.max(residual[detected]) / cfg.PIXEL_ARCSEC:.2f} px).')
    log.info(f'It travelled {steps.sum() / 3600:.1f} degrees across the sky in {span:.1f} days '
          f'(average {steps.sum() / 3600 / max(span, 1e-9):.2f} deg/day).'
          + (f' Off ephemeris: {", ".join(f"T{i + 1}" for i in np.flatnonzero(flagged)[:12])}'
             f'{"..." if flagged.sum() > 12 else ""}' if flagged.any() else ''))

    if (bright & ~detected).any():
        log.info(f'  Not counted as detections: {int((bright & ~detected).sum())} frames with light at the predicted spot but '
              f'no source peak within {cfg.TRACK_TOLERANCE_PIX} px of it (another source, or blended) - '
              f'tracking: {track_counts(frames)}')
    moving_flags = sum(f.get('moving_flags', 0) > 0 for f, d in zip(frames, detected) if d)
    if detected.any():
        log.info(f'  SPHEREx\'s own pipeline marked the asteroid\'s pixels as changing during the exposure '
              f'(OUTLIER/TRANSIENT flags - the sign of a moving source) in {moving_flags} of {int(detected.sum())} '
              f'detected frames')

    def note(i):
        if detected[i]:
            return f'SNR {snr[i]:.0f}   {residual[i]:.1f}" from ephemeris'
        if track[i] == 'off':
            return f'NOT TRACKED: nearest source {residual[i]:.1f}" from ephemeris (something else)'
        return f'NOT DETECTED (SNR {snr[i]:.1f})'
    notes = [note(i) for i in range(len(frames))]
    return {'reference': expected, 'reference_label': 'predicted position (JPL Horizons)',
            'flag_label': 'OFF EPHEMERIS', 'notes': notes, 'detected': detected,
            'd_ra_arcsec': d_ra, 'd_dec_arcsec': d_dec, 'offset_arcsec': offset, 'residual_arcsec': residual,
            'error_arcsec': np.array([f['object_err_arcsec'] for f in frames]), 'flagged': flagged}


# ----------------------------------------------------------------------------
# Brightness changes: compare each frame with the object's own spectrum
# ----------------------------------------------------------------------------

def brightness_changes(frames, glow=False):
    """SPHEREx sees the object at a different wavelength in almost every frame, and its brightness depends
    on wavelength. So each frame is compared with a reference: the median brightness of the
    SPECTRUM_NEIGHBORS frames closest in wavelength (from all visits, excluding itself).
    'deviation' = fractional difference from that reference; flagged if more than BRIGHTNESS_SIGMA x the
    typical deviation and x its own measurement error.

    glow=True: for SATURATED stars - use the brightness of the fitted outer glow (see fit_wings) of the frames
    where the core is saturated. Only relative changes mean anything then, which is exactly what this compares."""
    wavelength = np.array([f['wavelength_um'] for f in frames])
    if glow:
        flux = np.array([f.get('glow', np.nan) for f in frames], dtype=float)
        reliable = np.array([f.get('position_method') == 'wings' for f in frames], dtype=bool) & np.isfinite(flux) & (flux > 0)
        flux_err = cfg.GLOW_PRECISION * np.nan_to_num(flux)   # the method's own scatter (measured on Barnard's Star)
    else:
        reliable = reliable_frames(frames)
        flux = np.array([f['flux_mjy'] for f in frames])
        flux_err = np.array([f['flux_err_mjy'] for f in frames])
    label = 'Brightness of the saturated star\'s outer glow' if glow else 'Brightness'

    # reference spectrum from reliable frames only, so a glitched frame can't distort its neighbors' reference
    # Neighbors must be within SPECTRUM_WINDOW of the frame's wavelength: a reference built from far-away
    # wavelengths (where the star is naturally brighter or fainter) would fake a change.
    good = np.flatnonzero(reliable)
    reference = np.full_like(flux, np.nan)
    for i in range(len(frames)):
        distance = np.abs(wavelength[good] - wavelength[i])
        candidates = good[(distance <= cfg.SPECTRUM_WINDOW * wavelength[i]) & (good != i)]
        candidates = candidates[np.argsort(np.abs(wavelength[candidates] - wavelength[i]))[:cfg.SPECTRUM_NEIGHBORS]]
        if len(candidates) >= cfg.SPECTRUM_MIN_NEIGHBORS:
            reference[i] = np.median(flux[candidates])

    deviation = flux / reference - 1                       # NaN where there's no reference: can't be judged
    checkable = reliable & np.isfinite(deviation)
    scatter = robust_scatter(deviation[checkable])
    noise = np.hypot(flux_err / reference, scatter)
    flagged = checkable & (np.abs(deviation) > cfg.BRIGHTNESS_SIGMA * noise)
    if (reliable & ~checkable).any():
        log.info(f'  {int((reliable & ~checkable).sum())} frames have too few other frames at a similar wavelength '
              f'to judge their brightness')

    if not checkable.any():
        log.info(f'{label} vs the object\'s own spectrum: not measurable - no frame has a usable brightness '
              '(saturated, too faint, or bad pixels)')
    else:
        log.info(f'{label} vs the object\'s own spectrum ({int(checkable.sum())} frames): typical deviation '
              f'{scatter * 100:.1f}%. '
              + (f'FLAGGED ({flagged.sum()}): ' + ', '.join(f'T{i + 1} ({deviation[i] * 100:+.0f}%)'
                                                          for i in np.flatnonzero(flagged)[:15])
                 + (' ...' if flagged.sum() > 15 else '')
                 if flagged.any() else f'No significant brightness changes (nothing above {cfg.BRIGHTNESS_SIGMA}x)'))
        # per survey visit: a slowly varying star (e.g. a Mira, period ~8 months) changes between visits, not
        # between frames of one visit - so the visits' median deviations are the clearest summary
        mjd = np.array([f['mjd'] for f in frames])
        order = np.argsort(mjd[checkable])
        times, values = mjd[checkable][order], deviation[checkable][order]
        visit = survey_visits(times)
        parts = []
        for v in range(int(visit.max()) + 1):
            these = values[visit == v]
            error = 1.2533 * robust_scatter(these) / np.sqrt(len(these)) if len(these) >= 3 else np.nan
            parts.append(f'MJD {times[visit == v].min():.0f}: {np.median(these) * 100:+.0f}% +/- {error * 100:.0f}% ({len(these)})')
        precision = f' (method precision ~{cfg.GLOW_PRECISION * 100:.0f}% per frame, measured on a steady star)' if glow else ''
        log.info(f'  per visit (median vs its own spectrum): ' + ';  '.join(parts) + precision)
    return {'reference_mjy': reference, 'deviation': deviation, 'scatter': scatter, 'flagged': flagged,
            'used': checkable, 'source': 'glow' if glow else 'aperture'}


# ----------------------------------------------------------------------------
# Spectrum check: is the object in the circle the catalog star?
# ----------------------------------------------------------------------------

# wavelength ranges (um) of the catalog filters - SPHEREx is averaged over the same range for the comparison
CATALOG_BAND_RANGES = [(1.235, (1.11, 1.36)), (1.662, (1.50, 1.80)), (2.159, (1.99, 2.31)),     # 2MASS J, H, Ks
                       (3.368, (2.80, 3.90)), (4.618, (4.00, 5.10))]                             # WISE W1, W2


def spectrum_check(tracked, frames):
    """Compare the object's SPHEREx spectrum with its catalog brightness (2MASS / SIMBAD J, H, K + WISE W1, W2).

    For each catalog filter, SPHEREx's brightness is averaged (median) over that filter's wavelength range -
    the same light the filter collected - and divided by the catalog value. (Drawing a smooth curve between the
    filters instead fails for objects with deep absorption bands, like methane in T / Y brown dwarfs.) The
    measuring circle misses a little light and may catch some from neighbours, so the overall level can be off
    somewhat; the band-to-band SHAPE should match. Returns None if fewer than 3 filters can be compared."""
    reliable = reliable_frames(frames)
    points = np.array([(f['wavelength_um'], f['flux_mjy'], f['flux_err_mjy'])
                       for f, ok in zip(frames, reliable) if ok and np.isfinite(f['flux_mjy'])])
    catalog = sorted(tracked['catalog_sed'], key=lambda p: p[1])
    if len(points) == 0:
        log.info('Spectrum check: skipped - no frame with a measured brightness')
        return None
    points = points[np.argsort(points[:, 0])]

    bands = []
    for name, wavelength, flux, _ in catalog:
        centre, (lo, hi) = min(CATALOG_BAND_RANGES, key=lambda b: abs(b[0] - wavelength))
        inside = (points[:, 0] >= lo) & (points[:, 0] <= hi)
        if inside.sum() >= 3 and flux > 0:
            bands.append((name, wavelength, float(np.median(points[inside, 1])) / flux, int(inside.sum())))
    if len(bands) < 3:
        log.info(f'Spectrum check: skipped - SPHEREx overlaps only {len(bands)} catalog filter(s) with measured brightness')
        return None
    ratios = np.array([b[2] for b in bands])
    scale = float(np.median(ratios))
    scatter = float(np.sqrt(np.mean((ratios / scale - 1) ** 2)))   # band-to-band disagreement in SHAPE

    same_object = 0.4 <= scale <= 1.3 and scatter <= 0.25
    verdict = 'PASS - same object' if same_object else 'FAIL - may not be the catalog object'
    log.info(f'Spectrum check ({verdict}): SPHEREx / catalog = {scale:.2f} on average, shape agrees to {scatter * 100:.0f}% '
          f'across {len(bands)} filters: ' + ', '.join(f'{n} {r:.2f}' for n, _, r, _ in bands))
    return {'points': points, 'catalog': catalog, 'bands': bands, 'scale': scale, 'scatter': scatter,
            'same_object': same_object}


# ----------------------------------------------------------------------------
# A named star: SPHEREx's positions against the catalog's prediction (proper motion + parallax)
# ----------------------------------------------------------------------------

def compare_with_catalog(frames, movement, star):
    """How well SPHEREx's measured positions follow the catalog's prediction (SIMBAD / Gaia: proper motion,
    radial velocity and parallax), and SPHEREx's own measurement of the proper motion.

    The catalog's prediction for every frame is in 'expected_ra' / 'expected_dec'. The residuals (measured -
    predicted) are fitted with a straight line in time: its slope is how much SPHEREx's proper motion differs
    from the catalog's, so SPHEREx's measurement = catalog + slope. Adds the predicted track (relative to T1)
    to `movement` for the plots, and returns the comparison."""
    usable = usable_frames(frames)
    expected = SkyCoord([f['expected_ra'] for f in frames], [f['expected_dec'] for f in frames], unit='deg')
    found = SkyCoord([f['object_ra'] for f in frames], [f['object_dec'] for f in frames], unit='deg')
    first = int(np.flatnonzero(usable)[0]) if usable.any() else 0
    east, north = (a.to_value(u.arcsec) for a in expected[first].spherical_offsets_to(expected))
    movement.update(expected_d_ra_arcsec=east, expected_d_dec_arcsec=north, expected_offset_arcsec=np.hypot(east, north))

    res_east, res_north = (a.to_value(u.arcsec) for a in expected.spherical_offsets_to(found))
    residual = np.hypot(res_east, res_north)
    movement['residual_arcsec'] = residual
    years = (np.array([f['mjd'] for f in frames]) - frames[first]['mjd']) / 365.25
    result = {'usable': int(usable.sum()), 'median_residual_arcsec': float(np.median(residual[usable])) if usable.any() else np.nan}

    # Fit per VISIT, not per frame: frames within one visit share small systematic offsets (SPHEREx flips its
    # orientation between visits; measured on eps Indi B: ~0.1-0.2" visit to visit), so treating hundreds of frames
    # as independent would make the uncertainty far too small.
    t_all = years[usable]
    order = np.argsort(t_all)
    t_all = t_all[order]
    visit = survey_visits(t_all * 365.25)
    n_visits = int(visit.max()) + 1 if len(t_all) else 0
    span = np.ptp(t_all) if len(t_all) > 1 else 0.0
    if usable.sum() < 10 or n_visits < 2 or span < 0.25:
        log.info(f'Catalog comparison: too few detections over too short a time ({int(usable.sum())} frames, '
              f'{n_visits} visits, {span * 365.25:.0f} days) to measure the proper motion')
        return result

    t_visit = np.array([np.median(t_all[visit == v]) for v in range(n_visits)])
    for axis, values, catalog_pm in (('RA', res_east[usable][order], star['pmra'] or 0.0),
                                     ('Dec', res_north[usable][order], star['pmdec'] or 0.0)):
        medians, errors = [], []
        for v in range(n_visits):
            in_visit = values[visit == v]
            medians.append(np.median(in_visit))
            spread = robust_scatter(in_visit) if len(in_visit) >= 3 else 0.5 * cfg.PIXEL_ARCSEC
            errors.append(max(1.2533 * spread / np.sqrt(len(in_visit)), 0.01))
        medians, weights = np.array(medians), 1 / np.array(errors) ** 2
        t_mean = np.sum(weights * t_visit) / np.sum(weights)
        slope = np.sum(weights * (t_visit - t_mean) * medians) / np.sum(weights * (t_visit - t_mean) ** 2)
        intercept = np.sum(weights * (medians - slope * t_visit)) / np.sum(weights)
        error = 1 / np.sqrt(np.sum(weights * (t_visit - t_mean) ** 2))
        dof = n_visits - 2
        if dof > 0:                                       # visits scatter more than their errors: inflate
            chi2 = np.sum(weights * (medians - (slope * t_visit + intercept)) ** 2)
            error *= max(1.0, np.sqrt(chi2 / dof))
        result[f'pm_{axis}'] = (catalog_pm + slope * 1000, error * 1000, catalog_pm)    # mas/yr: SPHEREx, error, catalog
    total_catalog = np.hypot(star['pmra'] or 0, star['pmdec'] or 0)
    total_spherex = np.hypot(result['pm_RA'][0], result['pm_Dec'][0])
    sigma = [abs(result[k][0] - result[k][2]) / result[k][1] for k in ('pm_RA', 'pm_Dec')]
    result['agreement_sigma'] = sigma
    log.info(f"Catalog comparison ({int(usable.sum())} frames, {n_visits} visits over {span * 365.25:.0f} days): SPHEREx "
          f"positions are a median {result['median_residual_arcsec']:.2f}\" "
          f"({result['median_residual_arcsec'] / cfg.PIXEL_ARCSEC:.2f} px) from the catalog prediction.")
    log.info(f"  Proper motion measured by SPHEREx: RA {result['pm_RA'][0]:8.0f} +/- {result['pm_RA'][1]:.0f} mas/yr   "
          f"Dec {result['pm_Dec'][0]:8.0f} +/- {result['pm_Dec'][1]:.0f} mas/yr   (total {total_spherex / 1000:.2f}\"/yr)")
    log.info(f"  Catalog (SIMBAD/Gaia):            RA {result['pm_RA'][2]:8.0f}             Dec {result['pm_Dec'][2]:8.0f}"
          f"            (total {total_catalog / 1000:.2f}\"/yr)   -> differ by {sigma[0]:.1f} / {sigma[1]:.1f} sigma"
          + ('' if dof > 0 else '  (only 2 visits: uncertainty may be underestimated)'))
    return result


# ----------------------------------------------------------------------------
# Detection: per frame, and combining frames for targets too faint for any single one
# ----------------------------------------------------------------------------

DETECTION_BANDS_UM = [(0.75, 1.1), (1.1, 1.6), (1.6, 2.4), (2.4, 3.8), (3.8, 4.4), (4.4, 5.0)]
CONTROL_RADIUS_PIX = 10           # empty-sky comparison positions on a ring this far from the target
CONTROL_COUNT = 16


def detection_summary(frames):
    """How strongly the object is detected, per wavelength band.

    Per frame: signal-to-noise at its expected position. Combined: the weighted mean brightness of all reliable
    frames in the band - a source too faint for any single frame can still show up. The combined significance is
    judged against CONTROL_COUNT 'control' positions on a ring around the target, measured exactly the same way
    in every frame: their spread is the real noise of a combined measurement (including anything my error bars
    miss), so a combined detection only counts if the target stands out from the controls.
    Must run while the frames still hold their full pixels (before the zooms release them)."""
    reliable = reliable_frames(frames)
    wavelength = np.array([f['wavelength_um'] for f in frames])
    snr = np.array([f['snr'] for f in frames], dtype=float)
    flux = np.array([f['flux_mjy'] for f in frames], dtype=float)
    error = np.array([f['flux_err_mjy'] for f in frames], dtype=float)

    angles = np.linspace(0, 2 * np.pi, CONTROL_COUNT, endpoint=False)
    controls = np.full((len(frames), CONTROL_COUNT), np.nan)
    for i, frame in enumerate(frames):
        if 'image' not in frame:
            raise RuntimeError('detection_summary needs the full pixels: run it before the zooms release them')
        x, y = frame['catalog_xy']
        for k, angle in enumerate(angles):
            controls[i, k] = aperture_flux(frame, x + CONTROL_RADIUS_PIX * np.cos(angle),
                                           y + CONTROL_RADIUS_PIX * np.sin(angle))[0]

    def combine(values, use):
        weights = 1 / error[use] ** 2
        return np.sum(values[use] * weights) / np.sum(weights)

    log.info(f'Detection: in {int(np.sum(snr >= cfg.DETECTION_SNR))} of {len(frames)} single frames '
          f'(signal-to-noise >= {cfg.DETECTION_SNR}). Combining frames per band, against {CONTROL_COUNT} empty-sky controls:')
    rows = []
    for lo, hi in DETECTION_BANDS_UM:
        use = reliable & (wavelength >= lo) & (wavelength < hi) & np.isfinite(flux) & (error > 0)
        if use.sum() < 3:
            continue
        target = combine(flux, use)
        control_values = np.array([combine(controls[:, k], use & np.isfinite(controls[:, k])) for k in range(CONTROL_COUNT)
                                   if np.sum(use & np.isfinite(controls[:, k])) >= 3])
        noise = robust_scatter(control_values)
        significance = (target - np.median(control_values)) / noise if noise > 0 else np.nan
        formal = target * np.sqrt(np.sum(1 / error[use] ** 2))
        rows.append({'band_um': (lo, hi), 'frames': int(use.sum()), 'single_detections': int(np.sum(snr[use] >= cfg.DETECTION_SNR)),
                     'flux_mjy': float(target), 'noise_mjy': float(noise), 'significance': float(significance),
                     'formal_snr': float(formal)})
        log.info(f'  {lo:.2f}-{hi:.2f} um: {int(use.sum()):3d} frames ({rows[-1]["single_detections"]:3d} detected singly)  '
              f'combined {target:9.3f} mJy, control noise {noise:.3f} mJy -> {significance:6.1f} sigma'
              + ('  DETECTED' if significance >= cfg.DETECTION_SNR else ''))
    if not rows:
        saturated = sum(bool(f.get('saturated')) for f in frames)
        log.info(f'  (no frame has a measurable brightness'
              + (f': {saturated} of {len(frames)} are saturated - too bright for SPHEREx)' if saturated else ')'))
    return rows
