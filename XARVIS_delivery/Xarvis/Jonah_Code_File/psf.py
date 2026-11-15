"""SPHEREx's own PSF: position fits and aperture corrections.

Every SPHEREx image comes with an EPSF table: the "effective PSF" (the star image, already integrated over a
pixel) measured from hundreds of stars in each of ~450 zones across the detector, 5x oversampled, 33 x 33 samples
(6.6 pixels across). The PSF changes with wavelength and across the detector, so the zone nearest the cutout is
kept (irsa.open_cutout stores it as extension 'PSF', ~10 KB) and used here for:

  psf_fit        a source's position from fitting the PSF (instead of a 3x3 brightness-weighted centroid, which is
                 pulled by noise and by the pixel grid)
  encircled      the fraction of a point source's light inside the measuring circle: the circle catches less of a
                 blurrier (longer-wavelength) image, so fluxes are divided by it to compare wavelengths fairly

Checked on real stars: the stored EPSF orientation matches the images (no flip or transpose fits better).
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import config as cfg


def psf_hdu_for_cutout(full):
    """The EPSF zone nearest the centre of a just-downloaded cutout (HDUList with 'IMAGE' and 'EPSF'), as an
    ImageHDU named 'PSF' (or None if the image has no EPSF table)."""
    if 'EPSF' not in full:
        return None
    table, image = full['EPSF'], full['IMAGE']
    ny, nx = image.data.shape
    det_x, det_y = detector_position(image.header, (nx - 1) / 2, (ny - 1) / 2)
    data = table.data
    distance = np.hypot(data['XCENTER'] - det_x, data['YCENTER'] - det_y)
    k = int(np.argmin(distance))
    hdu = fits.ImageHDU(np.asarray(data['EPSF'][k], dtype=np.float32), name='PSF')
    hdu.header['OVERSAMP'] = (int(table.header.get('OVSMPX', 5)), 'samples per pixel')
    hdu.header['ZONE_X'] = (float(data['XCENTER'][k]), 'zone centre, detector pixel')
    hdu.header['ZONE_Y'] = (float(data['YCENTER'][k]), 'zone centre, detector pixel')
    hdu.header['ZONEDIST'] = (float(distance[k]), 'cutout centre to zone centre, pixels')
    hdu.header['NSTAR'] = (int(data['NSTAR'][k]), 'stars used for this PSF')
    hdu.header['CWAVE'] = (float(data['CWAVE'][k]), 'zone central wavelength (um)')
    return hdu


def detector_position(header, x, y):
    """Detector pixel (0-based) of cutout pixel (x, y), from alternate WCS 'A' - a plain offset in SPHEREx
    headers (CTYPE 'LINEAR'), computed directly: building an astropy WCS for it cost ~19 ms per frame."""
    if header.get('CTYPE1A') != 'LINEAR' or header.get('CTYPE2A') != 'LINEAR':
        return WCS(header, key='A').pixel_to_world_values(x, y)
    det_x = (np.asarray(x, dtype=float) + 1 - header['CRPIX1A']) * header.get('CDELT1A', 1.0) + header.get('CRVAL1A', 0.0)
    det_y = (np.asarray(y, dtype=float) + 1 - header['CRPIX2A']) * header.get('CDELT2A', 1.0) + header.get('CRVAL2A', 0.0)
    return det_x, det_y


def native_samples(psf, oversample, dx, dy):
    """The effective PSF at native-pixel offsets (dx, dy) from the source centre, as the fraction of the source's
    light in each pixel (sums to ~1 over the whole PSF)."""
    from scipy.ndimage import map_coordinates
    centre = (psf.shape[0] - 1) / 2
    u = centre + np.asarray(dx, dtype=float) * oversample
    v = centre + np.asarray(dy, dtype=float) * oversample
    return map_coordinates(psf, [v, u], order=3, mode='constant', cval=0.0) * oversample ** 2


def encircled(frame, x, y, radius=None):
    """Fraction of a point source's light at (x, y) that falls in the measuring circle (pixel centres within
    `radius` px, exactly as measure.aperture_flux counts them), from the frame's PSF. None without a PSF."""
    if frame.get('psf') is None:
        return None
    radius = cfg.APERTURE_RADIUS_PIX if radius is None else radius
    r = int(np.ceil(radius)) + 1
    cx, cy = int(np.round(x)), int(np.round(y))
    yy, xx = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
    inside = np.hypot(xx - x, yy - y) <= radius
    values = native_samples(frame['psf'], frame['psf_oversample'], xx - x, yy - y)
    total = native_samples(frame['psf'], frame['psf_oversample'],
                           *np.meshgrid(np.arange(-4, 5) + (x - cx), np.arange(-4, 5) + (y - cy))).sum()
    fraction = float(values[inside].sum() / total) if total > 0 else None
    return fraction if fraction and 0.2 < fraction <= 1.0 else None


def psf_fit(frame, x0, y0, half=3, max_shift=1.0):
    """Position of a point source near (x0, y0) by fitting the frame's PSF (position, brightness and a flat
    background) to the (2*half+1)^2 good pixels around it, weighted by the pixel variances.
    Returns (x, y, err_x, err_y, reduced_chi2) or None (no PSF, too few good pixels, or the fit wandered more
    than `max_shift` px from the start - then the centroid is kept)."""
    if frame.get('psf') is None:
        return None
    from scipy.optimize import least_squares
    image, variance = frame['image'], frame['variance']
    cx, cy = int(np.round(x0)), int(np.round(y0))
    ny, nx = image.shape
    if not (half <= cx < nx - half and half <= cy < ny - half):
        return None
    yy, xx = np.mgrid[cy - half:cy + half + 1, cx - half:cx + half + 1]
    good = ~frame['bad'][yy, xx] & np.isfinite(image[yy, xx]) & np.isfinite(variance[yy, xx]) & (variance[yy, xx] > 0)
    if good.sum() < 12:
        return None
    xs, ys = xx[good].astype(float), yy[good].astype(float)
    data, weight = image[yy, xx][good], 1 / np.sqrt(variance[yy, xx][good])
    psf, oversample = frame['psf'], frame['psf_oversample']

    def linear_fit(px, py):
        model = native_samples(psf, oversample, xs - px, ys - py)
        design = np.c_[model, np.ones_like(model)] * weight[:, None]
        coef, *_ = np.linalg.lstsq(design, data * weight, rcond=None)
        return coef, (data - np.c_[model, np.ones_like(model)] @ coef) * weight

    fit = least_squares(lambda p: linear_fit(*p)[1], [float(x0), float(y0)], diff_step=1e-3,
                        bounds=([x0 - max_shift - 0.5, y0 - max_shift - 0.5], [x0 + max_shift + 0.5, y0 + max_shift + 0.5]))
    coef, residual = linear_fit(*fit.x)
    if not fit.success or coef[0] <= 0 or np.hypot(fit.x[0] - x0, fit.x[1] - y0) > max_shift:
        return None
    dof = max(len(data) - 4, 1)
    chi2 = float(np.sum(residual ** 2) / dof)
    try:                                        # uncertainty from the fit's curvature, scaled by its quality
        covariance = np.linalg.inv(fit.jac.T @ fit.jac) * max(chi2, 1.0)
        err_x, err_y = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:
        return None
    return float(fit.x[0]), float(fit.x[1]), float(err_x), float(err_y), chi2
