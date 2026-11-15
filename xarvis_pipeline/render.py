"""Images for the website: each frame as a PNG, stacks per detector array, and differences between survey visits.

Everything is computed from the zooms the pipeline already keeps (original SPHEREx pixels, bad pixels marked), so no
extra downloads are made. PNGs are greyscale at the native resolution (one PNG pixel = one SPHEREx pixel, 6.15");
the browser scales them up without smoothing so the real pixels stay visible.
"""

import numpy as np
from PIL import Image

from . import config as cfg
from .measure import flag_bits, robust_scatter
from .spectrumanalyser import sample_frame, weighted_image


def _to_png(image, path, lo=None, hi=None, symmetric=False):
    """Save a 2-D array as an 8-bit PNG with an asinh stretch. NaN (no data / bad pixel) is drawn dark blue-grey.
    symmetric=True (differences): zero is mid-grey, positive white, negative black."""
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    out = np.zeros(image.shape + (3,), dtype=np.uint8)
    out[~finite] = (24, 30, 46)
    if finite.any():
        values = image[finite]
        if symmetric:
            scale = hi if hi is not None else max(np.nanpercentile(np.abs(values), 99.5), 1e-12)
            norm = 0.5 + 0.5 * np.arcsinh(image / (scale / 6)) / np.arcsinh(6.0)
        else:                                       # sky -> black, a soft asinh knee at ~3 sigma, brightest -> white
            lo = np.nanmedian(values) if lo is None else lo
            hi = np.nanpercentile(values, 99.8) if hi is None else hi
            soft = 3 * max(robust_scatter(values), 1e-12)
            norm = np.arcsinh(np.clip(image - lo, 0, None) / soft) / max(np.arcsinh(max(hi - lo, soft) / soft), 1e-6)
        grey = (np.clip(np.nan_to_num(norm), 0, 1) * 255).astype(np.uint8)
        out[finite] = np.repeat(grey[finite][:, None], 3, axis=1)
    Image.fromarray(out[::-1]).save(path)      # FITS rows go upwards; PNG rows go downwards


def flag_summary(zoom):
    """Names of the SPHEREx flags set inside the measurement circle, and how many pixels of the zoom are bad."""
    header, flags = zoom.get('flags_header'), zoom.get('flags')
    names = []
    if header is not None and flags is not None:
        x, y = zoom['catalog_xy']
        yy, xx = np.indices(flags.shape)
        near = np.hypot(xx - x, yy - y) <= cfg.APERTURE_RADIUS_PIX
        present = int(np.bitwise_or.reduce(flags[near].ravel())) if near.any() else 0
        for key in header['MSKN*']:
            if present & (1 << int(key[4:])):
                names.append(str(header[key]))
    bad = zoom.get('bad')
    return sorted(set(names)), int(np.sum(bad)) if bad is not None else 0


def frame_pngs(zooms, folder, size_pix=None):
    """frame_<n>.png for every zoom (n from 1, in time order), each resampled onto the same target-centred grid,
    north up and east left (SPHEREx images come in many rotations; on one grid they line up for blinking). One PNG
    pixel = one SPHEREx pixel; bad pixels (SPHEREx flags) are left empty. Returns per-frame display info."""
    size_pix = size_pix or cfg.ZOOM_SIZE_PIX
    east, north = _grid(size_pix)
    info = []
    for n, zoom in enumerate(zooms, start=1):
        image, _ = sample_frame(zoom, east, north)
        _to_png(image[::-1], folder / f'frame_{n}.png')
        names, n_bad = flag_summary(zoom)
        info.append({'png': f'frame_{n}.png', 'width': size_pix, 'height': size_pix,
                     'target_px': [(size_pix - 1) / 2, (size_pix - 1) / 2],   # the grid is centred on the target
                     'flags_in_aperture': names, 'bad_pixels': n_bad})
    return info


def _grid(size_pix):
    """A target-centred grid of (east, north) offsets in arcsec, north up and east LEFT (like a sky image)."""
    half = (size_pix - 1) / 2
    xs = (np.arange(size_pix) - half) * cfg.PIXEL_ARCSEC
    north, east = np.meshgrid(xs[::-1], -xs, indexing='ij')                # row 0 = north edge, column 0 = east edge
    return east, north


def _visits(zooms, gap_days=30.0):
    """Survey visit number of every zoom (a new visit after a gap of more than `gap_days`)."""
    mjd = np.array([z['mjd'] for z in zooms])
    order = np.argsort(mjd)
    visit = np.zeros(len(zooms), dtype=int)
    current = 0
    for prev, cur in zip(order[:-1], order[1:]):
        if mjd[cur] - mjd[prev] > gap_days:
            current += 1
        visit[cur] = current
    return visit


def stacks_and_differences(zooms, folder, size_pix=None):
    """Per detector array: the inverse-variance stack of all its frames (better signal-to-noise), and the last survey
    visit minus the first (things that moved or changed stand out). Frames are resampled onto one grid centred on the
    target (for a moving target that is a co-moving stack). Returns the list of PNGs made."""
    size_pix = size_pix or cfg.ZOOM_SIZE_PIX
    east, north = _grid(size_pix)
    samples = [sample_frame(z, east, north) for z in zooms]
    detectors = np.array([z['detector'] for z in zooms])
    visits = _visits(zooms)
    made = []
    for d in sorted(set(detectors.tolist())):
        idx = np.flatnonzero(detectors == d)
        images = np.array([samples[i][0] for i in idx])
        variances = np.array([samples[i][1] for i in idx])
        mean, _, coverage = weighted_image(images, variances)
        if np.isfinite(mean).any():
            _to_png(mean[::-1], folder / f'stack_d{d}.png')       # grid rows already run north->south
            wl = [zooms[i]['wavelength_um'] for i in idx]
            made.append({'kind': 'stack', 'detector': int(d), 'png': f'stack_d{d}.png', 'frames': int(len(idx)),
                         'wavelength_um': [float(min(wl)), float(max(wl))]})
        first, last = visits[idx].min(), visits[idx].max()
        if last > first:
            a = weighted_image(images[visits[idx] == first], variances[visits[idx] == first])[0]
            b = weighted_image(images[visits[idx] == last], variances[visits[idx] == last])[0]
            diff = b - a
            if np.isfinite(diff).any():
                _to_png(diff[::-1], folder / f'diff_d{d}.png', symmetric=True)
                made.append({'kind': 'difference', 'detector': int(d), 'png': f'diff_d{d}.png',
                             'from_mjd': float(min(zooms[i]['mjd'] for i in idx if visits[i] == first)),
                             'to_mjd': float(min(zooms[i]['mjd'] for i in idx if visits[i] == last))})
    return made
