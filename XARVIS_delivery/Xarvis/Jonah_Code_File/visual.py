"""Display-only image enhancement, used by the plots AFTER the analysis: colour, smoothing and a higher
(interpolated) resolution. Nothing here changes any measurement - it only ever works on copies, and its output
only ever goes to imshow.

Colour comes from each source's own spectrum: SPHEREx splits the light with a dichroic, so a short-wavelength
detector (D1-D3) and a long-wavelength detector (D4-D6) image the same sky at the same instant. Comparing how
bright each pixel is at those two wavelengths gives it a colour: sources brighter at the long wavelength than a
typical star in the field (cool stars, brown dwarfs) come out orange/red; brighter at the short wavelength
(hotter) come out blue; typical stars a warm white. Brightness sets how bright the colour is, so empty sky stays
black.
"""

import numpy as np

from . import config as cfg
from .measure import robust_scatter

# colour for a source's colour index (log10 long/short brightness ratio, relative to the field's typical star)
PALETTE_STOPS = np.array([-0.6, -0.25, 0.0, 0.25, 0.6])
PALETTE_RGB = np.array([[0.50, 0.66, 1.00],      # much bluer than a typical star
                        [0.74, 0.84, 1.00],
                        [1.00, 0.93, 0.82],      # typical star: warm white ("a bit of colour")
                        [1.00, 0.70, 0.42],
                        [1.00, 0.42, 0.28]])     # much redder (cool star / brown dwarf)


def find_partners(zooms):
    """For each frame, the index of the simultaneous frame on the other detector arm (same exposure, same sky,
    different wavelength), or None."""
    mjd = np.array([z['mjd'] for z in zooms])
    short_arm = np.array([z['detector'] <= 3 for z in zooms])
    partners = []
    for i in range(len(zooms)):
        candidates = np.flatnonzero((np.abs(mjd - mjd[i]) * 86400 < 5) & (short_arm != short_arm[i]))
        partners.append(int(candidates[0]) if len(candidates) else None)
    return partners


def _spline_matrix(n_in, factor):
    """(n_in * factor) x n_in matrix that upsamples a row by `factor` with a cubic B-spline: smooth, no ringing."""
    n_out = n_in * factor
    position = (np.arange(n_out) + 0.5) / factor - 0.5               # output pixel centres, in input pixels
    base = np.floor(position).astype(int)
    t = position - base
    weights = np.stack([(1 - t) ** 3, 3 * t ** 3 - 6 * t ** 2 + 4, -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3]) / 6
    matrix = np.zeros((n_out, n_in))
    for k in range(4):
        np.add.at(matrix, (np.arange(n_out), np.clip(base + k - 1, 0, n_in - 1)), weights[k])
    return matrix


def _lanczos_matrix(n_in, factor, a=3):
    """(n_in * factor) x n_in matrix that upsamples a row by `factor` with a Lanczos-3 kernel: much crisper than
    the B-spline (keeps edges sharp), at the cost of slight ringing next to very bright stars (clipped later)."""
    n_out = n_in * factor
    position = (np.arange(n_out) + 0.5) / factor - 0.5
    base = np.floor(position).astype(int)
    matrix = np.zeros((n_out, n_in))
    for k in range(-a + 1, a + 1):
        index = base + k
        t = position - index
        np.add.at(matrix, (np.arange(n_out), np.clip(index, 0, n_in - 1)), np.sinc(t) * np.sinc(t / a))
    return matrix / matrix.sum(axis=1, keepdims=True)


def upsample(image, factor, sharp=False):
    """Upsampled copy of a 2-D array (NaNs must be filled first): smooth (B-spline) or sharp (Lanczos-3)."""
    make = _lanczos_matrix if sharp else _spline_matrix
    return make(image.shape[0], factor) @ image @ make(image.shape[1], factor).T


def _resample_onto(source, target):
    """`source`'s pixels at `target`'s pixel centres (matched on the sky through each frame's coordinates),
    bilinear; NaN where `source` doesn't cover. Display only."""
    ny, nx = target['image'].shape
    yy, xx = np.indices((ny, nx))
    ra, dec = target['wcs'].pixel_to_world_values(xx, yy)
    x, y = source['wcs'].world_to_pixel_values(ra, dec)
    image = source['image']
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < image.shape[1]) & (y0 + 1 < image.shape[0])
    x0, y0 = np.clip(x0, 0, image.shape[1] - 2), np.clip(y0, 0, image.shape[0] - 2)
    value = (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
             + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)
    return np.where(inside, value, np.nan)


def _signal(image):
    """Background-subtracted copy, its noise, with NaNs set to the background."""
    finite = image[np.isfinite(image)]
    sky, noise = np.median(finite), robust_scatter(finite)
    signal = np.where(np.isfinite(image), image - sky, 0.0)
    return signal, max(noise, 1e-12)


def colour_of(index):
    """RGB (..., 3) for colour indices, from the palette."""
    index = np.clip(index, PALETTE_STOPS[0], PALETTE_STOPS[-1])
    return np.stack([np.interp(index, PALETTE_STOPS, PALETTE_RGB[:, c]) for c in range(3)], axis=-1)


def colour_index(zoom, partner):
    """Per-pixel colour index (log10 long/short-wavelength brightness, relative to the field's typical star)
    and its weight (0 where nothing is detected), from the comparison with `partner`. Zeros without a partner."""
    signal, noise = _signal(np.array(zoom['image'], dtype=float))
    index, weight = np.zeros_like(signal), np.zeros_like(signal)
    if partner is None:
        return index, weight
    other, other_noise = _signal(_resample_onto(partner, zoom))
    long_is_partner = partner['detector'] > zoom['detector']
    long, short = (other, signal) if long_is_partner else (signal, other)
    long_noise, short_noise = (other_noise, noise) if long_is_partner else (noise, other_noise)
    detected = (signal > 3 * noise) | (other > 3 * other_noise)
    ratio = np.log10(np.clip(long, long_noise, None) / np.clip(short, short_noise, None))
    if detected.sum() >= 5:
        weight = np.where(detected, np.clip(signal / noise, 0, 50) + np.clip(other / other_noise, 0, 50), 0.0)
        typical = np.median(ratio[detected & (weight > np.median(weight[detected]))])
        index = np.where(detected, ratio - typical, 0.0)
    return index, weight


def _near(shape, xy, radius):
    yy, xx = np.indices(shape)
    return np.hypot(xx - xy[0], yy - xy[1]) <= radius


def target_colour(zoom, partner, xy):
    """The target's colour index in one frame (brightness-weighted within ~1.5 px of `xy`), or NaN."""
    index, weight = colour_index(zoom, partner)
    near = _near(index.shape, xy, 1.5) & (weight > 0)
    return float(np.sum(index[near] * weight[near]) / np.sum(weight[near])) if near.any() else np.nan


def render(zoom, partner=None, factor=None, anchor=None):
    """Pretty RGB version of a zoom: colour from the two-wavelength comparison with `partner` (if any),
    sharpened and upsampled `factor` times. Returns (rgb, extent) - extent in the zoom's ORIGINAL pixel
    coordinates, so circles and RA/Dec axes still line up. Works on copies only.

    anchor = {'xy': target pixel in this zoom, 'colour': colour index to give it}: normalises every frame to
    the target - its brightness is drawn at the same level (TARGET_DISPLAY_LEVEL) and its colour is the same
    in every frame (everything else in the frame is shifted by the same amount, so colours stay relative)."""
    factor = factor or cfg.DISPLAY_UPSCALE
    signal, noise = _signal(np.array(zoom['image'], dtype=float))

    # brightness: asinh stretch from the noise level up; empty sky (below ~1.5 noise) is black
    stretch = lambda s: np.arcsinh(np.clip(s, 0, None) / noise)
    floor = stretch(1.5 * noise)
    peak = max(np.percentile(signal, 99.7), 20 * noise)
    target_peak = np.max(signal[_near(signal.shape, anchor['xy'], 1.5)]) if anchor else 0.0
    if anchor and target_peak > 5 * noise:                  # scale so the target lands at the same brightness
        top = floor + (stretch(target_peak) - floor) / cfg.TARGET_DISPLAY_LEVEL
        luminance = np.clip((stretch(signal) - floor) / (top - floor), 0, 1)
    else:
        luminance = np.clip((stretch(signal) - floor) / (stretch(peak) - floor), 0, 1)

    index, weight = colour_index(zoom, partner)
    if anchor and weight.any() and np.isfinite(anchor.get('colour', np.nan)):
        near = _near(index.shape, anchor['xy'], 1.5) & (weight > 0)
        if near.any():                                      # shift so the target gets its fixed colour
            here = np.sum(index[near] * weight[near]) / np.sum(weight[near])
            index = np.where(weight > 0, index + (anchor['colour'] - here), 0.0)

    # upsample: brightness sharp (Lanczos + a little unsharp masking); colour as a brightness-weighted smooth
    # average, so noise doesn't speckle it
    sharp = upsample(luminance, factor, sharp=True)
    smooth = upsample(luminance, factor)
    big_luminance = np.clip(sharp + cfg.DISPLAY_SHARPEN * (sharp - smooth), 0, 1)
    if weight.any():
        big_index = upsample(index * weight, factor) / np.maximum(upsample(weight, factor), 1e-9)
    else:
        big_index = np.zeros_like(big_luminance)
    rgb = np.clip(big_luminance[..., None] * colour_of(big_index), 0, 1)
    ny, nx = signal.shape
    return rgb, (-0.5, nx - 0.5, -0.5, ny - 0.5)


def anchor_colour(zooms, partners, max_frames=40):
    """The target's typical colour across frames (median of up to `max_frames` evenly spread frames that have a
    partner) - used to give it the same colour in every frame. NaN if it can't be measured."""
    candidates = [i for i, p in enumerate(partners) if p is not None]
    if not candidates:
        return np.nan
    picks = np.unique(np.linspace(0, len(candidates) - 1, min(max_frames, len(candidates))).astype(int))
    values = []
    for k in picks:
        i = candidates[k]
        if zooms[i].get('object_found'):                    # only where the target was actually seen
            values.append(target_colour(zooms[i], zooms[partners[i]], target_xy(zooms[i])))
    if not values:
        return np.nan
    values = np.array(values)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan


def target_xy(zoom):
    """The target's pixel position in a zoom: where it was found, else where it was expected."""
    ra, dec = zoom.get('object_ra', np.nan), zoom.get('object_dec', np.nan)
    if not (zoom.get('object_found') and np.isfinite(ra) and np.isfinite(dec)):
        ra, dec = zoom['expected_ra'], zoom['expected_dec']
    x, y = zoom['wcs'].world_to_pixel_values(ra, dec)
    return float(x), float(y)


def anchors(zooms, partners):
    """Per-frame anchor for render(): the target's position in each frame plus ONE colour for all frames."""
    colour = anchor_colour(zooms, partners)
    return [{'xy': target_xy(zoom), 'colour': colour} for zoom in zooms]
