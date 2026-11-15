"""Step 0: pick the star to track, and its catalog brightness for the spectrum check."""

import logging

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord

from . import config as cfg
from .irsa import query_catalog

log = logging.getLogger(__name__)

# catalog bands: (magnitude column, error column, wavelength um, zero-magnitude flux in Jy)
TWOMASS_BANDS = {'2MASS J': ('j_m', 'j_msigcom', 1.235, 1594.0),
                 '2MASS H': ('h_m', 'h_msigcom', 1.662, 1024.0),
                 '2MASS Ks': ('k_m', 'k_msigcom', 2.159, 666.7)}
ALLWISE_BANDS = {'WISE W1': ('w1mpro', 'w1sigmpro', 3.368, 309.54),
                 'WISE W2': ('w2mpro', 'w2sigmpro', 4.618, 171.787)}


def catalog_fluxes(row, bands):
    """Catalog magnitudes -> list of (band, wavelength_um, flux_mjy, flux_err_mjy)."""
    points = []
    for band, (mag_col, err_col, wavelength, zero_point_jy) in bands.items():
        mag, err = row[mag_col], row[err_col]
        if np.ma.is_masked(mag) or not np.isfinite(mag):
            continue
        flux = zero_point_jy * 10 ** (-0.4 * float(mag)) * 1e3                     # mJy
        flux_err = flux * 0.4 * np.log(10) * (float(err) if not np.ma.is_masked(err) else 0.1)
        points.append((band, wavelength, flux, flux_err))
    return points


def find_tracked_object(coord):
    """Nearest bright, ISOLATED 2MASS star to `coord`.

    Isolated: SPHEREx pixels are 6.15", so stars closer than a few pixels blend together in the
    measurement circle. Only stars whose neighbors within ISOLATION_RADIUS add up to less than
    ISOLATION_MAX_FRACTION of the star's own (Ks) brightness are used.

    One catalog query covers the snap search AND the whole zoom around the chosen star, so the
    alignment check can reuse it ('field_stars').
    """
    zoom_half_width = cfg.ZOOM_SIZE_PIX / 2 * cfg.PIXEL_ARCSEC * u.arcsec
    stars = query_catalog(coord, 'fp_psc', cfg.SNAP_RADIUS + max(cfg.ISOLATION_RADIUS, zoom_half_width * np.sqrt(2)))
    ks = np.ma.filled(stars['k_m'], 99.0)
    brightness = 10 ** (-0.4 * ks)
    positions = SkyCoord(stars['ra'], stars['dec'], unit='deg')

    chosen = None
    for i in np.flatnonzero((stars['dist_arcsec'] <= cfg.SNAP_RADIUS.to_value(u.arcsec)) & (ks <= cfg.SNAP_MAX_KS_MAG)):
        near = positions.separation(positions[i]) < cfg.ISOLATION_RADIUS
        near[i] = False
        if brightness[near].sum() <= cfg.ISOLATION_MAX_FRACTION * brightness[i]:
            chosen = i
            break
    if chosen is None:
        raise RuntimeError(f'No isolated 2MASS star brighter than Ks={cfg.SNAP_MAX_KS_MAG} within '
                           f'{cfg.SNAP_RADIUS} of the position - try another position or relax the settings')

    star = stars[chosen]
    position = positions[chosen]
    sed = catalog_fluxes(star, TWOMASS_BANDS)
    wise = query_catalog(position, 'allwise_p3as_psd', 3 * u.arcsec)         # the same star in AllWISE
    if len(wise):
        sed += catalog_fluxes(wise[0], ALLWISE_BANDS)

    in_zoom = positions.separation(position) <= zoom_half_width * np.sqrt(2)
    name = f"2MASS J{star['designation']}"
    magnitudes = {}
    for band, (mag_col, err_col) in (('J', ('j_m', 'j_msigcom')), ('H', ('h_m', 'h_msigcom')), ('K', ('k_m', 'k_msigcom'))):
        if not np.ma.is_masked(star[mag_col]):
            magnitudes[band] = (float(star[mag_col]), 0.05 if np.ma.is_masked(star[err_col]) else float(star[err_col]))
    log.info(f"Tracking {name}: {star['dist_arcsec']:.1f}\" from the entered position, Ks = {ks[chosen]:.2f} mag")
    return {'name': name, 'coord': position, 'catalog_sed': sed, 'field_stars': stars[in_zoom], 'magnitudes': magnitudes}
