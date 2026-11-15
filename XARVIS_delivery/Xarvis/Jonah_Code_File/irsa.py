"""IRSA: SPHEREx image search, catalogs (Gator) and cutouts.

IRSA is unreliable at times: it answers "502 Proxy Error", or accepts a connection and never replies. Every
request goes through net.py (timeouts, retries, caching), so neither can stop the program.
"""

import io
import logging
import threading

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.ipac.irsa import Irsa

from . import cache
from . import config as cfg
from . import net
from .net import RETRY_ERRORS, BadResponse   # noqa: F401  (re-exported: other modules import them from here)

log = logging.getLogger(__name__)

Irsa._session.mount('https://', net.TimeoutAdapter())     # used by Irsa.query_sia
Irsa._session.mount('http://', net.TimeoutAdapter())

_download_slots = threading.BoundedSemaphore(cfg.MAX_CONCURRENT_DOWNLOADS)


def with_retries(label, func, *args, quiet=False, **kwargs):
    """Call func(*args, **kwargs), retrying on IRSA errors (net.retry with IRSA's policy)."""
    return net.retry(label, func, *args, service='irsa', quiet=quiet, **kwargs)


def query_images(coord):
    """SPHEREx images covering `coord` (astropy Table), sorted by time - via astroquery (slower than
    query_images_at, which the analysis uses)."""
    results = with_retries('SPHEREx image search', Irsa.query_sia,
                           pos=(coord, 1 * u.arcsec), collection=cfg.COLLECTIONS)
    results.sort('t_min')
    return results


def _read_votable(content):
    return Table.read(io.BytesIO(content), format='votable', use_names_over_ids=True)


def _check_votable(content):
    try:
        _read_votable(content)
    except Exception as e:                      # an error page instead of a table
        raise BadResponse(f'unreadable reply: {e}')


def query_images_at(ra_deg, dec_deg, radius=0.02):
    """Every SPHEREx image whose footprint covers a sky position, at ANY time (astropy Table).

    One plain request - much lighter than astroquery's, which makes several. IRSA's own time filter is
    unreliable (it times out), so the caller filters by time (t_min / t_max, MJD). Replies are cached on disk
    (see cache.py) - a new SPHEREx data release can therefore take up to CACHE_MAX_AGE_DAYS to show up."""
    params = {'POS': f'CIRCLE {ra_deg:.6f} {dec_deg:.6f} {radius}', 'COLLECTION': cfg.COLLECTIONS, 'MAXREC': 5000}
    content = net.get(cfg.SIA_URL, params, service='irsa', label='SPHEREx image search', cache_kind='searches',
                      cache_key=cache.make_key(cfg.SIA_URL, params['POS'], *cfg.COLLECTIONS), check=_check_votable)
    return _read_votable(content)


CATALOG_COLUMNS = {'fp_psc': 'designation,ra,dec,j_m,j_msigcom,h_m,h_msigcom,k_m,k_msigcom',
                   'allwise_p3as_psd': 'designation,ra,dec,w1mpro,w1sigmpro,w2mpro,w2sigmpro',
                   'catwise_2020': 'source_name,ra,dec,pmra,pmdec,w1mpro,w2mpro'}


def _check_ipac(content):
    if not content.lstrip().startswith(b'\\'):  # IPAC tables start with '\'
        raise BadResponse('not an IPAC table')


def gator(catalog, coord, radius_arcsec, columns=None):
    """One cone search of an IRSA catalog through Gator (astropy Table), cached on disk. Gator rather than
    astroquery's Irsa.query_region (IRSA's TAP service): same catalogs, but TAP has been failing most requests
    while Gator answers reliably."""
    params = {'catalog': catalog, 'spatial': 'cone', 'objstr': f'{coord.ra.deg:.6f} {coord.dec.deg:.6f}',
              'radius': f'{radius_arcsec:.0f}' if radius_arcsec >= 10 else f'{radius_arcsec:.2f}',
              'radunits': 'arcsec', 'outfmt': 1, 'selcols': columns or CATALOG_COLUMNS[catalog]}  # 1 = IPAC table
    content = net.get(cfg.GATOR_URL, params, service='irsa', label=f'Catalog {catalog}', cache_kind='catalogs',
                      check=_check_ipac)
    return Table.read(content.decode(), format='ipac')


def query_catalog(coord, catalog, radius):
    """Catalog sources within `radius` (astropy Table), sorted by distance, with a 'dist_arcsec' column."""
    table = gator(catalog, coord, radius.to_value(u.arcsec))
    table['dist_arcsec'] = SkyCoord(table['ra'], table['dec'], unit='deg').separation(coord).arcsec
    table.sort('dist_arcsec')
    return table


def fetch_cutout(image_url, coord, size):
    """Small cutout (HDUList) of one SPHEREx image around `coord`, cut server-side by IRSA.

    IRSA sends ~4 MB per cutout, mostly a PSF table; only the extensions we use (plus the PSF at the cutout's
    position, see open_cutout) are kept, and those (~100 KB) are cached on disk so a re-run doesn't download them
    again (see cache.py). (= download_cutout + open_cutout; frames.load_frame calls the two separately.)"""
    return open_cutout(*download_cutout(image_url, coord, size))


def download_cutout(image_url, coord, size):
    """The waiting part of fetch_cutout: the cutout's bytes from the disk cache or from IRSA, no parsing - so
    many can run in threads at once. Returns (content, cache key, whether content is already trimmed)."""
    params = {'center': f'{coord.ra.deg:.6f},{coord.dec.deg:.6f}', 'size': f'{size.to_value(u.arcsec)}arcsec'}
    # ('with-psf': cutouts cached before the PSF was kept are fetched again once - they lack it)
    key = cache.make_key(image_url, params['center'], params['size'], 'with-psf')
    cached = cache.get('cutouts', key)
    if cached is not None:
        return cached, key, True
    # at most MAX_CONCURRENT_DOWNLOADS at a time, program-wide; net.Refused (a ValueError) = a real refusal.
    # IRSA sometimes answers 200 with an error page instead of the FITS file: that is retried.
    content = net.get(image_url, params, service='irsa', label='Cutout', slots=_download_slots, check=_check_fits)
    return content, key, False


def _check_fits(content):
    if not content.startswith(b'SIMPLE'):
        raise BadResponse('not a FITS file')


KEEP_EXTENSIONS = ('IMAGE', 'FLAGS', 'VARIANCE', 'ZODI', 'WCS-WAVE')


def open_cutout(content, key, trimmed):
    """The Python-work part of fetch_cutout: parse the bytes into an HDUList with only the extensions we use
    (a fresh download is trimmed and cached here). The large EPSF table is replaced by the one PSF image that
    applies at the cutout's centre (extension 'PSF', see psf.py), ~10 KB instead of ~4 MB."""
    if trimmed:
        return fits.open(io.BytesIO(content), memmap=False)
    from .psf import psf_hdu_for_cutout
    with fits.open(io.BytesIO(content), memmap=False) as full:
        hdulist = fits.HDUList([fits.PrimaryHDU(header=full[0].header)])
        for name in KEEP_EXTENSIONS:
            hdu = full[name]
            if isinstance(hdu, fits.CompImageHDU):          # FLAGS is compressed; keep it as a plain image
                hdu = fits.ImageHDU(data=hdu.data, header=hdu.header, name=name)
            hdulist.append(hdu.copy())
        psf = psf_hdu_for_cutout(full)
        if psf is not None:
            hdulist.append(psf)
    if cfg.CACHE_ENABLED:
        buffer = io.BytesIO()
        hdulist.writeto(buffer)
        cache.put('cutouts', key, buffer.getvalue())
    return hdulist
