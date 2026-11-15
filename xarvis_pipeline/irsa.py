"""Talking to IRSA: timeouts, retries, SPHEREx image search, catalogs (Gator) and cutouts.

IRSA is unreliable at times: it answers "502 Proxy Error", or accepts a connection and never replies.
Every request here has a timeout and is retried, so neither can stop the program.
"""

import io
import threading
import time

import requests
from requests.adapters import HTTPAdapter

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from pyvo.dal.exceptions import DALAccessError     # base of all pyvo errors: 502s, timeouts, bad replies

from . import cache
from . import config as cfg


class _TimeoutAdapter(HTTPAdapter):
    """Gives every request a timeout. astroquery/pyvo send theirs with none, so a stalled IRSA reply
    would make Python wait forever; with a timeout it fails and gets retried instead."""
    def send(self, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = cfg.REQUEST_TIMEOUT
        return super().send(request, **kwargs)


Irsa._session.mount('https://', _TimeoutAdapter())     # used by Irsa.query_sia
Irsa._session.mount('http://', _TimeoutAdapter())

_thread_local = threading.local()
_download_slots = threading.BoundedSemaphore(cfg.MAX_CONCURRENT_DOWNLOADS)


def _session():
    """One reusable connection pool per thread (reusing connections is much faster than reconnecting)."""
    if not hasattr(_thread_local, 'session'):
        session = requests.Session()
        session.mount('https://', _TimeoutAdapter(pool_maxsize=cfg.MAX_CONCURRENT_DOWNLOADS))
        _thread_local.session = session
    return _thread_local.session


class BadResponse(Exception):
    """IRSA answered, but not with what we asked for (e.g. an error page)."""


RETRY_ERRORS = (DALAccessError, requests.RequestException, OSError, BadResponse)


def retry_wait(attempt):
    """WAIT_SECONDS for the first FAST_ATTEMPTS retries; after that, back off (doubling, up to
    MAX_WAIT_SECONDS) so a longer IRSA outage can't use up all attempts within a few seconds."""
    if attempt < cfg.FAST_ATTEMPTS:
        return cfg.WAIT_SECONDS
    return min(0.1 * 2 ** (attempt - cfg.FAST_ATTEMPTS), cfg.MAX_WAIT_SECONDS)


def with_retries(label, func, *args, quiet=False, **kwargs):
    """Call func(*args, **kwargs), retrying on IRSA errors up to MAX_ATTEMPTS times."""
    for attempt in range(cfg.MAX_ATTEMPTS):
        try:
            return func(*args, **kwargs)
        except RETRY_ERRORS as e:
            if not quiet:
                message = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f'{label}: attempt {attempt + 1}/{cfg.MAX_ATTEMPTS} failed: {message[:120]}')
            time.sleep(retry_wait(attempt))
    raise RuntimeError(f'{label}: all {cfg.MAX_ATTEMPTS} attempts failed.')


def query_images(coord):
    """SPHEREx images covering `coord` (astropy Table), sorted by time."""
    results = with_retries('SPHEREx image search', Irsa.query_sia,
                           pos=(coord, 1 * u.arcsec), collection=cfg.COLLECTIONS)
    results.sort('t_min')
    return results


def query_images_at(ra_deg, dec_deg, radius=0.02):
    """Every SPHEREx image whose footprint covers a sky position, at ANY time (astropy Table).

    One plain request - much lighter than astroquery's, which makes several. IRSA's own time filter is
    unreliable (it times out), so the caller filters by time (t_min / t_max, MJD). Replies are cached on disk
    (see cache.py) - a new SPHEREx data release can therefore take up to CACHE_MAX_AGE_DAYS to show up."""
    params = {'POS': f'CIRCLE {ra_deg:.6f} {dec_deg:.6f} {radius}', 'COLLECTION': cfg.COLLECTIONS, 'MAXREC': 5000}
    key = cache.make_key(cfg.SIA_URL, params['POS'], *cfg.COLLECTIONS)

    def read(content):
        return Table.read(io.BytesIO(content), format='votable', use_names_over_ids=True)

    cached = cache.get('searches', key)
    if cached is not None:
        return read(cached)

    def once():
        response = _session().get(cfg.SIA_URL, params=params)
        if response.status_code != 200:
            raise BadResponse(f'HTTP {response.status_code}')
        try:
            table = read(response.content)
        except Exception as e:                      # an error page instead of a table
            raise BadResponse(f'unreadable reply: {e}')
        cache.put('searches', key, response.content)
        return table

    return with_retries('SPHEREx image search', once, quiet=True)


CATALOG_COLUMNS = {'fp_psc': 'designation,ra,dec,j_m,j_msigcom,h_m,h_msigcom,k_m,k_msigcom',
                   'allwise_p3as_psd': 'designation,ra,dec,w1mpro,w1sigmpro,w2mpro,w2sigmpro',
                   'catwise_2020': 'source_name,ra,dec,pmra,pmdec,w1mpro,w2mpro'}


def query_catalog(coord, catalog, radius):
    """Catalog sources within `radius` (astropy Table), sorted by distance, with a 'dist_arcsec' column.

    Uses IRSA's Gator service rather than astroquery's Irsa.query_region (IRSA's TAP service):
    same catalogs, but TAP has been failing most requests while Gator answers reliably.
    """
    params = {'catalog': catalog, 'spatial': 'cone', 'objstr': f'{coord.ra.deg} {coord.dec.deg}',
              'radius': radius.to_value(u.arcsec), 'radunits': 'arcsec', 'outfmt': 1,     # 1 = IPAC table
              'selcols': CATALOG_COLUMNS[catalog]}

    def once():
        response = _session().get(cfg.GATOR_URL, params=params)
        if response.status_code != 200 or not response.text.lstrip().startswith('\\'):  # IPAC tables start with '\'
            raise BadResponse(f'HTTP {response.status_code}')
        return Table.read(response.text, format='ipac')

    table = with_retries(f'Catalog {catalog}', once)
    table['dist_arcsec'] = SkyCoord(table['ra'], table['dec'], unit='deg').separation(coord).arcsec
    table.sort('dist_arcsec')
    return table


def fetch_cutout(image_url, coord, size):
    """Small cutout (HDUList) of one SPHEREx image around `coord`, cut server-side by IRSA.

    IRSA sends ~4 MB per cutout, mostly a PSF table we don't use; only the extensions we use are kept, and
    those (~95 KB) are cached on disk so a re-run doesn't download them again (see cache.py).
    (= download_cutout + open_cutout; frames.load_frame calls the two separately.)"""
    return open_cutout(*download_cutout(image_url, coord, size))


def download_cutout(image_url, coord, size):
    """The waiting part of fetch_cutout: the cutout's bytes from the disk cache or from IRSA, no parsing - so
    many can run in threads at once. Returns (content, cache key, whether content is already trimmed)."""
    params = {'center': f'{coord.ra.deg:.6f},{coord.dec.deg:.6f}', 'size': f'{size.to_value(u.arcsec)}arcsec'}
    key = cache.make_key(image_url, params['center'], params['size'])
    cached = cache.get('cutouts', key)
    if cached is not None:
        return cached, key, True

    def once():
        with _download_slots:                       # at most MAX_CONCURRENT_DOWNLOADS at a time, program-wide
            response = _session().get(image_url, params=params)
        if response.status_code in (429, 500, 502, 503, 504):
            raise BadResponse(f'HTTP {response.status_code}')       # IRSA is struggling: try again
        if response.status_code != 200:                             # a real refusal: retrying can't help
            raise ValueError(f'IRSA refused the cutout (HTTP {response.status_code})')
        return response.content

    return with_retries('Cutout', once, quiet=True), key, False


def open_cutout(content, key, trimmed):
    """The Python-work part of fetch_cutout: parse the bytes into an HDUList with only the extensions we use
    (a fresh download is trimmed and cached here). frames.load_frame does this one frame at a time."""
    if trimmed:
        return fits.open(io.BytesIO(content), memmap=False)
    with fits.open(io.BytesIO(content), memmap=False) as full:
        hdulist = fits.HDUList([fits.PrimaryHDU(header=full[0].header)])
        for name in ['IMAGE', 'FLAGS', 'VARIANCE', 'ZODI', 'WCS-WAVE']:      # drop the large EPSF table
            hdu = full[name]
            if isinstance(hdu, fits.CompImageHDU):          # FLAGS is compressed; keep it as a plain image
                hdu = fits.ImageHDU(data=hdu.data, header=hdu.header, name=name)
            hdulist.append(hdu.copy())
    if cfg.CACHE_ENABLED:
        trimmed = io.BytesIO()
        hdulist.writeto(trimmed)
        cache.put('cutouts', key, trimmed.getvalue())
    return hdulist
