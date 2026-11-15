"""A star (or brown dwarf) by name: catalog position, proper motion, parallax and radial velocity from SIMBAD
(Gaia DR3 where it has it), and where it was at the moment of every SPHEREx exposure.

Nearby stars move: Barnard's Star by 10.4" a year, so its 2MASS position (taken ~1999) is now ~4.5' away from
where SPHEREx sees it. Every SPHEREx frame is therefore cut around the star's position AT THAT FRAME'S TIME.
"""

import logging
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.time import Time

from . import config as cfg
from . import net
from .irsa import BadResponse, query_catalog
from .target import ALLWISE_BANDS, catalog_fluxes

log = logging.getLogger(__name__)

SIMBAD_URLS = ('https://simbad.harvard.edu/simbad/sim-id',       # mirror (the main CDS address has moved)
               'https://simbad.cds.unistra.fr/simbad/sim-id')
AU_PER_PARSEC = 206264.806
KM_PER_S_IN_AU_PER_YEAR = 0.210945
MAS = np.pi / (180 * 3600 * 1000)                                 # one milliarcsecond in radians
J2000_MJD = 51544.5
ALLWISE_EPOCH_MJD = 55400.0                                       # ~2010.5, mid-point of the AllWISE survey
ZERO_POINTS_JY = {'J': 1594.0, 'H': 1024.0, 'K': 666.7}          # Vega magnitude -> Jy (2MASS)
BAND_UM = {'J': 1.235, 'H': 1.662, 'K': 2.159}


# ----------------------------------------------------------------------------
# SIMBAD
# ----------------------------------------------------------------------------

def resolve_star(name):
    """SIMBAD's entry for `name` (dict: name, ra, dec at epoch J2000, pmra/pmdec in mas/yr (pmra = RA*cos Dec),
    parallax in mas, rv in km/s - each None if SIMBAD has none - magnitudes, spectral and object type)."""
    params = {'Ident': name, 'output.format': 'ASCII', 'output.console': 'off', 'output.script': 'off'}

    def check(content):                     # an entry, or SIMBAD's "not found" answer - anything else is retried
        if b'Coordinates(ICRS' not in content and b'!!' not in content:
            raise BadResponse('not a SIMBAD answer')

    last = None
    for url in SIMBAD_URLS:                 # the mirror first; the main CDS address if the mirror is down
        try:
            text = net.get(url, params, service='cds', label=f'SIMBAD "{name}"', check=check).decode(errors='replace')
        except (RuntimeError, net.Refused) as e:
            last = e
            continue
        if 'Coordinates(ICRS' not in text:
            raise ValueError(f'SIMBAD does not know "{name}": {text.strip().splitlines()[-1][:150]}')
        return parse_simbad(text, name)
    raise RuntimeError(f'SIMBAD unreachable ({last})')


def _numbers(text, label, count):
    """The first `count` numbers after 'label:' on its line (None for '~' = no value)."""
    match = re.search(rf'^{re.escape(label)}\s*:\s*(.*)$', text, re.M)
    if not match:
        return [None] * count
    values = []
    for token in match.group(1).split()[:count]:
        try:
            values.append(float(token.strip('[]')))                  # uncertainties are written as [0.02]
        except ValueError:
            values.append(None)
    return values + [None] * (count - len(values))


def parse_simbad(text, name):
    header = re.search(r'^Object (.+?)\s+---\s+(\S+)\s+---', text, re.M)
    coords = re.search(r'^Coordinates\(ICRS,ep=J2000,eq=2000\):\s*(\S+ \S+ \S+)\s+(\S+ \S+ \S+)', text, re.M)
    position = SkyCoord(coords.group(1), coords.group(2), unit=(u.hourangle, u.deg))
    pmra, pmdec = _numbers(text, 'Proper motions', 2)
    parallax, = _numbers(text, 'Parallax', 1)
    rv, = _numbers(text, 'Radial Velocity', 1)
    magnitudes = {}
    for band in ('J', 'H', 'K'):
        mag, err = _numbers(text, f'Flux {band} ', 2)
        if mag is not None:
            magnitudes[band] = (mag, err if err is not None else 0.05)
    spectral = re.search(r'^Spectral type:\s*(\S+)', text, re.M)
    main_id = re.sub(r'^(NAME|V\*|\*\*|\*)\s+', '', header.group(1).strip()) if header else name   # 'NAME Barnard's star'
    star = {'name': main_id, 'object_type': header.group(2) if header else '?',
            'query': name, 'ra': position.ra.deg, 'dec': position.dec.deg, 'pmra': pmra, 'pmdec': pmdec,
            'parallax': parallax, 'rv': rv, 'magnitudes': magnitudes,
            'spectral_type': spectral.group(1) if spectral and spectral.group(1) != '~' else '?'}
    pm = np.hypot(pmra or 0, pmdec or 0)
    log.info(f"{star['name']} ({star['spectral_type']}, SIMBAD type {star['object_type']}): proper motion {pm / 1000:.2f}\"/yr, "
          f"parallax {parallax if parallax is not None else '?'} mas"
          + (f" ({1000 / parallax:.2f} pc)" if parallax else '') + f", radial velocity {rv if rv is not None else '?'} km/s")
    return star


# ----------------------------------------------------------------------------
# Where the star is at a given time
# ----------------------------------------------------------------------------

def positions_at(star, mjd):
    """(RA, Dec) in degrees of the star at time(s) `mjd`, as seen from Earth.

    The catalog position (SIMBAD's, at J2000 - or at star['epoch_mjd'] if the star has one), moved along the star's straight-line motion through space (proper motion and
    radial velocity - the latter also gives the slow change of proper motion of very nearby stars), plus the
    annual parallax from Earth's position around the Sun. SPHEREx orbits 7000 km from Earth's centre: for
    parallax that's < 0.00001". No aberration: SPHEREx's sky coordinates are fitted to catalog positions."""
    mjd = np.atleast_1d(np.asarray(mjd, dtype=float))
    ra, dec = np.radians(star['ra']), np.radians(star['dec'])
    toward = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])

    parallax = star['parallax'] if star['parallax'] and star['parallax'] > 0 else None
    distance = AU_PER_PARSEC / (parallax / 1000) if parallax else 1e12          # AU (unknown: effectively infinite)
    velocity = (distance * MAS * ((star['pmra'] or 0) * east + (star['pmdec'] or 0) * north)
                + (star['rv'] or 0) * KM_PER_S_IN_AU_PER_YEAR * toward)         # AU per year
    years = (mjd - (star.get('epoch_mjd') or J2000_MJD)) / 365.25      # 'epoch_mjd': the epoch of ra/dec (Gaia: J2016.0)
    where = distance * toward[None, :] + years[:, None] * velocity[None, :]    # barycentric, AU
    if parallax:
        earth = get_body_barycentric('earth', Time(mjd, format='mjd'))
        where = where - np.stack([earth.x.to_value(u.au), earth.y.to_value(u.au), earth.z.to_value(u.au)], axis=1)
    ra_t = np.degrees(np.arctan2(where[:, 1], where[:, 0])) % 360
    dec_t = np.degrees(np.arcsin(where[:, 2] / np.linalg.norm(where, axis=1)))
    return ra_t, dec_t


def position_at(star, mjd):
    ra, dec = positions_at(star, mjd)
    return SkyCoord(ra[0], dec[0], unit='deg')


# ----------------------------------------------------------------------------
# Contamination: other sources inside the measuring circle
# ----------------------------------------------------------------------------

CATWISE_EPOCH_MJD = 57170.0                                      # ~2015.4, the CatWISE2020 reference epoch
WISE_YEARS_MJD = (55200.0, 58850.0)                               # 2010.0 - 2020.0: the WISE / NEOWISE data
W1_JY, W2_JY = 309.54, 171.787


def contamination_check(star, mjd, sed=None):
    """Other sources inside the measuring circle around the target at time `mjd`, from CatWISE2020 (infrared;
    positions moved with their own proper motions). For faint targets these can dominate what SPHEREx measures:
    returns a list of (name, separation_arcsec, W1 mJy, W2 mJy), and warns, comparing with the target's own W1/W2
    (from `sed`, the catalog brightness, if given - otherwise CatWISE).

    The target itself: any CatWISE source lying on the target's own path during the WISE years (2010-2020).
    A fast mover is often split into several catalog entries along its path - all of those are the target."""
    radius_arcsec = (cfg.APERTURE_RADIUS_PIX + 1) * cfg.PIXEL_ARCSEC
    here = position_at(star, mjd)
    track_ra, track_dec = positions_at(star, np.arange(WISE_YEARS_MJD[0], WISE_YEARS_MJD[1], 15.0))
    track = SkyCoord(track_ra, track_dec, unit='deg')
    reach = max(here.separation(track).arcsec)             # the target's own past positions must be in the search
    try:
        table = query_catalog(here, 'catwise_2020', (radius_arcsec + reach + 10) * u.arcsec)
    except (RuntimeError, KeyError) as e:
        log.info(f'Contamination check: CatWISE not available ({e})')
        return []
    years = (mjd - CATWISE_EPOCH_MJD) / 365.25
    flux = lambda mag, zero: zero * 10 ** (-0.4 * float(mag)) * 1e3 if not np.ma.is_masked(mag) else np.nan
    target, others, fragments = None, [], 0
    for row in table:
        at_epoch = SkyCoord(row['ra'], row['dec'], unit='deg')
        if np.min(at_epoch.separation(track).arcsec) < 4:  # on the target's path: the target (or a piece of it)
            fragments += 1
            if target is None:
                target = (flux(row['w1mpro'], W1_JY), flux(row['w2mpro'], W2_JY))
            continue
        pm_ra = 0.0 if np.ma.is_masked(row['pmra']) else float(row['pmra'])         # arcsec/yr
        pm_dec = 0.0 if np.ma.is_masked(row['pmdec']) else float(row['pmdec'])
        now = at_epoch.spherical_offsets_by(pm_ra * years * u.arcsec, pm_dec * years * u.arcsec)
        separation = now.separation(here).arcsec
        if separation <= radius_arcsec:
            others.append((str(row['source_name']), separation, flux(row['w1mpro'], W1_JY), flux(row['w2mpro'], W2_JY)))
    others.sort(key=lambda o: o[1])

    if sed:                                                 # the catalog's (AllWISE) values for the target
        allwise = {name: value for name, _, value, _ in sed if name.startswith('WISE')}
        if 'WISE W1' in allwise and 'WISE W2' in allwise:
            target = (allwise['WISE W1'], allwise['WISE W2'])
    note = f' (the target itself appears as {fragments} CatWISE entries along its path)' if fragments > 1 else ''
    if not others:
        log.info(f'Contamination check: no other CatWISE source inside the measuring circle ({radius_arcsec:.0f}"){note}')
        return others
    total_w1, total_w2 = np.nansum([o[2] for o in others]), np.nansum([o[3] for o in others])
    log.info(f'Contamination check: {len(others)} other source(s) inside the measuring circle ({radius_arcsec:.0f}"){note}:')
    for name, separation, w1, w2 in others:
        log.info(f'    CatWISE {name} at {separation:.1f}": W1 {w1:.3f} mJy, W2 {w2:.3f} mJy')
    if target:
        log.info(f'  target itself: W1 {target[0]:.3f} mJy, W2 {target[1]:.3f} mJy -> the others add '
              f'{total_w1 / target[0] * 100:.0f}% at W1 (3.4 um) and {total_w2 / target[1] * 100:.0f}% at W2 (4.6 um)')
        if total_w1 > 0.1 * target[0] or total_w2 > 0.1 * target[1]:
            log.info('  WARNING: measurements at the target are contaminated - especially at short wavelengths, where '
                  'the target may be much fainter than its neighbours. Positions may follow the neighbour.')
    return others


# ----------------------------------------------------------------------------
# Catalog brightness (for the spectrum check)
# ----------------------------------------------------------------------------

def catalog_sed(star):
    """The star's catalog brightness: J/H/K from SIMBAD, W1/W2 from AllWISE (matched at the star's position in
    ~2010, the AllWISE epoch, since it moves). List of (band, wavelength_um, flux_mjy, flux_err_mjy)."""
    sed = []
    for band, (mag, err) in star['magnitudes'].items():
        flux = ZERO_POINTS_JY[band] * 10 ** (-0.4 * mag) * 1e3
        sed.append((f'{band} (SIMBAD)', BAND_UM[band], flux, flux * 0.4 * np.log(10) * err))

    pm_arcsec_per_year = np.hypot(star['pmra'] or 0, star['pmdec'] or 0) / 1000
    then = position_at(star, ALLWISE_EPOCH_MJD)
    radius = (3 + 1.0 * pm_arcsec_per_year) * u.arcsec                         # AllWISE spans ~a year
    try:
        wise = query_catalog(then, 'allwise_p3as_psd', radius)
        if len(wise):
            sed += catalog_fluxes(wise[0], ALLWISE_BANDS)
    except RuntimeError as e:
        log.info(f'  (AllWISE not available: {e})')
    return sed
