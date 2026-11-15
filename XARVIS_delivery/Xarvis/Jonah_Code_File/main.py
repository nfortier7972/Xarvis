"""Run the whole analysis: run __main__.py, or python -m Jonah_Code_File (settings: config.json; or the command
line - python -m Jonah_Code_File --help)."""

import logging
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from . import cache
from . import config as cfg
from .frames import load_all_frames, pointing_summary, zoom_original
from .irsa import query_catalog, query_images_at
from .measure import (brightness_changes, check_alignment, compare_with_catalog, detection_summary,
                      reliable_frames, spectrum_check, track_moving, track_object)
from .moving import add_predicted_positions, find_asteroid, find_images
from .plots import LiveView, show_motion, show_spectrum, show_time_series, show_viewer
from .results import save_results
from .stars import catalog_sed, contamination_check, position_at, positions_at, resolve_star
from .target import find_tracked_object

log = logging.getLogger(__name__)

MID_DATA_MJD = 61000.0          # ~2025-11, middle of the SPHEREx data so far: where to search for a moving star

# Headless callers (the website: web.py) set this to a function(frame, done, total) to follow the downloads; it is
# passed to load_all_frames(on_frame=...) when no live window is shown. None (the default): nothing is called.
PROGRESS_HOOK = None


def select_images(images, max_images):
    """The images to analyse out of all that cover the target (sorted by time). No max_images: all of them.
    Otherwise the first `max_images` in time - or, with config.SPREAD_IMAGES (the website's time-lapse), up to that
    many spread evenly over the whole list, first and last included, so every survey visit is represented. The spread
    picks whole EXPOSURES: the images taken at the same moment (the target on a short- and a long-wavelength
    detector) stay together, since the display colour of each comes from the other (visual.py)."""
    if not max_images or len(images) <= max_images:
        return images
    if not cfg.SPREAD_IMAGES:
        return images[:max_images]
    t = np.asarray(images['t_min'], dtype=float)
    exposure = np.concatenate([[0], np.cumsum(np.diff(t) * 86400 > 5)])     # same exposure: within 5 s
    n_exposures = int(exposure[-1]) + 1
    for count in range(min(max_images, n_exposures), 0, -1):                  # as many exposures as fit
        picked = np.isin(exposure, np.unique(np.round(np.linspace(0, n_exposures - 1, count)).astype(int)))
        if picked.sum() <= max_images:
            return images[np.flatnonzero(picked)]
    return images[:max_images]                  # (an exposure with more images than max_images: its first ones)


def start_live_view(tracked, images, show):
    """The window that fills in while frames download (or None): only when the plots are shown and
    config.LIVE_VIEW is on. Its `update` goes to load_all_frames(on_frame=...)."""
    if not (show and cfg.LIVE_VIEW):
        return None
    return LiveView(tracked, len(images), float(min(images['t_min'])))


def make_zooms(frames, zoom_only):
    """Original-pixel zoom of every frame, in order. zoom_only=True releases each frame's full-size pixel
    arrays as its zoom is cut, so only the zooms stay in memory."""
    zooms = []
    for frame in frames:
        zooms.append(zoom_original(frame, size_pix=cfg.ZOOM_SIZE_PIX, release_original=zoom_only))   # the size now, not at import
    return zooms


def parse_coords(coords):
    """A sky position from '08h55m10.8s -07d14m42s', '08:55:10.8 -07:14:42', '133.795 -7.245' or (ra, dec) in
    degrees."""
    if isinstance(coords, SkyCoord):
        return coords
    if not isinstance(coords, str):
        ra, dec = coords
        return SkyCoord(float(ra), float(dec), unit='deg')
    text = coords.strip().replace(',', ' ')
    if any(c in text for c in 'hms:') or len(text.split()) == 6:      # sexagesimal: RA in hours
        return SkyCoord(text, unit=(u.hourangle, u.deg))
    return SkyCoord(text, unit='deg')


def run(ra_deg=cfg.RA_DEG, dec_deg=cfg.DEC_DEG, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY,
        asteroid=None, star=None, coords=None, window=cfg.ASTEROID_WINDOW):
    """Run the whole analysis. Tracks, in order of preference: the asteroid in config.ASTEROID (or the
    `asteroid` argument); else the star named in config.STAR (or `star`, e.g. "Barnard's Star"); else exactly
    the sky position config.COORDS (or `coords`); else the nearest bright isolated star to ra_deg / dec_deg.

    Returns a dict with 'tracked', 'frames', 'zooms', 'movement' (+ 'brightness' and 'spectrum' for a star).
    The measurements are also saved to results/<target>.ecsv (config.SAVE_RESULTS).

    zoom_only=True (the default, config.ZOOM_ONLY): after the analysis, keep only the zoomed pixels in memory -
    each frame's full-size cutout arrays are released as its zoom is cut (results['frames'] then hold
    measurements only, and results['zooms'] hold the pixels). False keeps both.

    A target passed here (asteroid / star / coords) replaces config.json's; with none, config.json decides."""
    if asteroid is None and star is None and coords is None:
        asteroid, star, coords = cfg.ASTEROID, cfg.STAR, cfg.COORDS
    if asteroid:
        return run_asteroid(asteroid, window, max_images, show, zoom_only)
    if star:
        return run_named_star(star, max_images, show, zoom_only)
    if coords is not None:
        return run_coords(coords, max_images, show, zoom_only)
    return run_star(ra_deg, dec_deg, max_images, show, zoom_only)


def run_named_star(name=cfg.STAR, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """A star or brown dwarf by name. SIMBAD (Gaia DR3 where available) gives its position, proper motion,
    parallax and radial velocity, so every SPHEREx frame is cut around where it was AT THAT FRAME'S TIME - nearby
    stars move several arcseconds a year. Its motion is then measured from SPHEREx alone (against T1) and
    compared with the catalog; its brightness against its own spectrum (variability); and if it's too faint for
    single frames, frames are combined to test whether SPHEREx detects it at all."""
    star = resolve_star(name)                                  # SIMBAD: position, motion, parallax
    return run_fixed_target(star, {'mode': 'named star', 'star': name}, max_images, show, zoom_only)


def run_coords(coords=cfg.COORDS, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """Point at EXACTLY this sky position (no snapping to a star). It is treated as fixed on the sky: whatever is
    there is measured in every frame, anything that moves away from it or changes brightness is flagged, and the
    whole field around it is compared with Gaia DR3 / CatWISE (what's known there) and JPL (known asteroids)."""
    coord = parse_coords(coords)
    name = f"RA {coord.ra.to_string(u.hourangle, sep=':', precision=1, pad=True)} Dec {coord.dec.to_string(sep=':', precision=0, alwayssign=True, pad=True)}"
    magnitudes = {}
    try:                                                       # 2MASS brightness of whatever is there (spectrum check)
        twomass = query_catalog(coord, 'fp_psc', 3 * u.arcsec)
        if len(twomass):
            for band, column in (('J', 'j_m'), ('H', 'h_m'), ('K', 'k_m')):
                mag, err = twomass[0][column], twomass[0][f'{column[0]}_msigcom']
                if not np.ma.is_masked(mag):
                    magnitudes[band] = (float(mag), 0.05 if np.ma.is_masked(err) else float(err))
    except RuntimeError as e:
        log.info(f'  (2MASS not available: {e})')
    star = {'name': name, 'object_type': 'sky position', 'query': str(coords), 'ra': coord.ra.deg,
            'dec': coord.dec.deg, 'pmra': 0.0, 'pmdec': 0.0, 'parallax': None, 'rv': None,
            'magnitudes': magnitudes, 'spectral_type': '?'}
    log.info(f'Pointing at {name} ({coord.ra.deg:.5f}, {coord.dec.deg:+.5f} deg) - fixed on the sky'
          + (f"; 2MASS source there: K = {magnitudes['K'][0]:.2f}" if 'K' in magnitudes else '; no 2MASS source within 3"'))
    return run_fixed_target(star, {'mode': 'coordinates', 'coords': str(coords)}, max_images, show, zoom_only)


def field_check(frames, coord, unknowns=True):
    """Every frame's sources vs Gaia DR3 / CatWISE, unexplained sources looked up at JPL (see field.py).
    Needs the full pixels (before make_zooms). Returns (field table, summary) - (None, None) if the catalogs
    can't be reached (never stops the run). unknowns=False: only the pointing check against Gaia - when the sky
    map runs too, it finds and identifies the unknown sources itself (with ONE JPL call), so this doesn't."""
    if not cfg.FIELD_CHECK:
        return None, None
    from .field import describe_target, field_summary, identify_candidates, query_field
    try:
        field = query_field(coord)
        describe_target(field, coord, frames[0]['mjd'])
        summary = field_summary(frames, field, target=[(f['expected_ra'], f['expected_dec']) for f in frames],
                                unknowns=unknowns)
        if unknowns:
            identify_candidates(summary['candidates'])
        return field, summary
    except RuntimeError as e:
        log.info(f'Field comparison with Gaia / CatWISE skipped: {e}')
        return None, None


def sky_map(frames, coord):
    """Name every object in the frames and find the new ones (skymap.py). Needs the full pixels. None if the
    reference services can't be reached (never stops the run)."""
    from .skymap import map_sky
    try:
        return map_sky(frames, coord)
    except RuntimeError as e:
        log.info(f'Sky map skipped: {e}')
        return None


def run_fixed_target(star, settings, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """The analysis of anything fixed on the sky, or moving by a known proper motion: a named star (from SIMBAD)
    or a sky position (no motion). See run_named_star / run_coords."""
    start = time.time()
    live = None
    try:
        now = position_at(star, MID_DATA_MJD)
        images = query_images_at(now.ra.deg, now.dec.deg, radius=0.01)      # every image that covered it
        if not len(images):
            raise RuntimeError(f"No SPHEREx image covers {star['name']} yet")
        images.sort('t_min')
        images_available = len(images)
        images = select_images(images, max_images)
        t_mid = 0.5 * (np.array(images['t_min'], dtype=float) + np.array(images['t_max'], dtype=float))
        images['target_ra'], images['target_dec'] = positions_at(star, t_mid)          # where it was each time
        log.info(f"{len(images)} SPHEREx images cover {star['name']}")
        zoom_corner = cfg.ZOOM_SIZE_PIX / 2 * cfg.PIXEL_ARCSEC * np.sqrt(2) * u.arcsec
        tracked = {'name': star['name'], 'coord': now, 'star': star, 'catalog_sed': catalog_sed(star),
                   'field_stars': query_catalog(now, 'fp_psc', zoom_corner)}       # for the alignment check
        tracked['images_available'] = images_available         # before max_images (shown by the website)
        contamination = contamination_check(star, MID_DATA_MJD, tracked['catalog_sed'])   # others in the circle

        live = start_live_view(tracked, images, show)          # shows frames as they come in
        frames = load_all_frames(None, images, on_frame=live.update if live else PROGRESS_HOOK, moving=False)
        if live:
            live.close()
            live = None
        if not frames:
            raise RuntimeError('None of the images could be measured at this position')
        pointing_summary(frames)
        check_alignment(frames, tracked['field_stars'])        # verifies the pointing, never adjusts it
        detection = detection_summary(frames)                  # (needs the full pixels: before make_zooms)
        use_sky_map = settings.get('mode') == 'coordinates' or cfg.SKYMAP_FOR_ALL_TARGETS
        tracked['field'], field = field_check(frames, now, unknowns=not use_sky_map)   # vs Gaia / CatWISE / JPL
        sky = None
        if use_sky_map:
            sky = tracked['skymap'] = sky_map(frames, now)      # name every object, find the new ones
        movement = track_object(frames)                        # SPHEREx alone: position vs T1
        catalog_motion = compare_with_catalog(frames, movement, star)     # vs SIMBAD/Gaia proper motion
        brightness = brightness_changes(frames)                # brightness vs its own spectrum (variability)
        if any(f.get('position_method') == 'wings' for f in frames):      # saturated: relative, from the glow
            from_glow = brightness_changes(frames, glow=True)
            if from_glow['used'].sum() > brightness['used'].sum():
                brightness = from_glow                         # the more complete of the two is shown
        snr = np.array([f['snr'] for f in frames], dtype=float)
        measured = np.sum(reliable_frames(frames) & np.isfinite(snr) & (snr >= cfg.DETECTION_SNR))
        spectrum = spectrum_check(tracked, frames) if measured >= 10 and len(tracked['catalog_sed']) >= 3 else None
        if spectrum is None and (measured < 10 or len(tracked['catalog_sed']) < 3):
            log.info(f'Spectrum check: skipped - only {measured} frames with a measured brightness (saturated or too faint), '
                  f'or too few catalog points')
        zooms = make_zooms(frames, zoom_only)
    finally:
        if live:
            live.close()
        cache.prune()                                          # clean up the disk cache, even if the run failed
    log.info(f'Done in {time.time() - start:.0f} s')

    results = {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement, 'brightness': brightness,
               'spectrum': spectrum, 'detection': detection, 'catalog_motion': catalog_motion,
               'contamination': contamination, 'field': field, 'skymap': sky}
    if cfg.SAVE_RESULTS:
        save_results(results, {**settings, 'max_images': max_images})
    if show:
        show_viewer(zooms, tracked, movement, brightness)
        show_time_series(frames, tracked, movement, brightness)
        if spectrum:
            show_spectrum(spectrum, tracked)
        plt.show()
    return results


def run_asteroid(asteroid=cfg.ASTEROID, window=cfg.ASTEROID_WINDOW, max_images=cfg.MAX_IMAGES, show=True,
                 zoom_only=cfg.ZOOM_ONLY):
    """A moving object: JPL Horizons predicts where the asteroid was at every SPHEREx exposure, IRSA finds
    the images that caught it, and each image is cut around the predicted position and measured there."""
    tracked = find_asteroid(asteroid, window)                  # its path, from JPL Horizons
    return run_moving(tracked, {'mode': 'asteroid', 'asteroid': asteroid, 'window': window}, max_images, show, zoom_only)


def run_path(name, mjd, ra, dec, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """A moving object with no known orbit, along a straight path fitted to its detections (a new object found
    by the sky map): every SPHEREx image along that path in its time span is searched for it."""
    from .moving import Ephemeris
    t = np.linspace(min(mjd), max(mjd), 50)
    order = np.argsort(mjd)
    ra_t = np.degrees(np.interp(t, np.asarray(mjd)[order], np.unwrap(np.radians(np.asarray(ra)[order])))) % 360
    dec_t = np.interp(t, np.asarray(mjd)[order], np.asarray(dec)[order])
    tracked = {'name': name, 'asteroid': None, 'ephemeris': Ephemeris.from_arrays(t, ra_t, dec_t),
               'window': (min(mjd), max(mjd))}
    log.info(f'Tracking {name} along its fitted path, MJD {min(mjd):.2f} - {max(mjd):.2f}')
    return run_moving(tracked, {'mode': 'path', 'name': name, 'mjd': list(mjd), 'ra': list(ra), 'dec': list(dec)},
                      max_images, show, zoom_only, spacing_deg=0.05)


def run_moving(tracked, settings, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY,
               spacing_deg=cfg.GRID_SPACING_DEG):
    """The analysis of anything moving along a known path (tracked['ephemeris'])."""
    start = time.time()
    live = None
    try:
        images = find_images(tracked, spacing_deg)             # the SPHEREx images that caught it
        if not len(images):
            raise RuntimeError('No SPHEREx image caught the asteroid in this window - try another window (config.json)')
        tracked['images_available'] = len(images)              # before max_images (shown by the website)
        images = select_images(images, max_images)
        images = add_predicted_positions(images, tracked)      # its exact position, as seen from SPHEREx

        live = start_live_view(tracked, images, show)          # shows frames as they come in
        frames = load_all_frames(None, images, on_frame=live.update if live else PROGRESS_HOOK)
        if live:
            live.close()                                       # all frames are in: the full analysis follows
            live = None
        pointing_summary(frames)
        movement = track_moving(frames)                        # distance travelled, found where predicted?
        zooms = make_zooms(frames, zoom_only)
    finally:
        if live:
            live.close()
        cache.prune()                                          # clean up the disk cache, even if the run failed
    log.info(f'Done in {time.time() - start:.0f} s')

    results = {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement}
    if cfg.SAVE_RESULTS:
        save_results(results, {**settings, 'max_images': max_images})
    if show:
        show_viewer(zooms, tracked, movement)
        show_motion(frames, tracked, movement)
        plt.show()
    return results


def run_star(ra_deg=cfg.RA_DEG, dec_deg=cfg.DEC_DEG, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """The nearest bright, isolated 2MASS star to (ra_deg, dec_deg) (see target.find_tracked_object), analysed like
    any star: it stays put on the sky, so its brightness and position are compared with its own history."""
    snapped = find_tracked_object(SkyCoord(ra_deg, dec_deg, unit='deg'))
    star = {'name': snapped['name'], 'object_type': '2MASS star', 'query': f'{ra_deg} {dec_deg}',
            'ra': snapped['coord'].ra.deg, 'dec': snapped['coord'].dec.deg, 'pmra': 0.0, 'pmdec': 0.0,
            'parallax': None, 'rv': None, 'magnitudes': snapped['magnitudes'], 'spectral_type': '?'}
    return run_fixed_target(star, {'mode': 'star', 'ra_deg': ra_deg, 'dec_deg': dec_deg}, max_images, show, zoom_only)


def run_catalog_star(star, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """A star picked from the sky map (Gaia / Hipparcos position, motion and parallax)."""
    return run_fixed_target(star, {'mode': 'catalog star', 'star': star}, max_images, show, zoom_only)


def run_track(spec, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """Track an object picked in the viewer (see skymap.py: each mapped object carries how to track it)."""
    kind = spec['kind']
    if kind == 'star':
        return run_catalog_star(spec['star'], max_images, show, zoom_only)
    if kind == 'asteroid':
        return run_asteroid(spec['id'], tuple(spec['window']), max_images, show, zoom_only)
    if kind == 'coords':
        return run_coords(tuple(spec['coords']), max_images, show, zoom_only)
    if kind == 'path':
        return run_path(spec['name'], spec['mjd'], spec['ra'], spec['dec'], max_images, show, zoom_only)
    log.info(f"Can't track {spec.get('name', 'this object')}: {spec.get('reason', 'not supported')}")
    return None


def run_settings(settings, max_images=None, show=True, zoom_only=cfg.ZOOM_ONLY):
    """Re-run a target from the settings stored with its results (the known-objects list)."""
    mode = settings.get('mode')
    max_images = max_images if max_images is not None else settings.get('max_images')
    if mode == 'asteroid':
        return run_asteroid(settings['asteroid'], tuple(settings['window']), max_images, show, zoom_only)
    if mode == 'named star':
        return run_named_star(settings['star'], max_images, show, zoom_only)
    if mode == 'coordinates':
        coords = settings['coords']
        return run_coords(tuple(coords) if isinstance(coords, list) else coords, max_images, show, zoom_only)
    if mode == 'catalog star':
        return run_catalog_star(settings['star'], max_images, show, zoom_only)
    if mode == 'path':
        return run_path(settings['name'], settings['mjd'], settings['ra'], settings['dec'], max_images, show, zoom_only)
    if mode == 'star':
        return run_star(float(settings['ra_deg']), float(settings['dec_deg']), max_images, show, zoom_only)
    raise ValueError(f'Unknown kind of run: {mode}')


def interactive(max_images=None):
    """The start window (launcher.py): pick a known object or enter coordinates; then the analysis and its
    windows. Objects picked in the viewer (press T) are tracked next; closing all windows returns here."""
    from .launcher import choose_target
    from .plots import take_track_request
    while True:
        choice = choose_target()
        if choice is None:
            return
        try:
            run_settings(choice, max_images)
            spec = take_track_request()
            while spec:
                run_track(spec, max_images)
                spec = take_track_request()
        except Exception as e:                              # back to the start window, not a crash
            log.info(f'\nThat run stopped: {type(e).__name__}: {e}\n')


def cli(argv=None):
    """Command line: python -m Jonah_Code_File [--coords "RA Dec" | --star NAME | --asteroid ID] ...
    With no options: the start window (pick a known object, or enter coordinates). --config: config.json's target."""
    parser = argparse.ArgumentParser(prog='python -m Jonah_Code_File',
                                     description='SPHEREx transient / moving-object analysis (settings: config.json)')
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--coords', help='point at exactly this sky position, e.g. "08h55m10.8s -07d14m42s" or "133.795 -7.245"')
    target.add_argument('--star', help="a star or brown dwarf by name (SIMBAD), e.g. \"Barnard's Star\"")
    target.add_argument('--asteroid', help="an asteroid or comet (JPL Horizons id), e.g. 433")
    parser.add_argument('--window', nargs=2, metavar=('START', 'END'), help='asteroid search window, e.g. 2026-02-15 2026-06-15')
    parser.add_argument('--max-images', type=int, help='only the first N images (quick test)')
    target.add_argument('--config', action='store_true', help="the target set in config.json (no start window)")
    parser.add_argument('--no-show', action='store_true', help="don't open any windows (results are still saved)")
    args = parser.parse_args(argv)

    if not (args.coords or args.star or args.asteroid or args.config or args.no_show):
        return interactive(args.max_images)
    options = dict(max_images=args.max_images if args.max_images else cfg.MAX_IMAGES, show=not args.no_show,
                   window=tuple(args.window) if args.window else cfg.ASTEROID_WINDOW)
    results = run(asteroid=args.asteroid, star=args.star, coords=args.coords, **options)   # none given: config.json's
    if not args.no_show:                                    # an object picked in the viewer (T): track it next
        from .plots import take_track_request
        spec = take_track_request()
        while spec:
            results = run_track(spec, options['max_images'])
            spec = take_track_request()
    return results


if __name__ == '__main__':
    cli()
