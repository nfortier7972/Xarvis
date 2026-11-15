"""Save / load the measurements of a run as a small table (one row per frame, ~100-200 KB per target).

Only measurements are saved - no pixels (those are in the disk cache while they're still useful, so re-running
the analysis is fast). One file per target, replaced on every run, so results never pile up.
Open it with load_results(), or with any program that reads CSV-like text (it's an ECSV file).
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.table import Table

from . import config as cfg

# per-frame measurements worth keeping (only the ones a frame actually has are saved)
FRAME_COLUMNS = ['obs_id', 'date_obs', 'mjd', 'detector', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr',
                 'n_bad_in_aperture', 'object_ra', 'object_dec', 'object_err_arcsec', 'expected_ra', 'expected_dec',
                 'alignment_px', 'rotation_deg', 'saturated', 'position_method', 'glow', 'track', 'track_sep_px',
                 'astrometry_ok', 'psf_fwhm_arcsec', 'moving_flags', 'field_unmatched']


def results_dir():
    return Path(cfg.RESULTS_DIR) if cfg.RESULTS_DIR else Path(__file__).resolve().parent.parent / 'results'


def results_path(name):
    return results_dir() / (re.sub(r'[^A-Za-z0-9_+-]+', '_', name).strip('_') + '.ecsv')


def save_results(results, settings):
    """Write one row per frame: the frame's measurements plus its movement / brightness results.
    `settings` = what was run (target, window ...), stored in the file's header. Returns the path."""
    frames, n = results['frames'], len(results['frames'])
    table = Table()
    table['T'] = np.arange(1, n + 1)
    for column in FRAME_COLUMNS:
        if any(column in f for f in frames):
            values = [f.get(column, np.nan) for f in frames]
            if any(v is None or isinstance(v, str) for v in values):          # e.g. position_method may be None
                values = [str(v) for v in values]
            table[column] = values
    for prefix in ('movement', 'brightness'):
        for name, values in (results.get(prefix) or {}).items():
            if isinstance(values, np.ndarray) and values.shape == (n,):    # per-frame arrays only
                table[name if prefix == 'movement' and name != 'flagged' else f'{prefix}_{name}'] = values

    name = results['tracked']['name']
    table.meta.update(target=name, saved=datetime.now().isoformat(timespec='seconds'),
                      settings_json=json.dumps(_plain(settings)), **{k: str(v) for k, v in settings.items()})
    path = results_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format='ascii.ecsv', overwrite=True)
    print(f'Results saved: {path} ({path.stat().st_size / 1e3:.0f} KB, {n} frames)')
    sky = results.get('skymap')
    if sky:                                                 # the new objects' detections: coordinates + UTC times
        from .skymap import new_objects_table
        new = new_objects_table(sky['new_objects'], frames)
        new_path = path.with_name(path.stem + '_new_objects.csv')
        new.write(new_path, format='ascii.csv', overwrite=True)
        print(f'New objects saved: {new_path} ({len(sky["new_objects"])} objects, {len(new)} detections '
              f'with coordinates and UTC times)')
    register(name, settings, n)
    return path


# ----------------------------------------------------------------------------
# Known objects: every target analysed so far (the start window's list)
# ----------------------------------------------------------------------------

# Kept next to the disk cache but NOT in it (cache cleanup never deletes it). If the results folder is deleted, the
# results can be regenerated from it (re-running is quick while the images are still in the cache).
DEFAULT_OBJECTS = [                                         # what is offered before anything has been run
    ('433 Eros', {'mode': 'asteroid', 'asteroid': '433', 'window': ['2026-02-15', '2026-06-15']}),
    ("Barnard's Star", {'mode': 'named star', 'star': "Barnard's Star"}),
    ('Epsilon Indi B', {'mode': 'named star', 'star': 'eps Indi B'}),
    ('WISE 0855-0714', {'mode': 'named star', 'star': 'WISE J085510.83-071442.5'}),
    ('RW Aurigae', {'mode': 'named star', 'star': 'RW Aur'}),
    ('R Draconis', {'mode': 'named star', 'star': 'R Dra'}),
]


def _plain(value):
    """Settings as plain JSON types (tuples -> lists, numpy numbers -> float/int)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def registry_path():
    from .cache import cache_dir
    return cache_dir().parent / 'known_objects.json'


def _read_registry():
    try:
        return json.loads(registry_path().read_text())
    except (OSError, ValueError):
        return {}


def register(name, settings, n_frames=None):
    """Remember a target and how to re-run it."""
    registry = _read_registry()
    registry[name] = {'settings': _plain(settings), 'frames': n_frames,
                      'saved': datetime.now().isoformat(timespec='seconds')}
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(registry, indent=1))
    tmp.replace(path)


def known_objects():
    """Every known target, for the start window: [{'name', 'settings', 'frames', 'saved', 'has_results'}], from the
    results folder (each file stores how it was made) plus the registry. Most recent first."""
    objects = {}
    for name, entry in _read_registry().items():
        objects[name] = {'name': name, 'settings': entry['settings'], 'frames': entry.get('frames'),
                         'saved': entry.get('saved', ''), 'has_results': results_path(name).exists()}
    folder = results_dir()
    for path in sorted(folder.glob('*.ecsv')) if folder.exists() else []:
        try:
            table = Table.read(path, format='ascii.ecsv')
            meta = table.meta
        except Exception:
            continue
        name = meta.get('target', path.stem)
        settings = json.loads(meta['settings_json']) if 'settings_json' in meta else _settings_from_meta(meta)
        if settings is None:
            continue
        frames = len(table)
        objects[name] = {'name': name, 'settings': settings, 'frames': frames, 'saved': meta.get('saved', ''),
                         'has_results': True}
    return sorted(objects.values(), key=lambda o: o['saved'], reverse=True)


def _settings_from_meta(meta):
    """Settings of a results file saved before settings_json existed (values stored as text)."""
    import ast
    mode = meta.get('mode')
    if mode == 'asteroid':
        return {'mode': mode, 'asteroid': meta['asteroid'], 'window': list(ast.literal_eval(meta['window']))}
    if mode == 'named star':
        return {'mode': mode, 'star': meta['star']}
    if mode == 'coordinates':
        return {'mode': mode, 'coords': meta['coords']}
    if mode == 'star':
        return {'mode': mode, 'ra_deg': float(meta['ra_deg']), 'dec_deg': float(meta['dec_deg'])}
    return None


def rebuild_results(on_progress=None):
    """The results folder is missing (or empty): regenerate it by re-running every known target without windows
    (quick for targets whose images are still in the disk cache). Known targets = the registry; if that is empty
    too, DEFAULT_OBJECTS. on_progress(text) is called before each one."""
    from .main import run_settings
    entries = [(name, e['settings']) for name, e in _read_registry().items()] or DEFAULT_OBJECTS
    results_dir().mkdir(parents=True, exist_ok=True)
    for k, (name, settings) in enumerate(entries, 1):
        if on_progress:
            on_progress(f'Regenerating results {k}/{len(entries)}: {name}')
        print(f'\n=== Regenerating results {k}/{len(entries)}: {name} ===')
        try:
            run_settings(settings, show=False)
        except Exception as e:
            print(f'  could not regenerate {name}: {type(e).__name__}: {e}')


def load_results(name_or_path):
    """Load a saved results table, by target name (e.g. '433 Eros') or file path."""
    path = Path(name_or_path)
    return Table.read(path if path.suffix == '.ecsv' else results_path(str(name_or_path)), format='ascii.ecsv')
