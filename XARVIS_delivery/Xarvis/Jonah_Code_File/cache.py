"""A small disk cache for downloads, so a re-run doesn't download the same data again - and cleans itself up.

What is stored: the trimmed cutouts (only the extensions we use, ~95 KB each instead of the ~4 MB IRSA sends)
and IRSA's image-search replies, compressed. Where: a local folder that OneDrive does not sync
(%LOCALAPPDATA%\\Jonah_Code_File\\cache on Windows). Clean-up, after every run (even one that crashes):
  - anything not used for CACHE_MAX_AGE_DAYS is deleted,
  - then the least recently used files are deleted until the cache is under CACHE_MAX_MB.
Set CACHE_ENABLED to false in config.json to never write anything; clear_cache() deletes it all.
"""

import logging
import hashlib
import os
import shutil
import threading
import time
import zlib
from pathlib import Path

from . import config as cfg

log = logging.getLogger(__name__)


def cache_dir():
    """The cache folder: CACHE_DIR if set, otherwise a local folder outside OneDrive."""
    if cfg.CACHE_DIR:
        return Path(cfg.CACHE_DIR)
    base = os.environ.get('LOCALAPPDATA') or (Path.home() / '.cache')
    return Path(base) / 'Jonah_Code_File' / 'cache'


def make_key(*parts):
    """A file name for a request: a hash of everything that identifies it."""
    return hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()


def _path(kind, key):
    return cache_dir() / kind / f'{key}.bin'


def get(kind, key):
    """The cached bytes, or None. Marks the file as recently used."""
    if not cfg.CACHE_ENABLED:
        return None
    path = _path(kind, key)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        os.utime(path)                                      # recently used: kept longest by prune()
        return zlib.decompress(data)
    except (OSError, zlib.error):                           # damaged file (e.g. power cut mid-write): drop it
        path.unlink(missing_ok=True)
        return None


def put(kind, key, data):
    """Store bytes (compressed). Written to a temporary file first, so a crash never leaves half a file."""
    if not cfg.CACHE_ENABLED:
        return
    path = _path(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_bytes(zlib.compress(data, 1))
    os.replace(tmp, path)


def prune(quiet=False):
    """Delete files unused for CACHE_MAX_AGE_DAYS, then the least recently used ones until the cache is under
    CACHE_MAX_MB. Also removes leftovers of interrupted writes."""
    root = cache_dir()
    if not root.exists():
        return
    now = time.time()
    for tmp in root.rglob('*.tmp'):
        if now - tmp.stat().st_mtime > 3600:
            tmp.unlink(missing_ok=True)

    files = sorted(((p.stat().st_mtime, p.stat().st_size, p) for p in root.rglob('*.bin')), key=lambda f: f[0])
    total, deleted = sum(size for _, size, _ in files), 0
    for used, size, path in files:                          # oldest use first
        if now - used <= cfg.CACHE_MAX_AGE_DAYS * 86400 and total <= cfg.CACHE_MAX_MB * 1e6:
            break
        try:
            path.unlink()
            total -= size
            deleted += 1
        except OSError:                                     # in use by another program right now: skip it
            pass
    if not quiet:
        log.info(f'Cache: {len(files) - deleted} files, {total / 1e6:.1f} MB in {root}'
              + (f' ({deleted} old files deleted)' if deleted else ''))


def clear_cache():
    """Delete the whole cache."""
    shutil.rmtree(cache_dir(), ignore_errors=True)
    log.info(f'Cache cleared: {cache_dir()}')
