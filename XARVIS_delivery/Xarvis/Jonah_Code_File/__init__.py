"""SPHEREx transient analysis: every SPHEREx image of a moving asteroid, a star or any sky position, T1 ... Tn.

Modules
  config.py / load_config.py   all settings, read from the project-level config.json
  net.py          every web request: timeouts, retries, caching and each service's rate limits
  irsa.py         IRSA: SPHEREx image search, catalogs (Gator), cutouts (and the PSF at the target)
  cache.py        disk cache of downloads (fast re-runs), cleaned up automatically after every run
  results.py      each run's measurements (results/<target>.ecsv), known-objects registry, new-object table
  moving.py       an asteroid: its path from JPL Horizons, and the SPHEREx images that caught it
  target.py       a star by coordinates: snap to the nearest bright isolated 2MASS star
  stars.py        a star by name (SIMBAD): proper motion + parallax positions, contamination, catalog SED
  frames.py       download + measure every image (placed by SPHEREx's pointing only); original-pixel zooms
  measure.py      brightness, position, tracking, movement, brightness changes, alignment and spectrum checks
  psf.py          SPHEREx's own PSF: PSF-fit positions and aperture corrections
  field.py        field vs Gaia DR3 / CatWISE, JPL small-body lookups
  skymap.py       name every object in the image (Hipparcos, Gaia, CatWISE, planets, asteroids), find NEW ones
  visual.py       display-only colour / sharpening / upscaling
  plots.py        frame viewer, live view, time series, spectrum, motion plots
  launcher.py     the start window (known objects, coordinates)
  algorithm.py    apparent-motion fits from frames or saved results
  spectrumanalyser.py   102-channel spectral atlas and capture comparisons
  api.py          web-app entry points: analyze(...) -> JSON-ready result, frame images as PNG
  main.py         runs everything (python -m Jonah_Code_File, or --help)

Messages go through the standard `logging` module (logger "Jonah_Code_File"); by default they are printed
plainly to the terminal, as before. A web app can attach its own handler (see api.py).
"""

import logging
import sys
import warnings

from astropy import log
from astropy.wcs import FITSFixedWarning

# astropy prints a long, harmless note about SIP distortion for every SPHEREx header; hide it
log.setLevel('WARNING')
warnings.simplefilter('ignore', FITSFixedWarning)

logger = logging.getLogger(__name__)
if not logger.handlers:                      # plain terminal output, exactly like print() before
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
