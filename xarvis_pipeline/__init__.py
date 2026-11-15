"""SPHEREx transient analysis: every SPHEREx image of a moving asteroid (or a star), T1 ... Tn in time order.

Modules
  config.py   all settings (which asteroid or star, time window, ...)
  irsa.py     talking to IRSA: timeouts, retries, image search, catalogs, cutouts
  cache.py    disk cache of downloads (fast re-runs), cleaned up automatically after every run
  results.py  save / load each run's measurements as a small table (results/<target>.ecsv)
  moving.py   an asteroid: its path from JPL Horizons, and the SPHEREx images that caught it
  target.py   a star: pick the one to track and its catalog brightness
  frames.py   download + measure every image (placed by SPHEREx's pointing only); original-pixel zooms
  measure.py  brightness, position, movement, brightness changes, alignment and spectrum checks
  plots.py    frame viewer (slider), frame grid, asteroid motion, star time series, spectrum
  main.py     runs everything (run __main__.py, or python -m xarvis_pipeline)
"""

import warnings

from astropy import log
from astropy.wcs import FITSFixedWarning

# astropy prints a long, harmless note about SIP distortion for every SPHEREx header; hide it
log.setLevel('WARNING')
warnings.simplefilter('ignore', FITSFixedWarning)
