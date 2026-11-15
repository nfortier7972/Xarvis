"""All settings in one place."""

from astropy import units as u

## WHAT TO TRACK. An asteroid (moves a lot - the transient case), or a star (stays put).
# JPL Horizons id of the asteroid: a number ('433' = Eros) or a name. Set to None to track the star below instead.
ASTEROID = '433'
# Time window to search, as dates (SPHEREx data runs 2025-04-29 to about 2026-06-14). SPHEREx always looks ~90 degrees
# away from the Sun, so an asteroid is only in its images while it is near that solar elongation (82-98 degrees).
ASTEROID_WINDOW = ('2026-02-15', '2026-06-15')      # Eros: ~125 images over 4 months, crossing ~100 degrees of sky
GRID_SPACING_DEG = 1.5              # image search: one position search every this many degrees along the path
DETECTION_SNR = 5                   # the asteroid counts as detected in a frame above this signal-to-noise
RESIDUAL_LIMIT_PX = 1.5             # flag a detection this far (pixels) from where the ephemeris says it should be

## a star or brown dwarf BY NAME (used when ASTEROID is None): anything SIMBAD knows, e.g. "Barnard's Star",
# 'R Dra', 'WISE J085510.83-071442.5'. Its proper motion and parallax are included (see stars.py).
STAR = None

## ANY point in the sky (used when ASTEROID and STAR are None): the software looks exactly there - no snapping.
# Either a string ('08h55m10.831s -07d14m42.53s', '08:55:10.8 -07:14:42', '133.795 -7.245') or (ra_deg, dec_deg).
# The spot is treated as fixed on the sky, so anything there that moves or changes gets flagged. Also from the
# command line: python -m xarvis_pipeline --coords "08h55m10.8s -07d14m42s"
COORDS = None

## a star by coordinates (used when ASTEROID, STAR and COORDS are None): snapped to the nearest bright, isolated
## 2MASS star (see target.py). Note: no proper motion - use STAR for nearby fast-moving stars.
RA_DEG = 304.693508808
DEC_DEG = 42.4436872991

SNAP_RADIUS = 60 * u.arcsec         # look for a catalog star within this distance of RA/DEC
SNAP_MAX_KS_MAG = 13.0              # only stars at least this bright (2MASS Ks magnitude)
ISOLATION_RADIUS = 18 * u.arcsec    # ~3 SPHEREx pixels: neighbors this close blend into the red circle...
ISOLATION_MAX_FRACTION = 0.15       # ...and together must be under 15% of the star's brightness

## dataset
SPREAD_IMAGES = False               # with MAX_IMAGES: True = spread over the whole time span (website), False = the first ones
MAX_IMAGES = None                   # None = every SPHEREx image of the object; e.g. 40 for a quick test
ZOOM_ONLY = True                    # after the analysis keep only the zoomed pixels in memory (frees the rest)
PRETTY_IMAGES = True                # after the analysis, show frames in colour, smoothed and upscaled (display only)
DISPLAY_UPSCALE = 8                 # ...this many times the original resolution (50x50 -> 400x400)
DISPLAY_SHARPEN = 0.6               # extra crispness (unsharp masking) on top of the sharp (Lanczos) upscaling; 0 = none
TARGET_DISPLAY_LEVEL = 0.7          # the target is drawn at this brightness (0-1) in every frame, so frames compare
WIDE_VIEW_SIZE = 30                 # arcmin: the wide view (press W in the viewer) - downloaded on demand, cached
WIDE_VIEW_UPSCALE = 3               # ...shown at this many times its resolution (~290 px -> ~880 px)
LIVE_VIEW = True                    # show the data in a window while it downloads (only when the plots are shown)
LIVE_REFRESH_SECONDS = 1.0          # redraw it at most this often, so drawing doesn't slow the processing down

## disk cache: makes re-runs fast (no re-downloading), and cleans itself up after every run (see cache.py)
CACHE_ENABLED = True                # False = never write anything to disk
CACHE_DIR = None                    # None = a local folder OneDrive doesn't sync (%LOCALAPPDATA%\xarvis_pipeline\cache)
CACHE_MAX_AGE_DAYS = 7              # files not used for this long are deleted...
CACHE_MAX_MB = 500                  # ...and the least recently used ones above this size (~95 KB per image)

## results: the measurements of every frame, saved as a small table (one file per target, replaced each run)
SAVE_RESULTS = True
RESULTS_DIR = None                  # None = a 'results' folder next to xarvis_pipeline
ZOOM_SIZE_PIX = 50                  # zoom: 50 x 50 ORIGINAL pixels centered on the object
CUTOUT_SIZE = 5.5 * u.arcmin        # downloaded cutout: ~54 x 54 pixels, a little bigger than the zoom

## measurements
APERTURE_RADIUS_PIX = 2.0           # brightness is summed inside this circle...
ANNULUS_PIX = (4.0, 6.5)            # ...minus the background measured in this ring
SEARCH_RADIUS_PIX = 3               # look this far from the expected position for the object's peak...
TRACK_TOLERANCE_PIX = 1.5           # ...but only trust a peak this close (9"; PSF ~7.5"): further = something else
TRACK_MIN_SIGMA = 4                 # a peak counts only this far above the sky (3x3-smoothed, bad pixels removed)
ALIGNMENT_TOLERANCE_PIX = 0.25      # alignment check warns about frames above this
MOVEMENT_SIGMA = 5                  # flag movement bigger than this many times the typical scatter
BRIGHTNESS_SIGMA = 5                # flag brightness this many times off the object's own spectrum
SPECTRUM_NEIGHBORS = 10             # brightness reference: median of up to this many nearest-wavelength frames...
SPECTRUM_WINDOW = 0.03              # ...within +/-3% of the frame's wavelength...
SPECTRUM_MIN_NEIGHBORS = 4          # ...and at least this many, otherwise the frame's brightness isn't judged

## comparison with Gaia DR3 / CatWISE and JPL's known asteroids (field.py)
FIELD_CHECK = True                  # compare every frame's sources with the catalogs (stars / coordinates)
FIELD_RADIUS = 4 * u.arcmin         # catalog search radius around the target (covers the 5.5' cutout)
FIELD_GAIA_MAX_RP = 18.0            # Gaia sources down to this RP magnitude (fainter ones SPHEREx can't see)...
FIELD_W1_MAX = 16.0                 # ...and CatWISE down to this W1 (red / dusty sources Gaia misses)
FIELD_MATCH_PIX = 1.5               # a SPHEREx peak this close to a catalog source is that source
FIELD_MIN_SIGMA = 5                 # unmatched peaks reported only above this significance
SB_IDENT_VMAG_LIMIT = 21            # JPL SB Ident: known asteroids/comets down to this visual magnitude
SB_IDENT_MAX_CALLS = 4              # look up at most this many unexplained sources per run (~20 s each, in parallel)
SB_IDENT_MATCH_ARCSEC = 15          # a known object this close to an unexplained source explains it

## sky map (skymap.py): name every object in the image, find the new ones. Always done for coordinates.
SKYMAP_FOR_ALL_TARGETS = False      # True: also for named stars / snapped stars (adds ~1-4 min of JPL lookups)
SKYMAP_MAX_SB_CALLS = 1             # JPL SB Ident lookups per run (~20 s each): one, for the burst of observations with
                                    # the most unexplained detections; if it finds nothing, move on. J in the viewer
                                    # checks any other frame. (JPL allows 20 per 5 minutes.)
SKYMAP_BURST_DAYS = 3.0             # one lookup covers observations up to this many days apart (bodies' rates do the rest)
SKYMAP_MAX_RATE_DEG_PER_DAY = 1.0   # asteroid search widened by this motion; new movers linked up to 4x this

## bad pixels (ignored in all measurements)
# SPHEREx FLAGS that make a pixel unreliable. Not included: SOURCE ("a known star is here"), FULLSAMPLE,
# DICHROIC, PHANMISS, and HALO (it marks the surroundings of ordinary stars too, so it isn't a defect).
# TRANSIENT here means a jump during detector readout (usually a cosmic ray).
BAD_FLAG_NAMES = {'TRANSIENT', 'OVERFLOW', 'SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 'COLD', 'NONLINEAR',
                  'PERSIST', 'OUTLIER', 'CROSSTALK', 'GHOST', 'GHOST_FPA', 'GHOST_EXT', 'STREAK', 'BLOOM',
                  'SNOWBALL'}
# For a MOVING object these two are not defects: SPHEREx's own pipeline marks a moving source's brightest pixels
# as OUTLIER / TRANSIENT, because it changes during the exposure the way a cosmic-ray hit does. (Measured on Eros:
# 103 of its pixels in 16 of 28 frames were flagged OUTLIER.) Throwing them away throws away the asteroid itself.
BAD_FLAG_NAMES_MOVING = BAD_FLAG_NAMES - {'OUTLIER', 'TRANSIENT'}
# A star too bright for SPHEREx (e.g. Barnard's Star, K = 4.5) saturates: its core pixels are flagged with these
# (measured: the whole 5x5 core BLOOM / SUR_ERROR / TRANSIENT, OVERFLOW or NaN at the centre, values capped).
# Its brightness can't be measured then, but its POSITION can, from the unsaturated outer glow (see fit_wings).
SATURATION_FLAG_NAMES = {'OVERFLOW', 'SUR_ERROR', 'BLOOM'}
WING_RADII_PIX = (2.0, 7.0)         # saturated star: fit its glow between these distances from the centre...
WING_OUTER_MAX_PIX = 11.0           # ...or out to this, when very bright stars leave too few good pixels
# Saturated star's RELATIVE brightness: median of the glow in this ring, minus the sky further out (glow_brightness).
GLOW_RING_PIX = (5.0, 10.0)
GLOW_SKY_PIX = (13.0, 18.0)
GLOW_PRECISION = 0.09               # its per-frame scatter, measured on Barnard's Star (a steady star): ~9%
# Spikes the pipeline missed (e.g. an unflagged hot pixel): a real source always spreads over several
# pixels, so a single pixel this many times brighter than ALL its neighbors, and far above the noise, isn't one.
# Measured: real star centers reach ~7x their brightest neighbor; the hot pixels found were 48x and 95x.
SPIKE_NEIGHBOR_RATIO = 20
SPIKE_MIN_SIGMA = 20

## IRSA access (IRSA often answers "502 Proxy Error" and then succeeds on a retry)
MAX_ATTEMPTS = 50
WAIT_SECONDS = 0.001                # wait between the first FAST_ATTEMPTS retries...
FAST_ATTEMPTS = 10
MAX_WAIT_SECONDS = 5                # ...then back off (0.1 s, 0.2 s, 0.4 s ... up to this) for longer outages
REQUEST_TIMEOUT = 30                # seconds to wait for IRSA to start answering before retrying
MAX_CONCURRENT_DOWNLOADS = 16       # parallel cutout downloads (measured: 16 is fastest; 24 makes IRSA refuse more)
COLLECTIONS = ['spherex_dr1', 'spherex_qr3']
GATOR_URL = 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query'
SIA_URL = 'https://irsa.ipac.caltech.edu/SIA'

PIXEL_ARCSEC = 6.15                 # SPHEREx pixel size
