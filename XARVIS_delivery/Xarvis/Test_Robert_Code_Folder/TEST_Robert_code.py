"""XARVIS test script: query IRSA for SPHEREx spectral images and inspect one.

Stand-alone (the website does not use it). In the team repo the same IRSA access, with retries and caching, lives in
the analysis pipeline's irsa.py.

Based on the IRSA "Introduction to SPHEREx Spectral Images" tutorial, with:
  - retry loops so a temporary IRSA outage (502 etc.) doesn't kill the script
  - the missing fits.open step restored (hdulist)
  - Firefly viewer made optional

Install:  pip install numpy matplotlib astropy astroquery firefly-client requests
"""
import http.client
import time
import urllib.error

import numpy as np
import matplotlib.pyplot as plt
import requests

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import conf

from astroquery.ipac.irsa import Irsa
from pyvo.dal.exceptions import DALServiceError

try:
    from firefly_client import FireflyClient
except ImportError:
    FireflyClient = None

# ----------------------------------------------------------------- settings ---
USE_FIREFLY = True        # set False to skip the interactive viewer
RETRY_INTERVAL = 1.0      # seconds between attempts. Please don't go much lower:
                          # IRSA is a shared public archive.
GIVE_UP_AFTER = None      # seconds before giving up, or None to retry forever

# Reading SPHEREx files can exceed astropy's default timeout.
conf.remote_timeout = 120

ra_deg = 304.693508808
dec_deg = 42.4436872991
coord = SkyCoord(ra_deg, dec_deg, unit="deg")
search_radius = 1 * u.arcsec

# Errors that mean "server not ready", not "your code is wrong".
RETRYABLE = (
    DALServiceError,
    requests.exceptions.RequestException,
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    ConnectionError,
)


def retry(label, func, interval=RETRY_INTERVAL, give_up_after=GIVE_UP_AFTER):
    """Call func() until it succeeds, pausing `interval` s between attempts."""
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[{label}] attempt {attempt}...")
            return func()
        except RETRYABLE as e:
            elapsed = time.time() - start
            print(f"  failed after {elapsed:.0f}s: {str(e)[:100]!r}")
            if give_up_after is not None and elapsed + interval > give_up_after:
                raise
            time.sleep(interval)


# -------------------------------------------------- 1. find spectral images ---
results = retry(
    "IRSA SIA query",
    lambda: Irsa.query_sia(pos=(coord, search_radius), collection=["spherex_dr1", "spherex_qr3"]),
)
print(f"Got {len(results)} spectral images")
print(results.colnames)

spectral_image_url = results["access_url"][0]
print(spectral_image_url)


# ----------------------------------------------- 2. download the FITS file ---
def load_fits():
    # memmap=False + touching every .data reads all extensions into memory,
    # so hdulist stays usable after the file is closed.
    with fits.open(spectral_image_url, memmap=False) as hdul:
        for hdu in hdul:
            hdu.data
        return hdul


hdulist = retry("FITS download", load_fits)
hdulist.info()

# ------------------------------------------ 3. optional Firefly visualization ---
if USE_FIREFLY and FireflyClient is not None:
    try:
        fc = FireflyClient.make_client(url="https://irsa.ipac.caltech.edu/irsaviewer")
        fc.reinit_viewer()  # prints a link: open it in your browser
        fc.show_fits_image(file_input=spectral_image_url, plot_id="spectral_image", Title="Spectral Image")
    except Exception as e:  # viewer is a nice-to-have; don't stop the script
        print(f"Firefly skipped: {e}")
elif USE_FIREFLY:
    print("firefly-client not installed (pip install firefly-client); skipping viewer.")

# ----------------------------------------------------- 4. header + spatial WCS ---
spectral_image_header = hdulist[1].header

for keyword in ["EXTNAME", "NAXIS1", "NAXIS2", "BUNIT", "DETECTOR", "OBSID", "DATE", "PSF_FWHM"]:
    print(f"{keyword}: {spectral_image_header.get(keyword, 'Keyword not found')}")

print(spectral_image_header["WCSNAME*"])

wcs = WCS(spectral_image_header)
print(wcs)

ra, dec = wcs.wcs.crval
x, y = wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg"))
print(ra, dec, x, y)

# ------------------------------------------------------------ 5. spectral WCS ---
# The wavelength lookup table lives in the WCS-WAVE extension, so pass the HDU list.
spectral_wcs = WCS(header=hdulist["IMAGE"].header, fobj=hdulist, key="W")
spectral_wcs.sip = None  # SIP distortion must be off for the spectral WCS

wl, bw = spectral_wcs.pixel_to_world(x, y)
print("Wavelength / bandwidth at image centre:", wl, bw)

spectral_image = hdulist["IMAGE"]
spectral_image_data = spectral_image.data

(yy, xx) = np.indices(spectral_image.shape)
wavelength, bandwidth = spectral_wcs.pixel_to_world(xx, yy)
print("Wavelength:\n", wavelength)
print("Bandwidth:\n", bandwidth)

# ----------------------------------------------------------------- 6. plot ---
plt.imshow(wavelength.value, origin="lower", cmap="gray")
plt.colorbar(label=f"Wavelength [{wavelength.unit}]")
plt.title(f"Wavelength - Detector {hdulist['IMAGE'].header['DETECTOR']}")
plt.show()

# Pixel-flag counts written by the processing pipeline
print(spectral_image_header["L2 N_*"])
