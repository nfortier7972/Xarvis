"""Pairwise diagnostics and interactive 102-channel SPHEREx visualization.

Standalone: python -m xarvis_pipeline.spectrumanalyser
Library: SpectrumAnalyzer.from_results(main.run(show=False)).show_bands()
Only this module needs replacing. No tracking or download logic is changed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from . import config as cfg
from .measure import reliable_frames, robust_scatter


class SpectrumAnalyzer:
    """Analyze and compare the spectral measurements of any two captures in a dataset.

    Initialized from `frames` (and optional `brightness` / `spectrum` dicts from `measure.py`),
    and optionally retained zooms and tracking results for the interactive viewer.
    """

    def __init__(self, frames, brightness=None, spectrum=None, *,
                 zooms=None, movement=None, tracked=None):
        self.frames = frames
        self.brightness = brightness
        self.spectrum = spectrum
        self.zooms = zooms
        self.movement = movement
        self.tracked = tracked

        self.reliable = reliable_frames(frames)
        self.wavelength = np.array([f['wavelength_um'] for f in frames], dtype=float)
        self.flux = np.array([f['flux_mjy'] for f in frames], dtype=float)
        self.flux_err = np.array([f['flux_err_mjy'] for f in frames], dtype=float)
        self.mjd = np.array([f['mjd'] for f in frames], dtype=float)
        self._obs_to_idx = {str(f['obs_id']): i for i, f in enumerate(frames)}

        # Precompute sorted reliable spectrum points for interpolation
        ok = (self.reliable & np.isfinite(self.wavelength) & np.isfinite(self.flux)
              & np.isfinite(self.flux_err) & (self.flux_err > 0))
        order = np.argsort(self.wavelength[ok])
        self.spec_wl = self.wavelength[ok][order]
        self.spec_flux = self.flux[ok][order]
        self.spec_err = self.flux_err[ok][order]

    @classmethod
    def from_results(cls, results):
        """Convenience constructor accepting the dict returned by `main.run()`."""
        return cls(
            frames=results['frames'],
            brightness=results.get('brightness'),
            spectrum=results.get('spectrum'),
            zooms=results.get('zooms'),
            movement=results.get('movement'),
            tracked=results.get('tracked'),
        )

    def band_spectrum(self, indices=None):
        """A 102-channel target aperture spectrum, retaining empty channels as NaN.

        Optional indices: 0-based frame indices or a boolean mask. No new downloads.
        Returns wavelength_um, flux_mjy, flux_err_mjy and count arrays of length 102.
        """
        selected = np.ones(len(self.frames), bool)
        if indices is not None:
            selected[:] = False
            ix = np.asarray(indices)
            if ix.dtype == bool:
                if ix.shape != selected.shape:
                    raise ValueError("Boolean mask must match frame count")
                selected = ix.copy()
            elif ix.size:
                if ix.ndim != 1 or not np.issubdtype(ix.dtype, np.integer):
                    raise TypeError("Use integer indices or a boolean mask")
                if np.any(ix < 0) or np.any(ix >= len(selected)):
                    raise IndexError("Frame indices out of range")
                selected[ix] = True
        detector = np.array([f['detector'] for f in self.frames])
        mean, error, count = binned_spectrum(self.wavelength, self.flux,
            self.flux_err, detector, selected & self.reliable)
        return dict(wavelength_um=CENTERS.ravel().copy(), flux_mjy=mean,
                    flux_err_mjy=error, count=count)

    def show_bands(self, zooms=None, movement=None, tracked=None, *, size=41):
        """Open linked channel gallery and capture/spectrum/tracking detail.

        Pass results['zooms'], results['movement'], results['tracked'] from run().
        Returns the viewer; call plt.show() to display its figures.
        """
        zooms = self.zooms if zooms is None else zooms
        movement = self.movement if movement is None else movement
        tracked = self.tracked if tracked is None else tracked
        if zooms is None or movement is None:
            raise ValueError('Use from_results(results), or pass zooms and movement explicitly.')
        return BandViewer(self, zooms, movement, tracked, size=size)

    def motion_fit(self, prediction_days=0.0, show_plot=False):
        """Optional apparent trajectory fit delegated to the new algorithm.py."""
        from .algorithm import orbital_prediction
        return orbital_prediction(self.frames, prediction_days=prediction_days,
                                  show_plot=show_plot)

    def spectrum_for_time(self, start_mjd, end_mjd):
        """Weighted target spectrum for an inclusive MJD interval."""
        if not np.isfinite([start_mjd, end_mjd]).all() or start_mjd > end_mjd:
            raise ValueError('Provide finite MJD bounds in increasing order.')
        return self.band_spectrum((self.mjd >= start_mjd) & (self.mjd <= end_mjd))

    def resolve_index(self, key):
        """Accept either an integer frame index (0-based) or an `obs_id` string."""
        if isinstance(key, (int, np.integer)):
            if not (0 <= key < len(self.frames)):
                raise IndexError(f'Frame index {key} out of range (0..{len(self.frames) - 1})')
            return int(key)
        if str(key) in self._obs_to_idx:
            return self._obs_to_idx[str(key)]
        raise KeyError(f'Unknown obs_id: {key}')

    def expected_flux_at(self, wavelength_um, exclude_idx=None):
        """Estimate expected flux (mJy) at `wavelength_um` using:
        1. Local median of reliable SPHEREx neighbors within +/- SPECTRUM_WINDOW (matching `brightness_changes`),
        2. Fallback to scaled 2MASS/AllWISE catalog SED interpolation if `spectrum` is available.
        Returns (expected_mjy, source_label).
        """
        # 1. Local SPHEREx neighbors (same logic as measure.brightness_changes)
        good = np.flatnonzero(self.reliable & np.isfinite(self.wavelength)
                              & np.isfinite(self.flux) & np.isfinite(self.flux_err)
                              & (self.flux_err > 0))
        if exclude_idx is not None:
            good = good[good != exclude_idx]

        if len(good):
            dist = np.abs(self.wavelength[good] - wavelength_um)
            candidates = good[dist <= cfg.SPECTRUM_WINDOW * wavelength_um]
            candidates = candidates[np.argsort(np.abs(self.wavelength[candidates] - wavelength_um))[:cfg.SPECTRUM_NEIGHBORS]]
            if len(candidates) >= cfg.SPECTRUM_MIN_NEIGHBORS:
                return float(np.median(self.flux[candidates])), 'spherex_neighbors'

        # 2. Fallback: log-log interpolation of scaled 2MASS/AllWISE catalog SED
        if self.spectrum and self.spectrum.get('catalog'):
            cat = sorted(self.spectrum['catalog'], key=lambda p: p[1])
            cat_wl = np.array([p[1] for p in cat], dtype=float)
            cat_flux = np.array([p[2] for p in cat], dtype=float)
            if cat_wl.min() <= wavelength_um <= cat_wl.max():
                scale = float(self.spectrum.get('scale', 1.0))
                interp_mjy = float(np.exp(np.interp(np.log(wavelength_um), np.log(cat_wl), np.log(cat_flux))) * scale)
                return interp_mjy, 'scaled_catalog_sed'

        return np.nan, 'unavailable'

    def frame_spectral_profile(self, idx):
        """Return spectral metrics for a single frame `idx`."""
        idx = self.resolve_index(idx)
        f = self.frames[idx]
        wl = float(f['wavelength_um'])
        flux = float(f['flux_mjy'])
        err = float(f['flux_err_mjy'])

        # Use precomputed reference from `brightness_changes` if available, else compute on the fly
        if self.brightness is not None and np.isfinite(self.brightness['reference_mjy'][idx]):
            ref_mjy = float(self.brightness['reference_mjy'][idx])
            ref_source = 'spherex_neighbors'
        else:
            ref_mjy, ref_source = self.expected_flux_at(wl, exclude_idx=idx)

        if np.isfinite(ref_mjy) and ref_mjy > 0:
            deviation = (flux / ref_mjy) - 1.0
            scatter = float(self.brightness['scatter']) if self.brightness is not None else robust_scatter(
                self.flux[self.reliable]
            ) / max(np.nanmedian(self.flux[self.reliable]), 1e-9)
            total_noise = float(np.hypot(err / ref_mjy, scatter))
            sigma_dev = float(deviation / total_noise) if total_noise > 0 else np.nan
        else:
            deviation, sigma_dev = np.nan, np.nan

        is_flagged = bool(self.brightness['flagged'][idx]) if self.brightness is not None else (
            np.isfinite(sigma_dev) and abs(sigma_dev) > cfg.BRIGHTNESS_SIGMA
        )
        align_px = f.get('alignment_px', np.nan)
        aligned_ok = bool(np.isnan(align_px) or align_px <= cfg.ALIGNMENT_TOLERANCE_PIX)

        return {
            'index': idx,
            'label': f'T{idx + 1}',
            'obs_id': str(f['obs_id']),
            'date_obs': f['date_obs'],
            'mjd': float(f['mjd']),
            'detector': int(f['detector']),
            'wavelength_um': wl,
            'flux_mjy': flux,
            'flux_err_mjy': err,
            'reference_mjy': ref_mjy if np.isfinite(ref_mjy) else None,
            'reference_source': ref_source,
            'pct_deviation': round(float(deviation * 100), 2) if np.isfinite(deviation) else None,
            'sigma_deviation': round(float(sigma_dev), 2) if np.isfinite(sigma_dev) else None,
            'is_reliable': bool(self.reliable[idx]),
            'n_bad_in_aperture': int(f['n_bad_in_aperture']),
            'is_aligned': aligned_ok,
            'is_brightness_flagged': is_flagged,
        }

    def compare(self, idx_a, idx_b):
        """Compare two captures `idx_a` and `idx_b` (by integer index or `obs_id`).

        Separates apparent brightness changes caused by observing at two different wavelengths
        from genuine astrophysical variability or detector artifacts.
        """
        a = self.frame_spectral_profile(idx_a)
        b = self.frame_spectral_profile(idx_b)

        delta_days = b['mjd'] - a['mjd']
        delta_wl = b['wavelength_um'] - a['wavelength_um']
        rel_wl_diff = abs(delta_wl) / max(a['wavelength_um'], 1e-9)
        same_wavelength_band = bool(rel_wl_diff <= cfg.SPECTRUM_WINDOW)

        raw_ratio = (b['flux_mjy'] / a['flux_mjy']) if a['flux_mjy'] > 0 else np.nan
        raw_diff_mjy = b['flux_mjy'] - a['flux_mjy']
        raw_diff_err = float(np.hypot(a['flux_err_mjy'], b['flux_err_mjy']))

        # Expected flux ratio purely from the object's baseline spectrum shape
        if (a['reference_mjy'] is not None and a['reference_mjy'] > 0
                and b['reference_mjy'] is not None and b['reference_mjy'] > 0):
            expected_ratio = b['reference_mjy'] / a['reference_mjy']
            # Excess ratio = 1.0 means the brightness difference is 100% explained by wavelength
            spectral_excess_ratio = raw_ratio / expected_ratio
        else:
            expected_ratio = np.nan
            spectral_excess_ratio = np.nan

        # Direct statistical significance when in the same wavelength window
        same_band_sigma = (
            abs(raw_diff_mjy) / raw_diff_err if (same_wavelength_band and a['flux_err_mjy'] > 0 and b['flux_err_mjy'] > 0 and raw_diff_err > 0 and a['index'] != b['index']) else np.nan
        )

        status, summary = self._classify_pair(a, b, same_wavelength_band, same_band_sigma, spectral_excess_ratio)

        return {
            'capture_a': a,
            'capture_b': b,
            'comparison': {
                'delta_days': round(float(delta_days), 4),
                'delta_wavelength_um': round(float(delta_wl), 4),
                'same_wavelength_band': same_wavelength_band,
                'raw_flux_diff_mjy': round(float(raw_diff_mjy), 3),
                'raw_flux_ratio': round(float(raw_ratio), 3) if np.isfinite(raw_ratio) else None,
                'expected_spectral_ratio': round(float(expected_ratio), 3) if np.isfinite(expected_ratio) else None,
                'spectral_excess_ratio': round(float(spectral_excess_ratio), 3) if np.isfinite(spectral_excess_ratio) else None,
                'same_band_sigma': round(float(same_band_sigma), 2) if np.isfinite(same_band_sigma) else None,
                'status_code': status,
                'plain_english_summary': summary,
            },
        }

    def find_comparison_partner(self, idx, prefer_normal=True):
        """Find the best second capture to compare against `idx`:
        picks the capture closest in wavelength from a different timestamp (preferring unflagged, reliable frames).
        """
        idx = self.resolve_index(idx)
        target_wl = self.wavelength[idx]
        candidates = [i for i in range(len(self.frames)) if i != idx and self.reliable[i] and self.mjd[i] != self.mjd[idx]
                      and np.isfinite(self.wavelength[i]) and np.isfinite(self.flux[i])
                      and np.isfinite(self.flux_err[i]) and self.flux_err[i] > 0]
        if prefer_normal and self.brightness is not None:
            normal = [i for i in candidates if not self.brightness['flagged'][i]]
            if normal:
                candidates = normal
        if not candidates:
            return None
        return int(min(candidates, key=lambda i: abs(self.wavelength[i] - target_wl)))

    def _classify_pair(self, a, b, same_band, same_band_sigma, excess_ratio):
        if not a['is_reliable'] or not b['is_reliable']:
            bad_labels = [x['label'] for x in (a, b) if not x['is_reliable']]
            return (
                'detector_glitch',
                f"{' and '.join(bad_labels)} had a bad pixel or cosmic-ray spike inside the measurement circle.",
            )

        if not a['is_aligned'] or not b['is_aligned']:
            return (
                'alignment_warning',
                'Telescope pointing offset exceeded tolerance in one capture; brightness comparison may be affected.',
            )

        if a['is_brightness_flagged'] or b['is_brightness_flagged'] or (
            np.isfinite(same_band_sigma) and same_band_sigma >= cfg.BRIGHTNESS_SIGMA
        ):
            return (
                'real_variability',
                f"Candidate significant brightness variation involving {a['label']} ({a['flux_mjy']:.2f} mJy) and "
                f"{b['label']} ({b['flux_mjy']:.2f} mJy); inspect the observations before attributing a cause.",
            )

        if not same_band:
            dir_word = 'brighter' if b['flux_mjy'] > a['flux_mjy'] else 'dimmer'
            return (
                'wavelength_shift',
                f"{b['label']} appears {dir_word} than {a['label']} ({b['flux_mjy']:.2f} vs. {a['flux_mjy']:.2f} mJy) "
                f"while SPHEREx observed at different wavelengths ({b['wavelength_um']:.2f} µm vs. {a['wavelength_um']:.2f} µm), "
                f"the wavelength difference may contribute; variability is not ruled out.",
            )

        return (
            'consistent',
            f"{a['label']} and {b['label']} were observed at matching wavelengths (~{a['wavelength_um']:.2f} µm) "
            f"and show consistent brightness.",
        )


def compare_captures(results, idx_a, idx_b):
    """Functional shortcut to compare two frames directly from `main.run()` output."""
    return SpectrumAnalyzer.from_results(results).compare(idx_a, idx_b)


# SPHEREx Explanatory Supplement v2.1, Tables 29–34 (2026-11-10).
# Rounded median wavelengths, not the geometric detector channel map.
CENTERS = np.array([
[.753,.771,.790,.809,.828,.848,.868,.889,.910,.932,.955,.978,1.001,1.025,1.050,1.075,1.102],
[1.113,1.139,1.167,1.195,1.224,1.254,1.284,1.315,1.346,1.378,1.412,1.446,1.481,1.517,1.554,1.592,1.631],
[1.656,1.695,1.735,1.775,1.818,1.861,1.905,1.949,1.995,2.041,2.089,2.139,2.189,2.240,2.292,2.345,2.399],
[2.445,2.506,2.575,2.646,2.719,2.794,2.871,2.951,3.033,3.118,3.204,3.293,3.383,3.476,3.570,3.667,3.768],
[3.834,3.867,3.901,3.936,3.970,4.005,4.041,4.076,4.112,4.148,4.184,4.221,4.257,4.294,4.331,4.368,4.405],
[4.435,4.468,4.500,4.533,4.567,4.601,4.635,4.669,4.704,4.739,4.774,4.808,4.843,4.879,4.915,4.952,4.989]])
LIMITS = np.array([[.743,1.116],[1.099,1.651],[1.636,2.421],[2.422,3.827],[3.815,4.427],[4.418,5.009]])


def channel_index(detector, wavelength):
    """Nearest nominal wavelength within detector's nominal range; -1 otherwise.
    This is a wavelength-based approximation, not official geometric assignment."""
    wl = np.asarray(wavelength, dtype=float)
    if not 1 <= detector <= 6:
        return np.full(wl.shape, -1, dtype=int)
    c = CENTERS[detector-1]
    i = np.searchsorted((c[:-1]+c[1:])/2, wl) + 17*(detector-1)
    lo, hi = LIMITS[detector-1]
    return np.where(np.isfinite(wl) & (wl >= lo) & (wl <= hi), i, -1)


def binned_spectrum(wl, flux, err, detectors, selected):
    """102 weighted estimates; retain NaN for missing channels and negative fluxes."""
    wl, flux, err, detectors, selected = map(np.asarray, (wl, flux, err, detectors, selected))
    channel = np.full(wl.shape, -1, dtype=int)
    for d in range(1,7):
        m = detectors == d
        channel[m] = channel_index(d, wl[m])
    good = selected.astype(bool) & (channel >= 0) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    k = channel[good]
    w = 1/err[good]**2
    n = np.bincount(k, minlength=102)
    sw = np.bincount(k, weights=w, minlength=102)
    sf = np.bincount(k, weights=w*flux[good], minlength=102)
    avg = np.divide(sf, sw, out=np.full(102,np.nan), where=sw>0)
    var = np.divide(1, sw, out=np.full(102,np.nan), where=sw>0)
    return avg, np.sqrt(var), n


def tangent_sky(ra, dec, east, north):
    """Inverse TAN projection for offsets in arcsec; handles RA wrap and poles."""
    a, b = np.deg2rad([ra, dec])
    x, y = np.deg2rad(east/3600), np.deg2rad(north/3600)
    den = np.cos(b)-y*np.sin(b)
    return np.rad2deg(a+np.arctan2(x,den))%360, np.rad2deg(np.arctan2(np.sin(b)+y*np.cos(b),np.hypot(den,x)))


def sample_frame(frame, east, north):
    """Bilinear image/variance sampling on a target-centered TAN grid.
    Reject pixels with incomplete valid support; propagate squared weights.
    """
    wcs = frame['wcs']
    ra, dec = wcs.pixel_to_world_values(*frame['catalog_xy'])
    ra, dec = tangent_sky(ra, dec, east, north)
    x,y = map(np.asarray,wcs.world_to_pixel_values(ra,dec))
    finite = np.isfinite(x)&np.isfinite(y)
    x,y = np.where(finite,x,0),np.where(finite,y,0)
    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
    dx,dy = x-ix,y-iy
    im, var = np.asarray(frame['image']),np.asarray(frame['variance'])
    valid = ~np.asarray(frame['bad'],bool)&np.isfinite(im)&np.isfinite(var)&(var>0)
    # Remove a scalar sky level for point-source detection, not diffuse photometry.
    yy,xx = np.indices(im.shape)
    cx,cy = frame['catalog_xy']
    bg = valid & (np.hypot(xx-cx,yy-cy)>8)
    sky = np.median(im[bg]) if bg.any() else 0.
    ny,nx = im.shape
    z,v,support = np.zeros(x.shape),np.zeros(x.shape),np.zeros(x.shape)
    for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        j,k=ix+ox,iy+oy
        inside=finite&(j>=0)&(j<nx)&(k>=0)&(k<ny)
        j,k=np.clip(j,0,nx-1),np.clip(k,0,ny-1)
        good=inside&valid[k,j]
        weight=np.where(good,w,0.)
        z+=weight*np.where(good,im[k,j]-sky,0.)
        v+=weight**2*np.where(good,var[k,j],0.)
        support+=weight
    good=(support>1-1e-7)&(v>0)
    z,v=np.where(good,z,np.nan),np.where(good,v,np.nan)
    return z,v


def weighted_image(images, variances, selection=None):
    """Inverse-variance mean, diagonal variance, and coverage. No data => NaN."""
    images,variances=np.asarray(images),np.asarray(variances)
    good=np.isfinite(images)&np.isfinite(variances)&(variances>0)
    if selection is not None:
        good &= selection
    w=np.divide(1.,variances,out=np.zeros_like(variances,dtype=float),where=good)
    sw=w.sum(axis=0)
    num=(w*np.where(good,images,0)).sum(axis=0)
    mean=np.divide(num,sw,out=np.full(sw.shape,np.nan),where=sw>0)
    var=np.divide(1.,sw,out=np.full(sw.shape,np.nan),where=sw>0)
    return mean,var,good.sum(axis=0)




class BandViewer:
    """Linked target-centered stacks and the pipeline's unchanged tracking results.

    The original loader retains one wavelength per exposure at the target. Thus
    channel membership is assigned to a whole cutout by its target wavelength;
    this is not a per-pixel calibrated spectral cube. Registration uses only WCS
    and the predicted/catalog center, never the detected brightness peak.
    """
    def __init__(self, analyzer, zooms, movement, tracked=None, size=41):
        from matplotlib.widgets import Slider, RadioButtons, RangeSlider
        self.analyzer, self.zooms, self.movement = analyzer, zooms, movement
        self.name = (tracked or {}).get('name', 'SPHEREx target')
        self.n = len(analyzer.frames)
        if not self.n or len(zooms) != self.n:
            raise ValueError('Provide the nonempty results["zooms"] from the same run.')
        if any(str(f['obs_id']) != str(z['obs_id']) for f,z in zip(analyzer.frames, zooms)):
            raise ValueError('Frame/zoom order does not match.')
        if size < 3 or size % 2 != 1:
            raise ValueError('size must be odd and at least 3')
        self.size = size
        self.detector = np.array([f['detector'] for f in analyzer.frames])
        if not np.isin(self.detector, np.arange(1, 7)).all():
            raise ValueError('SPHEREx detector IDs must be 1 through 6.')
        self.channels = np.full(self.n, -1, int)
        for d in range(1,7):
            m = self.detector == d
            self.channels[m] = channel_index(d, analyzer.wavelength[m])
        self.days = analyzer.mjd - analyzer.mjd[0]
        self.good = (analyzer.reliable & np.isfinite(analyzer.flux) &
                     np.isfinite(analyzer.flux_err) & (analyzer.flux_err > 0))
        yy,xx = np.indices((size,size), dtype=float)
        east = -(xx-size//2)*cfg.PIXEL_ARCSEC
        north = (yy-size//2)*cfg.PIXEL_ARCSEC
        # Register once, reuse for all 108 views; native cutouts remain unchanged.
        sampled = [sample_frame(z,east,north) for z in zooms]
        self.images,self.variances = (np.asarray(v,dtype=np.float32) for v in zip(*sampled))
        self.keys = [(d,c) for d in range(1,7) for c in range(18)]
        self.groups = {key: self._members(*key) for key in self.keys}
        self.stacks = {key: weighted_image(self.images[idx], self.variances[idx])
                       for key,idx in self.groups.items()}
        self.sed = analyzer.band_spectrum()
        self.time_mask = np.ones(self.n, bool)
        self.key = (int(self.detector[0]), self._local_channel(0))
        self.index = 0
        self._syncing = False
        self.gallery = plt.figure(figsize=(19,8.5))
        gs = self.gallery.add_gridspec(6,18,left=.045,right=.99,top=.88,bottom=.07,
                                       wspace=.10,hspace=.34)
        self.tiles = {}
        cmap = plt.get_cmap('gray').copy(); cmap.set_bad('#353b45')
        for d,c in self.keys:
            key = (d,c)
            ax = self.gallery.add_subplot(gs[d-1,c]); self.tiles[ax] = key
            mean,var,n = self.stacks[key]
            snr = mean / np.sqrt(var)
            ax.imshow(snr,origin='lower',cmap=cmap,vmin=-3,vmax=12,interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            title = 'All' if c == 0 else f'{CENTERS[d-1,c-1]:.3f}'
            ax.set_title(f'{title}\nn={len(self.groups[key])}',fontsize=7)
            if c == 0: ax.set_ylabel(f'D{d}',fontsize=10,weight='bold')
            if not np.any(n): ax.text(.5,.5,'—',ha='center',va='center',transform=ax.transAxes,color='white')
        covered = int(np.count_nonzero(self.sed['count']))
        self.gallery.suptitle(f'{self.name} | 102-channel atlas + six detector stacks\n'
                              f'{covered}/102 channels with reliable target photometry | '
                              'approximate target-wavelength bins; click any tile',fontsize=14)
        self.gallery.text(.5,.025,'All = broad detector stack  •  n = retained captures  •  '
                          'grayscale = approximate per-pixel S/N (−3 to +12)  •  — = no valid image pixels',ha='center',fontsize=10)
        self.detail = plt.figure(figsize=(14,8.5))
        grid = self.detail.add_gridspec(2,2,left=.075,right=.96,top=.83,bottom=.24,wspace=.31,hspace=.48)
        self.ax_stack = self.detail.add_subplot(grid[0,0])
        self.ax_native = self.detail.add_subplot(grid[0,1])
        self.ax_spec = self.detail.add_subplot(grid[1,0])
        self.ax_track = self.detail.add_subplot(grid[1,1])
        self.slider = Slider(self.detail.add_axes([.16,.065,.57,.035]),'Capture T',1,max(2,self.n),valinit=1,valstep=1,valfmt='%d')
        if self.n == 1: self.slider.set_active(False)
        self.mode = RadioButtons(self.detail.add_axes([.79,.018,.18,.09]),['S/N','Brightness','Coverage'])
        self.time_slider = None
        if np.ptp(self.days) > 0:
            self.time_slider = RangeSlider(self.detail.add_axes([.16,.135,.57,.025]),
                                           'Spectrum days',float(self.days.min()),float(self.days.max()),
                                           valinit=(float(self.days.min()),float(self.days.max())),valfmt='%.2f')
            self.time_slider.on_changed(self._on_time)
        self.slider.on_changed(self._on_slider)
        self.mode.on_clicked(lambda _: self._redraw())
        self.gallery.canvas.mpl_connect('button_press_event',self._on_tile)
        self.detail.canvas.mpl_connect('key_press_event',self._on_key)
        self.detail.canvas.mpl_connect('pick_event',self._on_pick)
        # Matplotlib keeps weak refs to bound callbacks; keep the viewer alive.
        self.gallery._band_viewer = self.detail._band_viewer = self
        self._redraw()

    def _local_channel(self,i):
        return int(self.channels[i]%17+1) if self.channels[i]>=0 else 0

    def _members(self,d,c):
        mask = self.detector == d
        if c: mask &= self.channels == (d-1)*17+c-1
        return np.flatnonzero(mask)

    def _on_tile(self,event):
        if event.inaxes not in self.tiles: return
        self.select(*self.tiles[event.inaxes])

    def select(self,detector,channel=0):
        """Select a detector, and optionally a local nominal channel (1–17)."""
        key=(int(detector),int(channel))
        if key not in self.groups: raise ValueError('detector 1–6, channel 0–17')
        self.key=key
        idx=self.groups[key]
        if len(idx):
            # Keep the current frame if it belongs to the chosen group; otherwise
            # select the closest time to the previous frame.
            closest=idx[np.argmin(np.abs(self.days[idx]-self.days[self.index]))]
            if self.index not in idx: self.index=int(closest)
            self._syncing=True
            try: self.slider.set_val(self.index+1)
            finally: self._syncing=False
        self._redraw()

    def _on_time(self, bounds):
        lo, hi = bounds
        self.time_mask = (self.days >= lo) & (self.days <= hi)
        self.sed = self.analyzer.band_spectrum(self.time_mask)
        self._redraw()

    def _on_slider(self,value):
        if self._syncing: return
        self.index=min(self.n-1,max(0,int(value)-1))
        # Chronological browsing follows the actual detector and wavelength.
        self.key=(int(self.detector[self.index]),self._local_channel(self.index))
        self._redraw()

    def _on_key(self,event):
        if event.key in ('left','right'):
            self.slider.set_val(np.clip(self.index+1+(1 if event.key=='right' else -1),1,self.n))

    def _on_pick(self,event):
        idx=getattr(event.artist,'_frame_indices',None)
        if idx is not None and len(event.ind):
            self.slider.set_val(int(idx[event.ind[0]])+1)

    def _redraw(self):
        from .plots import draw_zoom, circle_legend
        a=self.analyzer; d,c=self.key; idx=self.groups[self.key]
        for ax,key in self.tiles.items():
            for spine in ax.spines.values():
                spine.set_color('#e39b36' if key==self.key else '#666666')
                spine.set_linewidth(2.2 if key==self.key else .4)
        self.gallery.canvas.draw_idle()
        ax=self.ax_stack; ax.clear()
        mean,var,n=self.stacks[self.key]
        mode=self.mode.value_selected
        data=mean/np.sqrt(var) if mode=='S/N' else mean if mode=='Brightness' else np.where(n>0,n,np.nan)
        finite=data[np.isfinite(data)]
        lim=(-3,12) if mode=='S/N' else ((0,max(1,int(n.max()))) if mode=='Coverage' else
             ((float(np.percentile(finite,2)),float(np.percentile(finite,99))) if finite.size else (0,1)))
        if lim[1]<=lim[0]: lim=(lim[0],lim[0]+1)
        ax.imshow(data,origin='lower',cmap='gray' if mode!='Coverage' else 'viridis',
                  vmin=lim[0],vmax=lim[1],interpolation='nearest')
        center=self.size//2
        ax.add_patch(Circle((center,center),cfg.APERTURE_RADIUS_PIX,fill=False,ls='--',color='deepskyblue'))
        label='all wavelengths' if c==0 else f'C{c:02d} ≈ {CENTERS[d-1,c-1]:.3f} µm'
        ax.set_title(f'D{d} | {label}\n{len(idx)} captures · {mode} · center coverage {n[center,center]}',fontsize=10)
        ax.set(xlabel='Target-relative x (east ←)',ylabel='Target-relative y (north ↑)')
        if not np.any(n): ax.text(.5,.5,'No valid image pixels',transform=ax.transAxes,ha='center',color='white',bbox=dict(facecolor='#333333'))
        ax=self.ax_native; ax.clear()
        active=len(idx)>0 and self.index in idx
        if active:
            draw_zoom(ax,self.zooms[self.index],self.index,self.n,a.mjd[0],self.movement,a.brightness)
            ax.set(xlabel='Original cutout x (px)',ylabel='Original cutout y (px)')
            ax.legend(handles=circle_legend(self.movement),loc='lower left',fontsize=7)
        else:
            ax.text(.5,.5,'No retained captures in this channel',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
        if active: ax.set_axis_on()
        ax=self.ax_spec; ax.clear()
        colors=plt.get_cmap('tab10')
        for det in range(1,7):
            m=(self.detector==det)&self.good&self.time_mask
            if m.any():
                ax.errorbar(a.wavelength[m],a.flux[m],yerr=a.flux_err[m],fmt='.',ms=4,alpha=.38,color=colors(det-1))
            sl=slice((det-1)*17,det*17); s=self.sed
            ok=s['count'][sl]>0
            ax.errorbar(s['wavelength_um'][sl][ok],s['flux_mjy'][sl][ok],yerr=s['flux_err_mjy'][sl][ok],
                        fmt='o',ms=4,color=colors(det-1),label=f'D{det}')
        bad=(~self.good)&self.time_mask&np.isfinite(a.wavelength)&np.isfinite(a.flux)
        if bad.any(): ax.scatter(a.wavelength[bad],a.flux[bad],marker='x',color='0.5',s=10,label='Excluded')
        if c: ax.axvline(CENTERS[d-1,c-1],color='#e39b36',ls='--',lw=1)
        else: ax.axvspan(*LIMITS[d-1],color='#e39b36',alpha=.12)
        if active and self.time_mask[self.index]: ax.scatter(a.wavelength[self.index],a.flux[self.index],s=100,facecolors='none',edgecolors='red',zorder=5)
        ax.set(xlim=(.7,5.05),xlabel='Wavelength (µm)',ylabel='Aperture flux density (mJy)',
               title=f'Live spectrum | {np.count_nonzero(self.sed["count"])}/102 channels in time window')
        ax.legend(ncol=3,fontsize=7,loc='best'); ax.grid(alpha=.18)
        ax=self.ax_track; ax.clear()
        moving='residual_arcsec' in self.movement
        values=np.asarray(self.movement['residual_arcsec' if moving else 'offset_arcsec'])
        detected=np.asarray(self.movement.get('detected',np.ones(self.n,bool)),bool) & self.good
        flags=np.asarray(self.movement['flagged'],bool)
        for det in range(1,7):
            for ok,marker in [(True,'o'),(False,'x')]:
                keep=(self.detector==det)&(detected==ok)&np.isfinite(values)
                indices=np.flatnonzero(keep)
                if len(indices):
                    dots=ax.scatter(self.days[indices],values[indices],s=15,marker=marker,color=colors(det-1),alpha=.65,picker=5)
                    dots._frame_indices=indices
        if len(idx): ax.scatter(self.days[idx],values[idx],s=42,facecolors='none',edgecolors='#e39b36',label='Selected band')
        if flags.any(): ax.scatter(self.days[flags],values[flags],s=85,facecolors='none',edgecolors='red',label='Tracking flag')
        if active:
            ax.axvline(self.days[self.index],color='0.4',ls=':',lw=1)
            ax.scatter(self.days[self.index],values[self.index],marker='s',s=105,facecolors='none',edgecolors='black',label=f'T{self.index+1}')
        ax.set(xlabel='Days since T1',ylabel='Ephemeris residual (arcsec)' if moving else 'Offset from T1 (arcsec)',
               title='Existing tracking | click capture (× = low confidence)')
        ax.grid(alpha=.18)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7,loc='best')
        self.detail.suptitle(f'{self.name} | D{d} {label}\n'
                             'Stacks and tracking: all captures • spectrum: selected days',fontsize=13)
        self.detail.canvas.draw_idle()


def launch(show_original=True, show_fit=False, **run_kwargs):
    """Run the untouched pipeline, then open the atlas and existing tracking plots."""
    from .main import run
    from .plots import show_viewer, show_motion, show_time_series, show_spectrum
    run_kwargs.pop('show',None)
    results=run(show=False,**run_kwargs)
    analyzer = SpectrumAnalyzer.from_results(results)
    results['analyzer'] = analyzer
    results['band_viewer'] = analyzer.show_bands()
    if show_original:
        frames,tracked,movement=results['frames'],results['tracked'],results['movement']
        brightness=results.get('brightness')
        show_viewer(results['zooms'],tracked,movement,brightness)
        if brightness is None: show_motion(frames,tracked,movement)
        else:
            show_time_series(frames,tracked,movement,brightness)
            show_spectrum(results['spectrum'],tracked)
    if show_fit:
        results['motion_fit'] = analyzer.motion_fit()
    plt.show()
    return results


if __name__ == '__main__':
    launch()
