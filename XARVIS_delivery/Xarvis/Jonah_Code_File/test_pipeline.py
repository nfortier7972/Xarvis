"""Offline tests of the pipeline's building blocks, on synthetic data (no network, no cache needed).

Run:  python -m unittest discover -s Jonah_Code_File -p "test_*.py" -t .
"""

import unittest
from unittest import mock

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from Jonah_Code_File import config as cfg
from Jonah_Code_File import net
from Jonah_Code_File.measure import locate_object, survey_visits
from Jonah_Code_File.psf import detector_position, encircled, native_samples, psf_fit


def gaussian_psf(sigma_px=0.55, oversample=5, size=33):
    """An oversampled, normalised Gaussian 'effective PSF' like SPHEREx's EPSF stamps."""
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    psf = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * (sigma_px * oversample) ** 2))
    return psf / psf.sum()


def simple_wcs(ny=50, nx=50):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-cfg.PIXEL_ARCSEC / 3600, cfg.PIXEL_ARCSEC / 3600]
    return w


def frame_with_stars(stars, shape=(50, 50), noise=1.0, psf=None, seed=1):
    """A frame dict with point sources [(x, y, flux), ...] drawn from `psf`, plus Gaussian noise."""
    psf = gaussian_psf() if psf is None else psf
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    image = rng.normal(0, noise, shape)
    for x, y, flux in stars:
        image += flux * native_samples(psf, 5, xx - x, yy - y)
    return {'image': image, 'variance': np.full(shape, noise ** 2), 'bad': np.zeros(shape, bool),
            'wcs': simple_wcs(*shape), 'psf': psf, 'psf_oversample': 5, 'astrometry_ok': True}


class PsfTests(unittest.TestCase):
    def test_native_samples_conserve_light(self):
        psf = gaussian_psf()
        yy, xx = np.mgrid[-5:6, -5:6]
        self.assertAlmostEqual(native_samples(psf, 5, xx - 0.3, yy + 0.2).sum(), 1.0, places=2)

    def test_psf_fit_recovers_subpixel_position(self):
        frame = frame_with_stars([(24.37, 25.71, 400.0)])
        fit = psf_fit(frame, 24.0, 26.0)
        self.assertIsNotNone(fit)
        self.assertLess(abs(fit[0] - 24.37), 0.03)
        self.assertLess(abs(fit[1] - 25.71), 0.03)

    def test_encircled_fraction_matches_a_gaussian(self):
        frame = frame_with_stars([])
        fraction = encircled(frame, 25.0, 25.0, radius=2.0)
        # pixel-centre aperture of radius 2 on a sigma = 0.55 px Gaussian: essentially all of the light
        self.assertGreater(fraction, 0.97)
        wide = dict(frame, psf=gaussian_psf(sigma_px=1.2))
        self.assertLess(encircled(wide, 25.0, 25.0, radius=2.0), fraction)       # blurrier: less inside

    def test_detector_position_is_the_linear_offset(self):
        header = fits.Header({'CTYPE1A': 'LINEAR', 'CTYPE2A': 'LINEAR', 'CRPIX1A': -354.0, 'CRPIX2A': -126.0,
                              'CDELT1A': 1.0, 'CDELT2A': 1.0, 'CRVAL1A': 0.0, 'CRVAL2A': 0.0})
        self.assertEqual(tuple(float(v) for v in detector_position(header, 0.0, 0.0)), (355.0, 127.0))


class TrackingTests(unittest.TestCase):
    @staticmethod
    def expect_at(frame, x, y):
        ra, dec = frame['wcs'].pixel_to_world_values(x, y)
        frame.update(catalog_xy=(x, y), expected_ra=float(ra), expected_dec=float(dec))
        return locate_object(frame)

    def test_nearest_source_wins_over_a_brighter_neighbour(self):
        # the target where expected, a 2x brighter source 4 px away: the target is tracked, not the neighbour
        found = self.expect_at(frame_with_stars([(25.0, 25.0, 600.0), (29.0, 25.0, 1200.0)]), 25.0, 25.0)
        self.assertEqual(found['track'], 'ok')
        self.assertLess(found['track_sep_px'], 0.5)

    def test_blended_neighbour_is_refused_not_tracked(self):
        # a 5x brighter source only 2.7 px away swallows the faint target's peak (the Eros T243 situation): the
        # frame must be refused ('off'), never reported as the target at the neighbour's position
        found = self.expect_at(frame_with_stars([(25.0, 25.0, 300.0), (27.7, 25.0, 1500.0)]), 25.0, 25.0)
        self.assertEqual(found['track'], 'off')

    def test_nothing_near_means_undetected(self):
        frame = frame_with_stars([(10.0, 10.0, 800.0)])
        ra, dec = frame['wcs'].pixel_to_world_values(35.0, 35.0)
        frame.update(catalog_xy=(35.0, 35.0), expected_ra=float(ra), expected_dec=float(dec))
        self.assertEqual(locate_object(frame)['track'], 'undetected')

    def test_survey_visits_split_on_long_gaps(self):
        mjd = np.array([61000.0, 61000.5, 61003.0, 61180.0, 61181.0, 60800.0])
        self.assertEqual(list(survey_visits(mjd)), [1, 1, 1, 2, 2, 0])


class NetTests(unittest.TestCase):
    class Response:
        def __init__(self, status, content=b'ok', headers=None):
            self.status_code, self.content, self.headers = status, content, headers or {}

    def run_with(self, responses, **kwargs):
        session = mock.Mock()
        session.get.side_effect = responses
        with mock.patch.object(net, 'session', return_value=session), mock.patch.object(net.time, 'sleep') as sleep:
            try:
                return net.get('https://example.invalid/x', **kwargs), session, sleep
            except Exception as e:
                return e, session, sleep

    def test_retries_after_a_server_error(self):
        result, session, _ = self.run_with([self.Response(502), self.Response(200, b'data')])
        self.assertEqual(result, b'data')
        self.assertEqual(session.get.call_count, 2)

    def test_permanent_refusal_is_not_retried(self):
        result, session, _ = self.run_with([self.Response(404)])
        self.assertIsInstance(result, net.Refused)
        self.assertEqual(session.get.call_count, 1)

    def test_rate_limit_reply_waits_as_asked(self):
        result, _, sleep = self.run_with([self.Response(429, headers={'Retry-After': '7'}), self.Response(200)],
                                         service='jpl')
        self.assertEqual(result, b'ok')
        self.assertIn(mock.call(7.0), sleep.call_args_list)

    def test_wrong_content_is_retried(self):
        def check(content):
            if content != b'good':
                raise net.BadResponse('error page')
        result, session, _ = self.run_with([self.Response(200, b'<html>'), self.Response(200, b'good')], check=check)
        self.assertEqual(result, b'good')


class ApiTests(unittest.TestCase):
    def test_jsonable_handles_numpy_nan_and_nesting(self):
        from Jonah_Code_File.api import jsonable
        value = jsonable({'a': np.float32(1.5), 'b': np.array([1, 2]), 'c': np.nan, 'd': (np.bool_(True), None)})
        self.assertEqual(value, {'a': 1.5, 'b': [1, 2], 'c': None, 'd': [True, None]})

    def test_settings_override_restores_and_rejects_unknown(self):
        from Jonah_Code_File.api import settings_override
        before = cfg.APERTURE_RADIUS_PIX
        with settings_override({'APERTURE_RADIUS_PIX': 3.0, 'CUTOUT_SIZE': 7}):
            self.assertEqual(cfg.APERTURE_RADIUS_PIX, 3.0)
            self.assertEqual(cfg.CUTOUT_SIZE.to_value('arcmin'), 7)          # unit kept
        self.assertEqual(cfg.APERTURE_RADIUS_PIX, before)
        with self.assertRaises(KeyError):
            with settings_override({'NOT_A_SETTING': 1}):
                pass


class StarMotionTests(unittest.TestCase):
    def test_proper_motion_with_fixed_parallax(self):
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from Jonah_Code_File.algorithm import parallax_factors, star_motion_fit
        origin = SkyCoord(269.45, 4.69, unit='deg')
        t = np.r_[np.linspace(60800, 60806, 30), np.linspace(60980, 60986, 30), np.linspace(61160, 61166, 30)]
        pe, pn = parallax_factors(origin.ra.deg, origin.dec.deg, t)
        years = (t - t.mean()) / 365.25
        east = -0.8 * years + 0.548 * pe
        north = 10.36 * years + 0.548 * pn
        coords = origin.spherical_offsets_by(east * u.arcsec, north * u.arcsec)
        frames = [{'mjd': m, 'object_ra': c.ra.deg, 'object_dec': c.dec.deg, 'object_found': True,
                   'n_bad_in_aperture': 0, 'object_err_arcsec': 0.05, 'alignment_px': 0.0}
                  for m, c in zip(t, coords)]
        fit = star_motion_fit(frames, parallax_mas=548.0)
        np.testing.assert_allclose(fit['proper_motion_mas_per_yr'], [-800, 10360], atol=60)
        self.assertTrue(fit['parallax_reliable'])


if __name__ == '__main__':
    unittest.main()
