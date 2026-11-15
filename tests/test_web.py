"""Offline tests for the website and the image rendering (no internet needed): python -m unittest discover tests"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from webapp import analysis
from webapp.server import app
from xarvis_pipeline import render


class PagesAndApi(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_pages(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/about').status_code, 200)
        self.assertEqual(self.client.get('/no/such/page').status_code, 404)
        self.assertEqual(self.client.get('/api/no-such').get_json(), {'error': 'not found'})

    def test_analysis_validation(self):
        bad = self.client.post('/api/analysis', json={'mode': 'coordinates', 'ra': 400, 'dec': 0})
        self.assertEqual(bad.status_code, 400)
        self.assertIn('ra', bad.get_json()['error'])
        self.assertEqual(self.client.post('/api/analysis', json={'mode': 'nope'}).status_code, 400)
        self.assertEqual(self.client.get('/api/analysis/unknownjob123').status_code, 404)

    def test_request_is_normalised(self):
        req = analysis._request({'mode': 'gaia_star', 'max_images': '10',
                                 'target': {'name': 'Barnard', 'ra': 269.45, 'dec': 4.74, 'pmra': -801.6, 'pmdec': 10362.4}})
        self.assertEqual(req['max_images'], 10)
        self.assertEqual(req['patch_px'], 50)
        self.assertEqual(req['target']['pmdec'], 10362.4)


def fake_zoom(mjd, detector, star_at=(20.0, 20.0)):
    """A 40 x 40 SPHEREx-like cutout with one star, a north-up TAN WCS and 6.15" pixels."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [133.8, -7.2]
    w.wcs.crpix = [20.5, 20.5]
    w.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
    yy, xx = np.indices((40, 40))
    rng = np.random.default_rng(int(mjd))
    image = rng.normal(0, 0.05, (40, 40)) + 3 * np.exp(-((xx - star_at[0]) ** 2 + (yy - star_at[1]) ** 2) / 2)
    header = fits.Header()
    header['MSKN00'] = 'TRANSIENT'
    header['MSKN21'] = 'SOURCE'
    return {'image': image, 'variance': np.full((40, 40), 0.0025), 'bad': np.zeros((40, 40), bool),
            'flags': np.zeros((40, 40), np.int64), 'flags_header': header, 'wcs': w, 'catalog_xy': (19.5, 19.5),
            'mjd': mjd, 'detector': detector, 'wavelength_um': 1.0 + detector / 2}


class Rendering(unittest.TestCase):
    def test_frames_stacks_and_differences(self):
        zooms = [fake_zoom(60800.0, 1), fake_zoom(60800.1, 1), fake_zoom(60990.0, 1, star_at=(23.0, 20.0))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            info = render.frame_pngs(zooms, folder, 30)
            images = render.stacks_and_differences(zooms, folder, 30)
            self.assertEqual([i['png'] for i in info], ['frame_1.png', 'frame_2.png', 'frame_3.png'])
            self.assertTrue(all((folder / i['png']).stat().st_size > 100 for i in info))
            kinds = sorted(i['kind'] for i in images)
            self.assertEqual(kinds, ['difference', 'stack'])          # two visits -> a difference exists
            json.dumps(info), json.dumps(images)                     # browser-ready


if __name__ == '__main__':
    unittest.main()
