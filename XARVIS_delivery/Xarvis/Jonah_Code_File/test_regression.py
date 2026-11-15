"""Regression tests on REAL SPHEREx data: they pin the accuracy numbers measured during development, so a change
that makes the analysis worse fails loudly. They download data (minutes on the first run; seconds from the
cache), so they only run when asked:

    set JONAH_ONLINE_TESTS=1            (Windows; export JONAH_ONLINE_TESTS=1 elsewhere)
    python -m unittest Jonah_Code_File.test_regression

Measured values these protect (November 2026, USE_PSF on):
  433 Eros vs JPL Horizons: median 0.17", 95% 0.45", 353 of 389 frames detected
  Barnard's Star proper motion vs SIMBAD/Gaia: 0.2 / 0.1 sigma
  2MASS J20184628+4227134 spectrum check: PASS
  WISE 0855 field: one uncatalogued static source, cosmic rays rejected
"""

import os
import unittest

import numpy as np
from astropy.coordinates import SkyCoord

from Jonah_Code_File import api

ONLINE = os.environ.get('JONAH_ONLINE_TESTS') == '1'
QUIET = {'SAVE_RESULTS': False, 'LIVE_VIEW': False}


@unittest.skipUnless(ONLINE, 'set JONAH_ONLINE_TESTS=1 to run the real-data regression tests')
class RealDataRegression(unittest.TestCase):
    def test_eros_positions_match_horizons(self):
        _, raw = api.analyze({'asteroid': '433', 'window': ['2026-02-15', '2026-06-15']}, settings=QUIET,
                             return_raw=True)
        frames, movement = raw['frames'], raw['movement']
        detected = np.asarray(movement['detected'], bool)
        expected = SkyCoord([f['expected_ra'] for f in frames], [f['expected_dec'] for f in frames], unit='deg')
        found = SkyCoord([f['object_ra'] for f in frames], [f['object_dec'] for f in frames], unit='deg')
        residual = expected[detected].separation(found[detected]).arcsec
        self.assertGreaterEqual(detected.sum(), 340)
        self.assertLess(np.median(residual), 0.25)
        self.assertLess(np.percentile(residual, 95), 0.6)

    def test_barnards_star_proper_motion_matches_catalog(self):
        _, raw = api.analyze({'star': "Barnard's Star"}, settings=QUIET, return_raw=True)
        sigma = raw['catalog_motion']['agreement_sigma']
        self.assertLess(max(sigma), 3.0)

    def test_2mass_star_spectrum_matches_catalog(self):
        _, raw = api.analyze({'coords': [304.692867, 42.453728]}, settings=QUIET, return_raw=True)
        self.assertTrue(raw['spectrum']['same_object'])

    def test_wise0855_field_has_one_uncatalogued_source(self):
        result = api.analyze({'coords': '08h55m10.831s -07d14m42.53s'}, settings=QUIET)
        kinds = [o['class'] for o in result['new_objects']]
        self.assertEqual(kinds.count('static'), 1)
        self.assertNotIn('single', kinds)          # cosmic rays must not appear as new objects


if __name__ == '__main__':
    unittest.main()
