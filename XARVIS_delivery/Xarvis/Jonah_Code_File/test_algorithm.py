from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from Jonah_Code_File.algorithm import get_pos_and_speed, orbital_prediction, run_from_results


class ApparentTrajectoryTests(unittest.TestCase):
    def setUp(self):
        origin = SkyCoord(359.99999, 12.0, unit="deg")
        self.times = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 2.0])
        east = 0.15 * self.times + 0.008 * self.times**2
        north = -0.08 * self.times + 0.006 * self.times**2
        coords = origin.spherical_offsets_by(east * u.arcsec, north * u.arcsec)
        self.frames = [
            {
                "mjd": 60000.0 + time,
                "object_ra": coord.ra.deg,
                "object_dec": coord.dec.deg,
                "object_found": True,
                "n_bad_in_aperture": int(index == 5),
                "object_err_arcsec": 0.1,
                "alignment_px": 0.0,
            }
            for index, (time, coord) in enumerate(zip(self.times, coords))
        ]

    def tearDown(self):
        plt.close("all")

    def test_positions_and_speeds_handle_wrap_duplicate_epochs_and_bad_frames(self):
        positions, speeds = get_pos_and_speed(self.frames)

        self.assertEqual(positions.shape, (6, 2))
        self.assertTrue(np.allclose(positions[0], [0.0, 0.0], atol=1e-8))
        self.assertTrue(np.allclose(speeds[1], speeds[2], atol=1e-8))
        self.assertTrue(np.allclose(speeds[1], [0.166, -0.068], atol=2e-5))
        self.assertTrue(np.isnan(positions[5]).all())
        self.assertTrue(np.isnan(speeds[5]).all())

    def test_quadratic_fit_recovers_motion_and_returns_plot(self):
        result = orbital_prediction(self.frames, prediction_days=2.0, show_plot=False)

        self.assertEqual(result["fit_degree"], 2)
        self.assertTrue(np.allclose(result["velocity_arcsec_per_day"], [0.1724, -0.0632], atol=2e-5))
        self.assertTrue(np.allclose(result["acceleration_arcsec_per_day2"], [0.016, 0.012], atol=2e-5))
        self.assertLess(np.max(np.abs(result["residuals_arcsec"])), 2e-5)
        self.assertEqual(result["predicted_times_mjd"][-1], 60005.0)
        self.assertIsNotNone(result["figure"])

    def test_linear_fallback_and_insufficient_epochs(self):
        linear = orbital_prediction([self.frames[0], self.frames[3]], show_plot=False)
        self.assertEqual(linear["fit_degree"], 1)
        self.assertTrue(np.isnan(linear["acceleration_arcsec_per_day2"]).all())

        with self.assertRaises(ValueError):
            orbital_prediction([self.frames[0], dict(self.frames[0], mjd=60000.0)], show_plot=False)

    def test_results_directory_saves_accessible_paths_and_ignores_undetected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            table = Table()
            table["mjd"] = [frame["mjd"] for frame in self.frames[:5]] + [60004.0]
            table["object_ra"] = [frame["object_ra"] for frame in self.frames[:5]] + [200.0]
            table["object_dec"] = [frame["object_dec"] for frame in self.frames[:5]] + [20.0]
            table["object_err_arcsec"] = [0.1] * 6
            table["alignment_px"] = [0.0] * 6
            table["n_bad_in_aperture"] = [0] * 6
            table["detected"] = [True] * 5 + [False]
            table["track"] = ["ok"] * 6
            table["position_method"] = ["centroid"] * 6
            table.meta.update(target="synthetic", mode="asteroid")
            table.write(results_dir / "synthetic.ecsv", format="ascii.ecsv")

            result = run_from_results(results_dir, prediction_days=1.0)
            estimate = result["estimates"]["synthetic"]
            saved = Table.read(result["output_path"], format="ascii.ecsv")

            self.assertEqual(estimate["model"], "spherical_pchip")
            self.assertEqual(len(estimate["observation_times_mjd"]), 5)
            self.assertIn("synthetic", result["moving_candidates"])
            self.assertEqual(set(saved["source_file"]), {"synthetic.ecsv"})
            self.assertEqual(len(saved), 5)
            self.assertTrue(np.isfinite(np.asarray(saved["ra_deg"], dtype=float)).all())
            self.assertTrue(bool(saved["extrapolated"][-1]))


if __name__ == "__main__":
    unittest.main()
