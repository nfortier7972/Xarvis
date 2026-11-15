import json
from pathlib import Path
import unittest

from astropy import units as u

from . import config
from .load_config import load_config


class ConfigTests(unittest.TestCase):
    def test_json_contains_and_loads_every_runtime_setting(self):
        project_config = Path(__file__).resolve().parent.parent / "config.json"
        with project_config.open(encoding="utf-8") as file:
            raw_settings = json.load(file)

        runtime_settings = {name for name in vars(config) if name.isupper()}
        self.assertEqual(set(raw_settings), runtime_settings)
        self.assertEqual(config.ASTEROID, raw_settings["ASTEROID"])
        self.assertIsNone(config.MAX_IMAGES)
        self.assertEqual(config.MAX_CONCURRENT_DOWNLOADS, 16)
        self.assertEqual(config.SNAP_RADIUS, 60 * u.arcsec)
        self.assertEqual(config.CUTOUT_SIZE, 5.5 * u.arcmin)
        self.assertEqual(config.ANNULUS_PIX, (4.0, 6.5))
        self.assertEqual(config.BAD_FLAG_NAMES_MOVING, config.BAD_FLAG_NAMES - {"OUTLIER", "TRANSIENT"})

    def test_loader_converts_json_lists_to_python_types(self):
        settings = load_config()

        self.assertIsInstance(settings["ASTEROID_WINDOW"], tuple)
        self.assertIsInstance(settings["BAD_FLAG_NAMES"], set)
        self.assertEqual(settings["FIELD_RADIUS"], 4 * u.arcmin)


if __name__ == "__main__":
    unittest.main()
