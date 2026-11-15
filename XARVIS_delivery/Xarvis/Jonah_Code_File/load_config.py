import json
from pathlib import Path

import astropy.units as u


def load_config(filepath=None):
    """Load project settings from JSON and restore Python-only value types."""
    if filepath is None:
        filepath = Path(__file__).resolve().parent.parent / "config.json"

    with open(filepath, "r") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration in {filepath} must be a JSON object.")

    unit_mappings = {
        "SNAP_RADIUS": u.arcsec,
        "ISOLATION_RADIUS": u.arcsec,
        "CUTOUT_SIZE": u.arcmin,
        "FIELD_RADIUS": u.arcmin,
    }

    for key, unit in unit_mappings.items():
        if key in config:
            config[key] = config[key] * unit

    tuple_keys = {
        "ASTEROID_WINDOW",
        "ANNULUS_PIX",
        "WING_RADII_PIX",
        "GLOW_RING_PIX",
        "GLOW_SKY_PIX",
    }
    for key in tuple_keys:
        if key in config and isinstance(config[key], list):
            config[key] = tuple(config[key])

    set_keys = {"BAD_FLAG_NAMES", "BAD_FLAG_NAMES_MOVING", "SATURATION_FLAG_NAMES"}
    for key in set_keys:
        if key in config and isinstance(config[key], list):
            config[key] = set(config[key])

    return config
