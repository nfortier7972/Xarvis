"""Fit apparent SPHEREx motion; run ``python -m Jonah_Code_File.algorithm`` to process sibling results/."""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from matplotlib.patches import Ellipse
from scipy.interpolate import PchipInterpolator

from . import config as cfg
from .measure import usable_frames

log = logging.getLogger(__name__)

_NO_SNR_LIMIT = 1e12        # stands in for "signal-to-noise not measured" (a finite number usable_frames accepts)


def _observations(frames: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, SkyCoord]:
    """Return usable times and tangent-plane offsets, plus their original-frame indices and origin."""
    if not frames:
        raise ValueError("frames must contain at least one measured position")

    times = np.asarray([float(frame["mjd"]) for frame in frames], dtype=float)
    ra = np.asarray([float(frame["object_ra"]) for frame in frames], dtype=float)
    dec = np.asarray([float(frame["object_dec"]) for frame in frames], dtype=float)
    # Same rule as the rest of the pipeline (measure.usable_frames): detected, on track and no bad pixel in the
    # circle (or a saturated star's glow fit). Frames where the object was NOT found keep its expected position
    # for display - fitting those would mix catalog/ephemeris positions into the measured motion.
    found = np.asarray([frame.get("object_found", True) for frame in frames], dtype=bool)
    # (frames without these measurements - e.g. synthetic ones - aren't excluded by them)
    reliable = found & usable_frames([dict(frame, n_bad_in_aperture=frame.get("n_bad_in_aperture", 0),
                                           snr=frame.get("snr", _NO_SNR_LIMIT)) for frame in frames])
    valid = reliable & np.isfinite(times) & np.isfinite(ra) & np.isfinite(dec)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("frames contain no reliable finite positions")

    origin_index = indices[np.argmin(times[indices])]
    origin = SkyCoord(ra[origin_index], dec[origin_index], unit="deg")
    coords = SkyCoord(ra[indices], dec[indices], unit="deg")
    east, north = origin.spherical_offsets_to(coords)
    offsets = np.full((len(frames), 2), np.nan, dtype=float)
    offsets[indices, 0] = east.to_value("arcsec")
    offsets[indices, 1] = north.to_value("arcsec")
    return times, offsets, indices, origin


def get_pos_and_speed(frames: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Get per-frame sky offsets and angular velocities.

    Positions are east/north offsets in arcseconds from the first reliable observation.
    Speeds are east/north angular rates in arcseconds per day, estimated by finite
    differences between distinct timestamps. Simultaneous frames share one rate.
    Unusable frames have NaN rows; physical distances and linear speeds are not
    available from sky positions alone.
    """
    times, positions, indices, _ = _observations(frames)
    speeds = np.full_like(positions, np.nan)

    valid_times = times[indices]
    unique_times, epoch_index = np.unique(valid_times, return_inverse=True)
    if len(unique_times) < 2:
        return positions, speeds

    epoch_positions = np.zeros((len(unique_times), 2), dtype=float)
    epoch_counts = np.bincount(epoch_index)
    np.add.at(epoch_positions, epoch_index, positions[indices])
    epoch_positions /= epoch_counts[:, None]
    epoch_speeds = np.gradient(epoch_positions, unique_times, axis=0,
                               edge_order=2 if len(unique_times) >= 3 else 1)
    speeds[indices] = epoch_speeds[epoch_index]
    return positions, speeds


def _position_uncertainties(frames: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    """Combine centroid and available alignment errors, filling missing errors conservatively."""
    errors = np.asarray([float(frames[i].get("object_err_arcsec", np.nan)) for i in indices])
    alignment = np.asarray([float(frames[i].get("alignment_px", np.nan)) for i in indices])
    alignment = np.where(np.isfinite(alignment) & (alignment >= 0),
                         alignment * cfg.PIXEL_ARCSEC, 0.0)
    sigma = np.hypot(errors, alignment)
    known = np.isfinite(sigma) & (sigma > 0)
    if known.any():
        sigma[~known] = np.median(sigma[known])
    else:
        sigma[:] = 1.0
    return sigma


def orbital_prediction(
    frames: Sequence[Mapping[str, Any]] | None = None,
    prediction_days: float = 0.0,
    show_plot: bool = True,
    make_plot: bool = True,
) -> dict[str, Any]:
    """Fit and plot the measured angular trajectory, optionally extrapolating in time.

    This is a weighted polynomial fit to apparent east/north motion, not a physical
    Keplerian orbit. SPHEREx frames provide direction and time but not distance or
    observer ephemerides, so they cannot determine a six-dimensional orbital state
    or support Gauss-Radau/JPL ephemeris propagation by themselves. A quadratic
    angular model is used when at least three distinct epochs are available; with
    two epochs the fit is linear. Extrapolation is disabled by default.

    Returns the observations, fitted/predicted positions, rates, covariance, and
    Matplotlib figure. All angular quantities are in arcseconds (rates per day).
    """
    if frames is None:
        raise ValueError("frames are required to fit an apparent trajectory")
    prediction_days = float(prediction_days)
    if not np.isfinite(prediction_days) or prediction_days < 0:
        raise ValueError("prediction_days must be a finite non-negative number")
    if show_plot and not make_plot:
        raise ValueError("show_plot=True requires make_plot=True")

    times, all_positions, indices, origin = _observations(frames)
    fit_times = times[indices]
    positions = all_positions[indices]
    unique_times = np.unique(fit_times)
    if len(unique_times) < 2:
        raise ValueError("at least two distinct observation times are required to fit motion")

    degree = min(2, len(unique_times) - 1)
    n_coefficients = degree + 1
    sigma = _position_uncertainties(frames, indices)
    center_time = float(np.average(fit_times, weights=1 / sigma**2))
    time_scale = float(np.max(np.abs(fit_times - center_time)))
    if time_scale == 0:
        raise ValueError("observation times must span a non-zero interval")

    scaled_time = (fit_times - center_time) / time_scale
    design = np.column_stack((np.ones(len(indices)), scaled_time, 0.5 * scaled_time**2))[:, :n_coefficients]
    weighted_design = design / sigma[:, None]
    weighted_positions = positions / sigma[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_positions, rcond=None)
    if rank < n_coefficients:
        raise ValueError("observations do not constrain the requested motion model")

    residuals = positions - design @ coefficients
    covariance_base = np.linalg.inv(weighted_design.T @ weighted_design)
    degrees_of_freedom = max(len(indices) - n_coefficients, 1)
    reduced_chi_squared = np.sum((residuals / sigma[:, None]) ** 2, axis=0) / degrees_of_freedom
    covariance_axes = [covariance_base * max(1.0, float(scale)) for scale in reduced_chi_squared]
    covariance = np.zeros((2 * n_coefficients, 2 * n_coefficients), dtype=float)
    covariance[:n_coefficients, :n_coefficients] = covariance_axes[0]
    covariance[n_coefficients:, n_coefficients:] = covariance_axes[1]

    velocity = coefficients[1] / time_scale
    acceleration = (coefficients[2] / time_scale**2 if degree == 2
                    else np.full(2, np.nan, dtype=float))

    start_time, last_time = float(np.min(fit_times)), float(np.max(fit_times))
    end_time = last_time + prediction_days
    sample_count = max(100, 4 * len(indices))
    prediction_times = np.linspace(start_time, end_time, sample_count)
    prediction_scaled_time = (prediction_times - center_time) / time_scale
    prediction_design = np.column_stack((np.ones(sample_count), prediction_scaled_time,
                                         0.5 * prediction_scaled_time**2))[:, :n_coefficients]
    predicted_positions = prediction_design @ coefficients

    fig = None
    if make_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(positions[:, 0], positions[:, 1], s=24, color="black", label="observations", zorder=3)
        observed_path = prediction_times <= last_time
        ax.plot(predicted_positions[observed_path, 0], predicted_positions[observed_path, 1],
                color="tab:blue", label="weighted fit")
        if prediction_days > 0:
            ax.plot(predicted_positions[~observed_path, 0], predicted_positions[~observed_path, 1],
                    color="tab:orange", linestyle="--", label="extrapolation")

        endpoint_design = prediction_design[-1]
        endpoint_east_var = max(float(endpoint_design @ covariance_axes[0] @ endpoint_design), 0.0)
        endpoint_north_var = max(float(endpoint_design @ covariance_axes[1] @ endpoint_design), 0.0)
        if endpoint_east_var > 0 and endpoint_north_var > 0:
            ax.add_patch(Ellipse(predicted_positions[-1], 2 * np.sqrt(endpoint_east_var),
                                 2 * np.sqrt(endpoint_north_var), facecolor="tab:orange",
                                 edgecolor="tab:orange", alpha=0.2, label="1-sigma endpoint uncertainty"))

        ax.set_xlabel("East offset (arcsec; east to the left)")
        ax.set_ylabel("North offset (arcsec)")
        ax.set_title("Apparent sky-track fit (not a physical orbit)")
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        if show_plot:
            plt.show()

    return {
        "reference_coord": origin,
        "observation_times_mjd": fit_times,
        "observed_positions_arcsec": positions,
        "predicted_times_mjd": prediction_times,
        "predicted_positions_arcsec": predicted_positions,
        "residuals_arcsec": residuals,
        "velocity_arcsec_per_day": velocity,
        "acceleration_arcsec_per_day2": acceleration,
        "covariance": covariance,
        "fit_degree": degree,
        "prediction_days": prediction_days,
        "figure": fig,
    }


STAR_MODES = ("named star", "catalog star", "star", "coordinates")
VISIT_SYSTEMATIC_ARCSEC = 0.15     # per-visit systematic position offset (measured on eps Indi B: ~0.1-0.2")


def parallax_factors(ra_deg: float, dec_deg: float, mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East / north apparent shift (arcsec) of a star at (ra, dec) per arcsecond of parallax, at times `mjd`
    (Earth's barycentric position; the same model as stars.positions_at)."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    earth = get_body_barycentric("earth", Time(np.asarray(mjd, dtype=float), format="mjd"))
    x, y, z = (earth.x.to_value(u.au), earth.y.to_value(u.au), earth.z.to_value(u.au))
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    east = x * np.sin(a) - y * np.cos(a)
    north = x * np.cos(a) * np.sin(d) + y * np.sin(a) * np.sin(d) - z * np.cos(d)
    return east, north


def star_motion_fit(frames: Sequence[Mapping[str, Any]], prediction_days: float = 0.0,
                    parallax_mas: float | None = None) -> dict[str, Any]:
    """Fit a STAR's measured positions with the physical model for stars: a straight line on the sky (proper
    motion) plus the yearly parallax wobble from Earth's orbit - five numbers: position (2), proper motion (2),
    parallax. A quadratic can't follow the wobble (Barnard's Star: +/-0.55"), so for stars this replaces
    orbital_prediction. Needs >= 3 survey visits over >= 180 days.

    parallax_mas: the catalog parallax, if known - then it is held fixed and only position and proper motion are
    fitted. Use it when you can: SPHEREx sees a patch of sky from opposite sides of its orbit, turned around, so
    its per-visit offsets follow the same yearly rhythm as the parallax wobble, and with 3-4 visits SPHEREx alone
    can't tell them apart (measured: free fits gave R Dra, ~4 mas, a parallax of 448 +/- 131 mas). A free fit's
    parallax is returned with parallax_reliable=False.

    Returns proper motion (mas/yr, RA*cos(Dec) and Dec) and parallax (mas) with 1-sigma errors (scaled up by the
    fit's reduced chi-square when that is > 1), residuals, and the sampled model path."""
    from .measure import robust_scatter, survey_visits
    times, all_positions, indices, origin = _observations(frames)
    t = times[indices]
    positions = all_positions[indices]
    if np.ptp(t) < 180:
        raise ValueError("a proper-motion + parallax fit needs observations spanning at least 180 days")
    sigma = _position_uncertainties(frames, indices)
    p_east, p_north = parallax_factors(origin.ra.deg, origin.dec.deg, t)

    # Fit per survey VISIT, not per frame: frames of one visit share a small systematic offset (SPHEREx turns
    # between visits; measured on eps Indi B: ~0.1-0.2" - see measure.compare_with_catalog), so hundreds of
    # frames are NOT hundreds of independent measurements - per frame, the parallax came out at 1.8x the catalog
    # value with a 20x too small error. Each visit: weighted mean position, time and parallax factor, with an
    # error that includes VISIT_SYSTEMATIC_ARCSEC.
    visit = survey_visits(t)
    n_visits = int(visit.max()) + 1
    if n_visits < 3:
        raise ValueError("a proper-motion + parallax fit needs at least 3 survey visits")
    rows = []
    for v in range(n_visits):
        m = visit == v
        w = 1 / sigma[m] ** 2
        mean = lambda a: float(np.sum(w * a) / np.sum(w))
        scatter = max(robust_scatter(positions[m, 0]), robust_scatter(positions[m, 1])) / np.sqrt(m.sum())
        error = float(np.hypot(max(1 / np.sqrt(np.sum(w)), np.nan_to_num(scatter)), VISIT_SYSTEMATIC_ARCSEC))
        rows.append((mean(t[m]), mean(positions[m, 0]), mean(positions[m, 1]), mean(p_east[m]), mean(p_north[m]), error))
    vt, ve, vn, vpe, vpn, verr = (np.array(c) for c in zip(*rows))
    t_ref = float(np.average(vt, weights=1 / verr ** 2))
    years = (vt - t_ref) / 365.25
    n = n_visits
    fixed = parallax_mas is not None and np.isfinite(parallax_mas)
    n_params = 4 if fixed else 5
    design = np.zeros((2 * n, n_params))
    design[:n, 0], design[n:, 1] = 1.0, 1.0
    design[:n, 2], design[n:, 3] = years, years
    values = np.r_[ve, vn]
    if fixed:       # the catalog parallax's wobble is removed first; only position + proper motion are fitted
        values = values - float(parallax_mas) / 1000 * np.r_[vpe, vpn]
    else:
        design[:n, 4], design[n:, 4] = vpe, vpn
    weights = 1 / np.r_[verr, verr]
    params, *_ = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)
    dof = 2 * n - n_params
    chi2 = float(np.sum(((values - design @ params) * weights) ** 2) / dof) if dof > 0 else np.nan
    covariance = np.linalg.inv((design * weights[:, None]).T @ (design * weights[:, None])) * max(np.nan_to_num(chi2), 1.0)
    errors = np.sqrt(np.diag(covariance))
    if fixed:
        params, errors = np.r_[params, float(parallax_mas) / 1000], np.r_[errors, 0.0]
    all_years = (t - t_ref) / 365.25
    model_east = params[0] + params[2] * all_years + params[4] * p_east
    model_north = params[1] + params[3] * all_years + params[4] * p_north
    residuals = np.r_[positions[:, 0] - model_east, positions[:, 1] - model_north]

    path_t = np.linspace(t.min(), t.max() + prediction_days, max(200, 4 * len(t)))
    pe, pn = parallax_factors(origin.ra.deg, origin.dec.deg, path_t)
    path_years = (path_t - t_ref) / 365.25
    path_east = params[0] + params[2] * path_years + params[4] * pe
    path_north = params[1] + params[3] * path_years + params[4] * pn
    path = origin.spherical_offsets_by(path_east * u.arcsec, path_north * u.arcsec)
    return {
        "reference_coord": origin,
        "observation_times_mjd": t,
        "observed_positions_arcsec": positions,
        "proper_motion_mas_per_yr": params[2:4] * 1000,
        "proper_motion_err_mas_per_yr": errors[2:4] * 1000,
        "parallax_mas": float(params[4] * 1000),
        "parallax_err_mas": float(errors[4] * 1000),
        "parallax_fixed": bool(fixed),
        "parallax_reliable": bool(fixed),   # (a free SPHEREx-only parallax is not - see above)
        "reduced_chi2": chi2,
        "n_visits": n_visits,
        "residuals_arcsec": np.c_[residuals[:len(t)], residuals[len(t):]],
        "predicted_times_mjd": path_t,
        "predicted_positions_arcsec": np.c_[path_east, path_north],
        "predicted_ra_deg": path.ra.deg,
        "predicted_dec_deg": path.dec.deg,
        "model": "proper_motion_parallax",
        "prediction_days": prediction_days,
    }


def _column(table: Table, name: str, default: Any, dtype: Any) -> np.ndarray:
    """Read an ECSV column, filling masked cells or absent columns with a default."""
    if name not in table.colnames:
        return np.full(len(table), default, dtype=dtype)
    values = np.ma.asarray(table[name], dtype=dtype)
    return np.asarray(np.ma.filled(values, default), dtype=dtype)


def _frames_from_table(table: Table, source: Path) -> list[dict[str, Any]]:
    """Convert saved per-frame ECSV measurements into the algorithm's observation format."""
    required = {"mjd", "object_ra", "object_dec"}
    missing = required.difference(table.colnames)
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {', '.join(sorted(missing))}")

    times = _column(table, "mjd", np.nan, float)
    ra = _column(table, "object_ra", np.nan, float)
    dec = _column(table, "object_dec", np.nan, float)
    bad_pixels = _column(table, "n_bad_in_aperture", 0, float)
    errors = _column(table, "object_err_arcsec", np.nan, float)
    alignment = _column(table, "alignment_px", np.nan, float)
    detected = _column(table, "detected", True, bool)
    snr = _column(table, "snr", _NO_SNR_LIMIT, float)   # (older files without an snr column: nothing excluded)
    astrometry_ok = _column(table, "astrometry_ok", True, bool)

    if "track" in table.colnames:
        # only 'ok' is a trusted position ('off', 'undetected' and 'no astrometry' are not - see measure.locate_object)
        track = np.char.lower(np.char.strip(np.asarray(table["track"]).astype(str)))
        track_ok = track == "ok"
    else:
        track_ok = np.ones(len(table), dtype=bool)
    if "position_method" in table.colnames:
        method = np.char.lower(np.asarray(table["position_method"]).astype(str))
        position_ok = ~np.isin(method, ("", "none", "undetected", "nan"))
    else:
        position_ok = np.ones(len(table), dtype=bool)

    frames = []
    for i in range(len(table)):
        method_is_wings = ("position_method" in table.colnames
                           and str(table["position_method"][i]).lower() == "wings")
        frames.append({
            "mjd": float(times[i]),
            "object_ra": float(ra[i]),
            "object_dec": float(dec[i]),
            "object_found": bool(detected[i] and astrometry_ok[i] and track_ok[i] and position_ok[i]),
            "n_bad_in_aperture": (0 if method_is_wings or not np.isfinite(bad_pixels[i])
                                  else int(bad_pixels[i])),
            "object_err_arcsec": float(errors[i]),
            "alignment_px": float(alignment[i]),
            "snr": float(snr[i]),
            "position_method": "wings" if method_is_wings else "centroid",
        })
    return frames


def _spherical_interpolated_path(
    frames: Sequence[Mapping[str, Any]],
    prediction_days: float,
) -> dict[str, Any]:
    """Interpolate a long apparent track on the unit sphere instead of fitting a local polynomial."""
    times, _, indices, origin = _observations(frames)
    observed_times = times[indices]
    coords = SkyCoord([float(frames[i]["object_ra"]) for i in indices],
                      [float(frames[i]["object_dec"]) for i in indices], unit="deg")
    unique_times, epoch_index = np.unique(observed_times, return_inverse=True)
    if len(unique_times) < 2:
        raise ValueError("at least two distinct observation times are required to estimate a path")

    vectors = coords.cartesian.xyz.value.T
    epoch_vectors = np.zeros((len(unique_times), 3), dtype=float)
    np.add.at(epoch_vectors, epoch_index, vectors)
    epoch_vectors /= np.linalg.norm(epoch_vectors, axis=1)[:, None]

    last_time = float(unique_times[-1])
    end_time = last_time + prediction_days
    path_times = unique_times.copy()
    if prediction_days > 0:
        cadence = float(np.median(np.diff(unique_times)))
        future_count = max(1, int(np.ceil(prediction_days / cadence)))
        path_times = np.r_[path_times, np.linspace(last_time, end_time, future_count + 1)[1:]]
    path_vectors = np.empty((len(path_times), 3), dtype=float)
    inside = path_times <= last_time
    interpolator = PchipInterpolator(unique_times, epoch_vectors, axis=0, extrapolate=False)
    path_vectors[inside] = interpolator(path_times[inside])
    path_vectors[inside] /= np.linalg.norm(path_vectors[inside], axis=1)[:, None]

    if (~inside).any():
        previous, last = epoch_vectors[-2], epoch_vectors[-1]
        cosine = float(np.clip(np.dot(previous, last), -1.0, 1.0))
        angle = float(np.arccos(cosine))
        tangent = last - cosine * previous
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-12:
            tangent /= tangent_norm
            tangent_at_last = -np.sin(angle) * previous + np.cos(angle) * tangent
            fraction = (path_times[~inside] - last_time) / (unique_times[-1] - unique_times[-2])
            path_vectors[~inside] = (np.cos(angle * fraction)[:, None] * last
                                     + np.sin(angle * fraction)[:, None] * tangent_at_last)
        else:
            path_vectors[~inside] = last
        path_vectors[~inside] /= np.linalg.norm(path_vectors[~inside], axis=1)[:, None]

    path_ra = np.degrees(np.arctan2(path_vectors[:, 1], path_vectors[:, 0])) % 360.0
    path_dec = np.degrees(np.arcsin(np.clip(path_vectors[:, 2], -1.0, 1.0)))
    return {
        "reference_coord": origin,
        "observation_times_mjd": observed_times,
        "predicted_times_mjd": path_times,
        "predicted_ra_deg": path_ra,
        "predicted_dec_deg": path_dec,
        "model": "spherical_pchip",
    }


def run_from_results(
    results_dir: str | Path | None = None,
    prediction_days: float = 0.0,
) -> dict[str, Any]:
    """Fit apparent paths from ECSV files in the sibling ``results`` directory.

    Reads each direct ``*.ecsv`` file once, excludes undetected or invalid positions,
    and saves all estimated sky paths to ``results/estimated_paths/estimated_paths.ecsv``.
    The returned mapping also exposes each target's sampled path to callers. Long
    arcs (or files marked as asteroid results) use spherical interpolation; short
    arcs use the weighted polynomial fit. Neither model supplies a physical orbit
    without range and observer-state data.
    """
    directory = (Path(results_dir) if results_dir is not None
                 else Path(__file__).resolve().parent.parent / "results")
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Results directory not found: {directory}")
    files = sorted(directory.glob("*.ecsv"))
    if not files:
        raise FileNotFoundError(f"No .ecsv result files found in {directory}")
    prediction_days = float(prediction_days)
    if not np.isfinite(prediction_days) or prediction_days < 0:
        raise ValueError("prediction_days must be a finite non-negative number")

    estimates: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for source in files:
        table = Table.read(source, format="ascii.ecsv")
        frames = _frames_from_table(table, source)
        target = str(table.meta.get("target", source.stem))
        mode = str(table.meta.get("mode", "unknown")).lower()
        try:
            times, positions, indices, origin = _observations(frames)
            max_offset_arcsec = float(np.max(np.hypot(positions[indices, 0], positions[indices, 1])))
            use_spherical_path = mode == "asteroid" or max_offset_arcsec > 3600.0
            star_fit = None
            if mode in STAR_MODES and not use_spherical_path:
                try:                                # stars: proper motion + parallax (the physical model)
                    catalog_parallax = table.meta.get("catalog_parallax_mas")
                    star_fit = star_motion_fit(frames, prediction_days,
                                               float(catalog_parallax) if catalog_parallax not in (None, "None") else None)
                except ValueError:                  # too short a span: the apparent-track fit below
                    star_fit = None
            if use_spherical_path:
                estimate = _spherical_interpolated_path(frames, prediction_days)
            elif star_fit is not None:
                estimate = star_fit
            else:
                estimate = orbital_prediction(frames, prediction_days=prediction_days,
                                              show_plot=False, make_plot=False)
                path_coords = origin.spherical_offsets_by(
                    estimate["predicted_positions_arcsec"][:, 0] * u.arcsec,
                    estimate["predicted_positions_arcsec"][:, 1] * u.arcsec,
                )
                estimate["predicted_ra_deg"] = path_coords.ra.deg
                estimate["predicted_dec_deg"] = path_coords.dec.deg
                estimate["model"] = "weighted_polynomial"

            chronological = indices[np.argsort(times[indices])]
            first, last = positions[chronological[0]], positions[chronological[-1]]
            first_sigma = _position_uncertainties(frames, chronological[:1])[0]
            last_sigma = _position_uncertainties(frames, chronological[-1:])[0]
            displacement_significance = float(
                np.hypot(*(last - first)) / np.hypot(first_sigma, last_sigma)
            )
            estimates[source.stem] = {
                **estimate,
                "target": target,
                "mode": mode,
                "source_file": source,
                "motion_significance": displacement_significance,
                "is_moving_candidate": mode == "asteroid" or displacement_significance >= cfg.MOVEMENT_SIGMA,
                "physical_orbit_available": False,
            }
        except ValueError as exc:
            skipped[source.name] = str(exc)
            log.info(f"Skipping {source.name}: {exc}")

    output_path = None
    if estimates:
        output_dir = directory / "estimated_paths"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "estimated_paths.ecsv"
        tables = []
        for estimate in estimates.values():
            path_times = estimate["predicted_times_mjd"]
            n = len(path_times)
            tables.append(Table({
                "source_file": np.full(n, estimate["source_file"].name),
                "target": np.full(n, estimate["target"]),
                "mode": np.full(n, estimate["mode"]),
                "model": np.full(n, estimate["model"]),
                "mjd": path_times,
                "ra_deg": estimate["predicted_ra_deg"],
                "dec_deg": estimate["predicted_dec_deg"],
                "motion_significance": np.full(n, estimate["motion_significance"]),
                "moving_candidate": np.full(n, estimate["is_moving_candidate"]),
                "physical_orbit_available": np.zeros(n, dtype=bool),
                "extrapolated": path_times > float(np.max(estimate["observation_times_mjd"])),
            }))
        output = vstack(tables, metadata_conflicts="silent")
        output.meta["physical_orbit"] = (
            "Not estimated: source files do not provide line-of-sight distance and observer state."
        )
        output.write(output_path, format="ascii.ecsv", overwrite=True)

    candidates = [name for name, estimate in estimates.items() if estimate["is_moving_candidate"]]
    log.info(f"Estimated {len(estimates)} apparent paths from {len(files)} result files; "
          f"{len(candidates)} moving candidate(s).")
    if output_path is not None:
        log.info(f"Estimated paths saved to {output_path}")
    return {
        "results_dir": directory,
        "estimates": estimates,
        "moving_candidates": candidates,
        "skipped": skipped,
        "output_path": output_path,
    }


if __name__ == "__main__":
    run_from_results()
