"""Tests of the website <-> pipeline bridge. From the Xarvis folder:

    python -m unittest test_web_integration                 offline (no internet needed)
    XARVIS_ONLINE=1 python -m unittest test_web_integration   also one real run against NASA/IPAC IRSA
        (PowerShell:  $env:XARVIS_ONLINE=1; python -m unittest test_web_integration)
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND', 'Agg')
for _folder in (ROOT, ROOT / 'Test_Robert_Code_Folder'):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from PIL import Image

import analysis_jobs
from app import app
from Jonah_Code_File import config as cfg
from Jonah_Code_File import main as pipeline
from Jonah_Code_File import stars, web
from Jonah_Code_File.api import settings_override

PAYLOAD = {'request': 'sky_point_analysis', 'version': 1,
           'time': {'utc': '2025-11-01 00:00 UTC', 'jd_utc': 2460980.5, 'simulated': True},
           'target': {'ra_deg': 133.795129, 'dec_deg': -7.245147},
           'patch': {'half_width_deg': 50 * 6.15 / 3600 / 2, 'size': '5 x 5 arcmin'},
           'named_target': None, 'spherex': {'altitude_km': 650.0}}
BARNARD = {'name': "Barnard's Star", 'source': 'ESA Gaia DR3', 'gaia_dr3': '4472832130942575872',
           'ra_deg_at_epoch': 269.448503, 'dec_deg_at_epoch': 4.739420, 'epoch': 2016.0,
           'pmra_mas_yr': -801.551, 'pmdec_mas_yr': 10362.394, 'parallax_mas': 546.976, 'gaia_g_mag': 8.19}


def web_job(request):
    return analysis_jobs.job_id(request)


def wait_for(client, job, seconds=30):
    for _ in range(int(seconds * 20)):
        state = client.get(f'/api/analysis/{job}').get_json()
        if state['status'] in ('done', 'error'):
            return state
        time.sleep(0.05)
    raise AssertionError(f'job still {state["status"]} after {seconds} s')


def fake_zoom(mjd, detector, rotation_deg=0.0, star_east_px=0.0, size=60, star_peak=4.0):
    """A SPHEREx-like cutout (6.15" pixels, rotated by rotation_deg) with the target at the centre and one star
    `star_east_px` pixels EAST of it on the sky."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [133.8, -7.2]
    w.wcs.crpix = [size / 2 + 1, size / 2 + 1]                       # FITS counts from 1: 0-based pixel size/2
    c, s = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
    scale = 6.15 / 3600
    w.wcs.cd = np.array([[-scale * c, scale * s], [scale * s, scale * c]])
    centre = (size / 2, size / 2)                                    # 0-based pixel of crval
    ra = 133.8 + star_east_px * scale / np.cos(np.radians(-7.2))
    sx, sy = w.world_to_pixel_values(ra, -7.2)
    yy, xx = np.indices((size, size))
    rng = np.random.default_rng(int(mjd * 10))
    image = 5 + rng.normal(0, 0.05, (size, size)) + star_peak * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 2)
    header = fits.Header()
    header['MSKN00'] = 'TRANSIENT'
    header['MSKN07'] = 'HOT'
    flags = np.zeros((size, size), np.int32)
    flags[int(centre[1]), int(centre[0])] = 1 << 7                   # a HOT pixel on the target
    return {'image': image, 'variance': np.full((size, size), 0.0025), 'bad': np.zeros((size, size), bool),
            'flags': flags, 'flags_header': header, 'wcs': w, 'catalog_xy': centre, 'mjd': mjd,
            'detector': detector, 'wavelength_um': 0.9 + detector / 2, 'snr': 12.5, 'flux_mjy': 1.0,
            'flux_err_mjy': 0.1, 'n_bad_in_aperture': 1, 'saturated': False, 'track': 'ok', 'astrometry_ok': True,
            'obs_id': f'fake_{mjd}', 'object_found': True, 'object_ra': 133.8, 'object_dec': -7.2,
            'expected_ra': 133.8, 'expected_dec': -7.2, 'date_obs': '2025-05-05T00:00:00'}


def fake_analyse(request, folder, on_progress=None):
    """Stands in for the pipeline run (no internet): two fake frames through the real image and summary code."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    zooms = [fake_zoom(60800.0, 1), fake_zoom(60990.5, 3, rotation_deg=90.0)]
    if on_progress:
        on_progress(1, 2, 'Downloading and measuring SPHEREx images: 1 of 2')
    results = {'tracked': {'name': 'fake target', 'images_available': 40}, 'frames': zooms, 'zooms': zooms}
    summary = web.summary_of(results, request, web.write_frames(zooms, folder, request['patch_px']))
    (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return summary


class Request(unittest.TestCase):
    """The popup's payload -> the pipeline's inputs."""

    def test_coordinates(self):
        request = web.web_request(PAYLOAD)
        self.assertEqual(request, {'ra_deg': 133.795129, 'dec_deg': -7.245147, 'patch_px': 50,
                                   'max_images': web.DEFAULT_MAX_IMAGES, 'star': None})
        star = web.pipeline_star(request)
        self.assertEqual((star['ra'], star['dec'], star['pmra'], star['pmdec']), (133.795129, -7.245147, 0.0, 0.0))
        self.assertNotIn('epoch_mjd', star)

    def test_number_of_images(self):
        self.assertEqual(web.web_request({**PAYLOAD, 'max_images': 8})['max_images'], 8)
        self.assertIsNone(web.web_request({**PAYLOAD, 'max_images': None})['max_images'])
        self.assertIsNone(web.web_request({**PAYLOAD, 'max_images': 'all'})['max_images'])
        for bad in (0, -3, 'many', web.MAX_IMAGES_LIMIT + 1):
            with self.assertRaises(ValueError):
                web.web_request({**PAYLOAD, 'max_images': bad})

    def test_patch_size_is_limited(self):
        wide = {**PAYLOAD, 'patch': {'half_width_deg': 2000 * 6.15 / 3600 / 2}}
        self.assertEqual(web.web_request(wide)['patch_px'], web.PATCH_PX_LIMITS[1])
        settings = web.run_settings(web.web_request(wide))
        self.assertGreaterEqual(settings['ZOOM_SIZE_PIX'], 400 * np.sqrt(2))       # filled at any rotation
        self.assertGreaterEqual(settings['CUTOUT_SIZE'] * 60, settings['ZOOM_SIZE_PIX'] * 6.15)   # the cutout covers it
        for request in (web.web_request(PAYLOAD), web.web_request(wide)):          # the catalogs cover the zoom's corners
            settings = web.run_settings(request)
            self.assertGreaterEqual(settings['FIELD_RADIUS'] * 60, settings['ZOOM_SIZE_PIX'] / 2 * np.sqrt(2) * 6.15)
        self.assertEqual(web.run_settings(web.web_request(PAYLOAD))['ZOOM_SIZE_PIX'], 75)

    def test_bad_input(self):
        for bad in (None, [], {}, {'target': {'ra_deg': 10}, 'patch': {'half_width_deg': 0.04}},
                    {**PAYLOAD, 'target': {'ra_deg': 400, 'dec_deg': 0}},
                    {**PAYLOAD, 'target': {'ra_deg': 10, 'dec_deg': 'north'}},
                    {**PAYLOAD, 'target': {'ra_deg': float('nan'), 'dec_deg': 0}},
                    {**PAYLOAD, 'patch': {}}):
            with self.assertRaises(ValueError):
                web.web_request(bad)

    def test_gaia_star_keeps_its_epoch(self):
        request = web.web_request({**PAYLOAD, 'named_target': BARNARD})
        star = web.pipeline_star(request)
        self.assertAlmostEqual(star['epoch_mjd'], 57388.5, places=3)                # J2016.0
        ra, dec = stars.positions_at({**star, 'parallax': None}, star['epoch_mjd'])
        self.assertAlmostEqual(ra[0], BARNARD['ra_deg_at_epoch'], places=7)         # at its epoch: the catalog position
        self.assertAlmostEqual(dec[0], BARNARD['dec_deg_at_epoch'], places=7)
        _, later = stars.positions_at({**star, 'parallax': None}, star['epoch_mjd'] + 10 * 365.25)
        self.assertAlmostEqual((later[0] - BARNARD['dec_deg_at_epoch']) * 3600, 103.62, delta=0.05)   # 10.36"/yr north
        # the same star WITHOUT the epoch would be moved from J2000: 16 years (166") too far - the bug this prevents
        _, wrong = stars.positions_at({**star, 'parallax': None, 'epoch_mjd': None}, star['epoch_mjd'])
        self.assertGreater(abs(wrong[0] - BARNARD['dec_deg_at_epoch']) * 3600, 150)

    def test_named_target_without_motion_is_a_fixed_point(self):
        named = {'name': 'somewhere', 'ra_deg_at_epoch': 1.0, 'dec_deg_at_epoch': 2.0, 'pmra_mas_yr': None}
        self.assertIsNone(web.web_request({**PAYLOAD, 'named_target': named})['star'])


class Pipeline(unittest.TestCase):
    """The small pipeline changes the website relies on - and that the desktop defaults did not move."""

    def test_image_selection(self):
        images = Table({'t_min': np.arange(100.0)})
        self.assertEqual(len(pipeline.select_images(images, None)), 100)
        self.assertFalse(cfg.SPREAD_IMAGES)                                         # desktop default: the first ones
        self.assertEqual(list(pipeline.select_images(images, 8)['t_min']), list(range(8)))
        with settings_override({'SPREAD_IMAGES': True}):
            spread = list(pipeline.select_images(images, 8)['t_min'])
            self.assertEqual(len(pipeline.select_images(images, 500)), 100)
        self.assertEqual((len(spread), spread[0], spread[-1]), (8, 0.0, 99.0))      # over all dates, ends included
        self.assertTrue(np.all(np.diff(spread) > 10))
        # pairs taken at the same moment (short + long arm) stay together, so each image can get its colour
        pairs = Table({'t_min': np.repeat(np.arange(50.0), 2) + np.tile([0, 1e-6], 50)})
        with settings_override({'SPREAD_IMAGES': True}):
            picked = np.asarray(pipeline.select_images(pairs, 8)['t_min'])
        self.assertEqual(len(picked), 8)
        self.assertTrue(np.all(np.round(picked[0::2]) == np.round(picked[1::2])))
        self.assertEqual((round(picked[0]), round(picked[-1])), (0, 49))

    def test_patch_size_is_read_at_run_time(self):
        frame = fake_zoom(60800.0, 1, size=120)
        frame['catalog_xy'] = (60.0, 60.0)
        self.assertEqual(pipeline.make_zooms([dict(frame)], False)[0]['image'].shape, (cfg.ZOOM_SIZE_PIX,) * 2)
        with settings_override({'ZOOM_SIZE_PIX': 80}):
            self.assertEqual(pipeline.make_zooms([dict(frame)], False)[0]['image'].shape, (80, 80))

    def test_rejection_reasons_reach_the_user(self):
        logged = ['Downloading and measuring 8 images (16 at a time)...',
                  '  skipped 8 images that could not be used, e.g.: RuntimeError: Cutout: all 50 attempts failed (HTTP 502)']
        message = web.explained(RuntimeError('None of the images could be measured at this position'), logged)
        self.assertEqual(message, 'None of the images could be measured at this position. skipped 8 images that could '
                                  'not be used, e.g.: RuntimeError: Cutout: all 50 attempts failed (HTTP 502)')
        self.assertIn('IRSA did not answer', analysis_jobs.friendly(RuntimeError(message)))
        self.assertEqual(web.explained(RuntimeError('No SPHEREx image covers x yet'), []), 'No SPHEREx image covers x yet.')

    def test_failed_images_keep_their_reason(self):
        """load_all_frames: an exception whose message starts with a blank line still gives its type and text."""
        from Jonah_Code_File import frames
        logged = []
        def broken(row, coord, moving):
            raise OSError('\nthe file is not readable')
        with mock.patch.object(frames, 'load_frame', broken), mock.patch.object(
                frames.log, 'info', lambda message: logged.append(message)):
            self.assertEqual(frames.load_all_frames(None, [{}, {}], max_images=None), [])
        self.assertTrue(any('OSError: the file is not readable' in line for line in logged), logged)

    def test_web_runs_measure_in_process(self):
        self.assertEqual(web.run_settings(web.web_request(PAYLOAD))['MEASURE_WORKERS'], 0)

    def test_web_settings_are_restored(self):
        before = {name: getattr(cfg, name) for name in web.run_settings(web.web_request(PAYLOAD))}
        with settings_override(web.run_settings(web.web_request({**PAYLOAD, 'patch': {'half_width_deg': 0.2}}))):
            self.assertTrue(cfg.SPREAD_IMAGES)
            self.assertTrue(cfg.FIELD_CHECK)                                        # compared with the catalogs
            self.assertTrue(cfg.SKYMAP_FOR_ALL_TARGETS)                             # every object named
        self.assertEqual({name: getattr(cfg, name) for name in before}, before)


class Rendering(unittest.TestCase):
    """Offline: fake frames -> the desktop viewer's images (colour, correction, circles) on one north-up grid, and
    the summary the browser reads (with the T1, T2 ... titles)."""
    F = int(cfg.DISPLAY_UPSCALE)

    @staticmethod
    def png(folder, name):
        with Image.open(Path(folder) / name) as image:
            return np.asarray(image.convert('RGB'), dtype=float)

    def test_frames_line_up(self):
        zooms = [fake_zoom(60800.0, 1, rotation_deg=0.0, star_east_px=8), fake_zoom(60800.1, 2, 90.0, 8),
                 fake_zoom(60990.0, 4, 217.0, 8)]
        with tempfile.TemporaryDirectory() as tmp:
            shown = web.write_frames(zooms, tmp, 40)
            self.assertEqual([s['png'] for s in shown], ['frame_1.png', 'frame_2.png', 'frame_3.png'])
            for s in shown:
                rgb = self.png(tmp, s['png'])
                self.assertEqual(rgb.shape, (40 * self.F, 40 * self.F, 3))          # upscaled like the desktop
                white = rgb.min(axis=2)                                             # (the circles are not white)
                row, column = np.argwhere(white > 0.8 * white.max()).mean(axis=0)  # the star's (flat-topped) centre
                # the star is 8 px EAST of the target: 8 px LEFT of the centre, same row - whatever the rotation
                self.assertLessEqual(abs(column - (11.5 + 0.5) * self.F), self.F, s['png'])
                self.assertLessEqual(abs(row - (19.5 + 0.5) * self.F), self.F, s['png'])
                self.assertEqual(s['flags_in_aperture'], ['HOT'])

    def test_circles(self):
        """The desktop viewer's circles, in PNG pixels for the overlay: red = found here, blue dashed = at T1."""
        from astropy.coordinates import SkyCoord
        zooms = [fake_zoom(60800.0, 1), fake_zoom(60990.0, 2)]
        zooms[1]['object_ra'] += 2 * 6.15 / 3600 / np.cos(np.radians(-7.2))       # found 2 px EAST of T1's position
        movement = {'reference': SkyCoord(133.8, -7.2, unit='deg')}                 # the object at T1: the centre
        with tempfile.TemporaryDirectory() as tmp:
            shown = web.write_frames(zooms, tmp, 40, movement)
        marks, centre = shown[1]['marks'], 20 * self.F
        self.assertAlmostEqual(marks['blue'][0], centre, delta=0.5)
        self.assertAlmostEqual(marks['blue'][1], centre, delta=0.5)
        self.assertAlmostEqual(marks['red'][0], centre - 2 * self.F, delta=0.5)    # east is LEFT
        self.assertAlmostEqual(marks['red'][1], centre, delta=0.5)
        self.assertEqual((marks['radius'], marks['blue_radius']),
                         (cfg.APERTURE_RADIUS_PIX * self.F, (cfg.APERTURE_RADIUS_PIX + 0.6) * self.F))
        self.assertIsNone(shown[0]['objects'])                                      # no sky map in these frames

    def test_colour_from_the_other_arm(self):
        # the same exposure on a short (D1) and a long (D4) arm; the star is much brighter at the long wavelength
        zooms = [fake_zoom(60800.0, 1, star_east_px=8, star_peak=1.0), fake_zoom(60800.0, 4, star_east_px=8, star_peak=8.0),
                 fake_zoom(60801.0, 1, star_east_px=0, star_peak=4.0), fake_zoom(60801.0, 4, star_east_px=0, star_peak=4.0)]
        with tempfile.TemporaryDirectory() as tmp:
            shown = web.write_frames(zooms, tmp, 40)
            rgb = self.png(tmp, 'frame_1.png')
        self.assertEqual(shown[0]['colour_um'], [1.4, 2.9])                          # short, long: the colour key
        star = rgb[(19 + 1) * self.F - 4:(20 + 1) * self.F + 4, (11 + 1) * self.F - 4:(12 + 1) * self.F + 4]
        self.assertGreater(star[..., 0].mean(), star[..., 2].mean() + 20)            # redder than the typical star

    def test_summary_and_titles(self):
        zooms = [fake_zoom(60800.0, 1), fake_zoom(60990.5, 3)]
        zooms[1]['snr'] = np.nan                                                    # not measurable -> null, not NaN
        request = web.web_request(PAYLOAD)
        movement = {'flagged': np.array([False, True]), 'offset_arcsec': np.array([0.0, 7.5]),
                    'reference_label': 'object at T1'}
        brightness = {'flagged': np.array([False, False]), 'deviation': np.array([0.012, np.nan])}
        with tempfile.TemporaryDirectory() as tmp:
            results = {'tracked': {'name': 'x', 'images_available': 7}, 'frames': zooms, 'zooms': zooms,
                       'movement': movement, 'brightness': brightness}
            summary = json.loads(json.dumps(web.summary_of(results, request, web.write_frames(zooms, tmp, 50))))
        self.assertEqual((summary['n_frames'], summary['images_available'], summary['patch_px']), (2, 7, 50))
        first, second = summary['frames']
        self.assertEqual(first['time_utc'], '2025-05-05 00:00:00')
        self.assertEqual((first['detector'], first['wavelength_um'], first['snr']), (1, 1.4, 12.5))
        self.assertIsNone(second['snr'])
        self.assertEqual((first['name'], second['name']), ('T1', 'T2'))
        self.assertEqual(first['title'][0], 'T1 of 2   2025-05-05 00:00   +0.00 d   detector 1   '
                                            '[1 bad pixel(s) in circle: not used]')
        self.assertEqual(first['title'][1], 'λ = 1.400 µm   moved 0.00"')
        self.assertEqual(first['title'][2], 'flux 1.00 ± 0.10 mJy   vs spectrum +1.2%')
        self.assertEqual(second['title'][1], 'λ = 2.400 µm   moved 7.50"  << MOVED')
        self.assertIn('n/a (too few similar-wavelength frames)', second['title'][2])
        self.assertEqual(summary['circles'], {'red': 'object found in this frame', 'blue': 'object at T1'})
        for key in ('n', 'png', 'mjd', 'flags_in_aperture', 'n_bad_in_aperture', 'saturated', 'colour_um'):
            self.assertIn(key, first)


def fake_sky_map(zooms):
    """A sky map like skymap.map_sky's on two fake frames: a Gaia star, a named Hipparcos star, an asteroid,
    an artifact and one NEW object (seen in both frames)."""
    from astropy.table import Table as T
    catalog = T(rows=[('Gaia DR3 1', 'Gaia', 'Gaia DR3 1', 133.801, -7.199, 57388.5, 1.0, -2.0, 0.5, 0),
                      ('Sirius (HIP 32349)', 'Gaia', 'Gaia DR3 2', 133.79, -7.21, 57388.5, 0.0, 0.0, np.nan, 32349)],
                names=('label', 'source', 'name', 'ra', 'dec', 'epoch_mjd', 'pmra', 'pmdec', 'parallax', 'hip'))
    for n, zoom in enumerate(zooms):
        zoom['map'] = [
            {'kind': 'star', 'source': 'Gaia', 'label': 'Gaia DR3 1', 'ra': 133.801, 'dec': -7.199, 'detected': True,
             'named': False, 'catalog_row': 0},
            {'kind': 'star', 'source': 'Gaia', 'label': 'Sirius (HIP 32349)', 'ra': 133.79, 'dec': -7.21,
             'detected': True, 'named': True, 'catalog_row': 1},
            {'kind': 'small body', 'label': '433 Eros (A898 PA)', 'ra': 133.805, 'dec': -7.195, 'detected': n == 1,
             'body': '433', 'info': 'an, Amor'},
            {'kind': 'artifact', 'label': 'artifact: GHOST', 'ra': 133.796, 'dec': -7.204, 'detected': True},
            {'kind': 'new', 'label': 'NEW-1 (moving)', 'ra': 133.803 + 0.001 * n, 'dec': -7.198, 'detected': True,
             'new_id': 'NEW-1', 'info': 'moves 3"/h'},
            {'kind': 'star', 'source': 'Gaia', 'label': 'far away', 'ra': 140.0, 'dec': 0.0, 'detected': True,
             'named': False, 'catalog_row': 0}]
    new = [{'id': 'NEW-1', 'kind': 'moving', 'description': 'moves 3"/h', 'ra': 133.803, 'dec': -7.198,
            'detections': [{'mjd': z['mjd'], 'ra': 133.803, 'dec': -7.198} for z in zooms],
            'track': {'kind': 'path', 'name': 'NEW-1 (unknown mover)', 'mjd': [z['mjd'] for z in zooms],
                      'ra': [133.803, 133.804], 'dec': [-7.198, -7.198]}}]
    bodies = [{'fullname': '433 Eros (A898 PA)', 'kind': 'an', 'orbit_class': 'Amor', 'positions': {1: (133.805, -7.195)}}]
    return {'catalog': catalog, 'bodies': bodies, 'planets': {}, 'new_objects': new, 'checked': {0, 1}}


class SkyMap(unittest.TestCase):
    """Offline: the sky map, the catalog comparison and tracking - what the desktop viewer has - reach the browser."""

    def setUp(self):
        self.zooms = [fake_zoom(60800.0, 1), fake_zoom(60990.0, 2)]
        self.sky = fake_sky_map(self.zooms)
        self.results = {'tracked': {'name': 'x', 'images_available': 2, 'skymap': self.sky}, 'frames': self.zooms,
                        'zooms': self.zooms, 'skymap': self.sky,
                        'field': {'offset_px': np.array([[0.1, 0.0], [0.0, 0.1], [0.05, 0.0]]), 'candidates': []}}

    def test_objects_drawn_and_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            shown = web.write_frames(self.zooms, tmp, 40)
        objects = shown[0]['objects']
        self.assertEqual([o['id'] for o in objects], ['star:0', 'star:1', 'small body:433', 'artifact:artifact: GHOST', 'new:NEW-1'])
        self.assertEqual(shown[0]['map_text'], 'mapped: 2 Gaia DR3 (+), 1 known asteroid/comet, 1 flagged artifact, 1 NEW (yellow)')
        new = objects[-1]
        F = int(cfg.DISPLAY_UPSCALE)
        x = (19.5 - 0.003 * 3600 * np.cos(np.radians(-7.2)) / 6.15 + 0.5) * F            # 0.003 deg EAST: left
        self.assertAlmostEqual(new['x'], x, delta=0.6 * F)
        self.assertLess(new['y'], 20 * F)                                               # north: up
        self.assertFalse(objects[2]['detected'])                                        # Eros not seen in T1

    def shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            return web.write_frames(self.zooms, tmp, 40)

    def test_objects_and_how_to_track_them(self):
        objects = web.mapped_objects(self.results, self.shown())
        self.assertNotIn('star:far away', objects)                                       # outside the images
        self.assertEqual(objects['new:NEW-1']['frames'], [1, 2])
        self.assertEqual(objects['new:NEW-1']['track']['kind'], 'path')
        self.assertEqual(objects['small body:433']['detected_in'], [2])
        self.assertEqual(objects['small body:433']['track']['kind'], 'asteroid')
        self.assertEqual(objects['small body:433']['track']['id'], '433')
        star = objects['star:0']['track']
        self.assertEqual((star['kind'], star['star']['name']), ('star', 'Gaia DR3 1'))
        self.assertIsNone(objects['artifact:artifact: GHOST']['track'])
        self.assertIn('artifact', objects['artifact:artifact: GHOST']['reason'])
        json.dumps(objects)                                                             # browser-ready
        # what the browser sends back to track an object is accepted as it is
        for oid in ('new:NEW-1', 'small body:433', 'star:0'):
            request = web.web_request({**PAYLOAD, 'track': objects[oid]['track']})
            self.assertEqual(request['track']['kind'], objects[oid]['track']['kind'])

    def test_database_comparison(self):
        logged = ['(Hipparcos / Stellarium names not available: Hipparcos: all 4 attempts failed)']
        db = web.database_comparison(self.results, logged, web.mapped_objects(self.results, self.shown()))
        self.assertEqual(db['new_outside'], 0)
        self.assertTrue(db['sky_map'] and db['field_check'])
        self.assertEqual(db['catalog_stars'], {'Gaia': 2, 'Hipparcos': 1})
        self.assertEqual(db['pointing']['matches'], 3)
        self.assertAlmostEqual(db['pointing']['median_offset_arcsec'], 0.1 * 6.15)
        self.assertEqual(db['known_small_bodies'], [{'name': '433 Eros (A898 PA)', 'kind': 'an', 'orbit_class': 'Amor', 'frames': [2]}])
        self.assertEqual(db['new_objects'][0]['id'], 'NEW-1')
        self.assertEqual(db['new_objects'][0]['object'], 'new:NEW-1')
        self.assertEqual(db['notes'], ['Hipparcos / Stellarium names not available: Hipparcos: all 4 attempts failed'])

    def test_objects_outside_the_images_are_left_out(self):
        for zoom in self.zooms:                                       # NEW-1 moved 3' away: in the margin only
            zoom['map'][4] = {**zoom['map'][4], 'ra': 133.85, 'dec': -7.15}
        objects = web.mapped_objects(self.results, self.shown())
        self.assertNotIn('new:NEW-1', objects)
        db = web.database_comparison(self.results, [], objects)
        self.assertEqual((db['new_objects'], db['new_outside']), ([], 1))

    def test_track_requests(self):
        star = {'kind': 'star', 'name': 'Gaia DR3 1', 'star': {'name': 'Gaia DR3 1', 'ra': 10.0, 'dec': 5.0, 'pmra': 3.0,
                                                               'pmdec': -1.0, 'parallax': 2.0}}
        path = {'kind': 'path', 'name': 'NEW-1', 'mjd': [60800.0, 60801.0], 'ra': [1.0, 1.1], 'dec': [2.0, 2.0]}
        asteroid = {'kind': 'asteroid', 'name': '433 Eros', 'id': '433', 'window': ['2025-06-01', '2025-08-30']}
        coords = {'kind': 'coords', 'name': 'NEW-2', 'coords': [10.5, -3.0]}
        calls = []
        fake = mock.Mock()
        fake.run_fixed_target.side_effect = lambda star, settings, n, show: calls.append(('fixed', star['ra'], star['pmra'], settings['mode'], n, show))
        fake.run_asteroid.side_effect = lambda id_, window, n, show: calls.append(('asteroid', id_, window, n, show))
        fake.run_path.side_effect = lambda name, mjd, ra, dec, n, show: calls.append(('path', name, len(mjd), n, show))
        for spec in (star, path, asteroid, coords):
            web.run(web.web_request({**PAYLOAD, 'max_images': 8, 'track': spec}), fake)
        web.run(web.web_request(PAYLOAD), fake)
        self.assertEqual(calls, [('fixed', 10.0, 3.0, 'catalog star', 8, False), ('path', 'NEW-1', 2, 8, False),
                                 ('asteroid', '433', ('2025-06-01', '2025-08-30'), 8, False),
                                 ('fixed', 10.5, 0.0, 'web', 8, False), ('fixed', 133.795129, 0.0, 'web', 24, False)])
        for bad in ({'kind': 'teleport'}, {**asteroid, 'window': ['soon', 'later']}, {**path, 'ra': [1.0]},
                    {**coords, 'coords': [500, 0]}, {'kind': 'star', 'name': 'x'}, 'NEW-1'):
            with self.assertRaises(ValueError):
                web.web_request({**PAYLOAD, 'track': bad})


class Endpoints(unittest.TestCase):
    """Flask test client: the three new routes, with the pipeline run replaced by fake_analyse (no internet)."""

    def setUp(self):
        self.client = app.test_client()
        self.tmp = tempfile.TemporaryDirectory()
        patches = [mock.patch.object(web, 'results_folder', lambda: Path(self.tmp.name)),
                   mock.patch.object(web, 'analyse', fake_analyse)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        analysis_jobs._jobs.clear()

    def test_existing_pages_still_answer(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertIn(b'id="tlPlayer"', self.client.get('/').data)
        self.assertNotIn(b'coming soon', self.client.get('/').data)
        self.assertNotIn(b'id="anaMap"', self.client.get('/').data)                # only the analysed images are shown
        self.assertNotIn(b'id="anaSB"', self.client.get('/').data)                 # no predicted (JPL) list either
        self.assertIn(b'id="tlSvg"', self.client.get('/').data)                    # the sky map / circles overlay
        page = self.client.get('/').data
        for element in (b'id="tlLoad"', b'id="tlTip"', b'id="anaMin"', b'id="anaMini"', b'class="anaLeft"', b'class="anaRight"',
                        b'id="tlFirst"', b'id="tlPrev"', b'id="tlNext"', b'id="tlLast"', b'id="tlSpeed"'):
            self.assertIn(element, page)                                            # loading card, minimize, layout, controls
        self.assertIn(b'slight display enhancements', page)                         # the image disclaimer
        self.assertIn(b'can make mistakes', page)
        self.assertIn(b'up to about a minute for low image counts', page)            # the timing notice
        self.assertIn(b'up to about 3 minutes for high counts', page)
        self.assertEqual(self.client.get('/about').status_code, 200)
        self.assertEqual(self.client.get('/no/such/page').status_code, 404)
        self.assertEqual(self.client.get('/api/no-such').get_json(), {'error': 'not found'})
        self.assertEqual(self.client.get('/api/meta').status_code, 200)

    def test_bad_input_is_rejected(self):
        for body in ({}, {**PAYLOAD, 'target': {'ra_deg': 400, 'dec_deg': 0}}, {**PAYLOAD, 'max_images': 0},
                     {**PAYLOAD, 'patch': {'half_width_deg': 'wide'}}):
            reply = self.client.post('/api/analysis', json=body)
            self.assertEqual(reply.status_code, 400, body)
            self.assertIn('error', reply.get_json())
        self.assertEqual(self.client.post('/api/analysis', data='not json').status_code, 400)
        self.assertEqual(self.client.post('/api/analysis', json={**PAYLOAD, 'track': {'kind': 'teleport'}}).status_code, 400)
        self.assertEqual(self.client.get('/api/analysis').status_code, 405)
        self.assertEqual(self.client.get('/api/analysis/0123456789abcdef').status_code, 404)        # unknown job
        self.assertEqual(self.client.get('/api/analysis/..%2F..%2Fconfig').status_code, 404)
        self.assertEqual(self.client.get('/api/analysis/0123456789abcdef/frame_1.png').status_code, 404)
        self.assertEqual(self.client.get('/api/analysis/0123456789abcdef/..%2Fsummary.png').status_code, 404)

    def test_run_poll_and_images(self):
        reply = self.client.post('/api/analysis', json={**PAYLOAD, 'max_images': 8})
        self.assertEqual(reply.status_code, 202)
        job = reply.get_json()['job']
        self.assertEqual(reply.get_json()['request']['max_images'], 8)
        state = wait_for(self.client, job)
        self.assertEqual(state['status'], 'done', state)
        summary = state['summary']
        self.assertEqual((summary['n_frames'], summary['images_available']), (2, 40))
        self.assertEqual(summary['frames'][1]['flags_in_aperture'], ['HOT'])
        for frame in summary['frames']:
            image = self.client.get(f"/api/analysis/{job}/{frame['png']}")
            self.assertEqual((image.status_code, image.mimetype), (200, 'image/png'))
            self.assertEqual(image.data[:8], b'\x89PNG\r\n\x1a\n')
            image.close()
        self.assertTrue((Path(self.tmp.name) / job / 'summary.json').exists())
        # the same request again: the same job, answered from the saved results (also after a restart)
        analysis_jobs._jobs.clear()
        again = self.client.post('/api/analysis', json={**PAYLOAD, 'max_images': 8}).get_json()['job']
        self.assertEqual(again, job)
        self.assertEqual(self.client.get(f'/api/analysis/{job}').get_json()['status'], 'done')
        # another number of images is another job
        other = self.client.post('/api/analysis', json={**PAYLOAD, 'max_images': 24}).get_json()['job']
        self.assertNotEqual(other, job)
        wait_for(self.client, other)

    def test_analysed_patches_for_the_viewer(self):
        """/api/analyses: every finished analysis, drawn on the sky by the viewer (and reopened from its request)."""
        self.assertEqual(self.client.get('/api/analyses').get_json(), {'analyses': []})
        job = self.client.post('/api/analysis', json={**PAYLOAD, 'max_images': 8}).get_json()['job']
        self.assertEqual(wait_for(self.client, job)['status'], 'done')
        moving = Path(self.tmp.name) / 'abcdef0123456789'                           # a tracked asteroid, saved earlier
        moving.mkdir()
        (moving / 'summary.json').write_text(json.dumps({
            'target': {'name': '433 Eros', 'ra_deg': 1.0, 'dec_deg': 2.0, 'moving': True}, 'tracking': '433 Eros',
            'request': {**web.web_request(PAYLOAD), 'track': {'kind': 'asteroid', 'name': '433 Eros', 'id': '433',
                                                              'window': ['2026-02-15', '2026-06-15']}},
            'patch_px': 50, 'pixel_arcsec': 6.15, 'n_frames': 3, 'images_available': 9,
            'frames': [{'time_utc': f'2026-03-0{k} 00:00:00', 'expected_ra': 100.0 + k, 'expected_dec': 10.0 + k} for k in (1, 2, 3)],
            'database': {'sky_map': False}}))
        (Path(self.tmp.name) / '0000000000000000').mkdir()                           # unreadable: skipped
        analyses = {a['job']: a for a in self.client.get('/api/analyses').get_json()['analyses']}
        self.assertEqual(set(analyses), {job, 'abcdef0123456789'})
        fixed, eros = analyses[job], analyses['abcdef0123456789']
        self.assertFalse(fixed['moving'])
        self.assertIsNone(fixed['path'])
        self.assertAlmostEqual(fixed['ra_deg'], 133.8, places=6)                    # the target's position in the images
        self.assertEqual((fixed['n_frames'], fixed['images_available'], fixed['patch_px']), (2, 40, 50))
        self.assertEqual(fixed['request'], web.web_request({**PAYLOAD, 'max_images': 8}))   # reopens the same job
        self.assertEqual(web_job(fixed['request']), job)
        self.assertTrue(eros['moving'])
        self.assertEqual(eros['path'], [[101.0, 11.0], [102.0, 12.0], [103.0, 13.0]])
        self.assertEqual((eros['ra_deg'], eros['dec_deg']), (102.0, 12.0))            # labelled mid-path
        self.assertEqual((eros['first_utc'], eros['last_utc'], eros['tracking']), ('2026-03-01 00:00:00', '2026-03-03 00:00:00', '433 Eros'))
        # what the browser sends to reopen the tracked one is accepted and is the same request
        request = eros['request']
        again = web.web_request({**PAYLOAD, 'max_images': request['max_images'], 'track': request['track']})
        self.assertEqual(again, request)

    def test_failure_becomes_a_message(self):
        def irsa_down(request, folder, on_progress=None):
            raise RuntimeError('SPHEREx image search: all 50 attempts failed (HTTP 502)')
        with mock.patch.object(web, 'analyse', irsa_down):
            job = self.client.post('/api/analysis', json=PAYLOAD).get_json()['job']
            state = wait_for(self.client, job)
        self.assertEqual(state['status'], 'error')
        self.assertIn('IRSA did not answer', state['error'])
        self.assertIn('try again', state['error'])
        with mock.patch.object(web, 'analyse', fake_analyse):                       # "Try again" then works
            self.assertEqual(wait_for(self.client, self.client.post('/api/analysis', json=PAYLOAD).get_json()['job'])['status'], 'done')


@unittest.skipUnless(os.environ.get('XARVIS_ONLINE'), 'needs the internet (NASA/IPAC IRSA): set XARVIS_ONLINE=1')
class RealRun(unittest.TestCase):
    """One real run through the website: WISE 0855-0714, 8 images."""

    def test_wise_0855(self):
        client = app.test_client()
        reply = client.post('/api/analysis', json={**PAYLOAD, 'max_images': 8})
        self.assertEqual(reply.status_code, 202)
        job = reply.get_json()['job']
        start = time.time()
        state = wait_for(client, job, seconds=900)
        self.assertEqual(state['status'], 'done', state.get('error'))
        summary = state['summary']
        print(f"\n  real run: {summary['n_frames']} frames of {summary['images_available']} images in "
              f"{time.time() - start:.0f} s -> {analysis_jobs.folder(job)}")
        for frame in summary['frames']:
            print(f"    {frame['n']:2d}  {frame['time_utc']}  detector {frame['detector']}  "
                  f"{frame['wavelength_um']:.3f} um  S/N {frame['snr']}  flags {frame['flags_in_aperture']}")
        self.assertGreaterEqual(summary['n_frames'], 1)
        self.assertLessEqual(summary['n_frames'], 8)
        self.assertTrue((analysis_jobs.folder(job) / 'summary.json').exists())
        for frame in summary['frames']:
            image = client.get(f"/api/analysis/{job}/{frame['png']}")
            self.assertEqual(image.status_code, 200)
            image.close()
            with Image.open(analysis_jobs.folder(job) / frame['png']) as png:
                self.assertEqual(png.size, (50 * int(cfg.DISPLAY_UPSCALE),) * 2)
            self.assertIsNotNone(frame['objects'])                                  # every frame has its sky map
        database = summary['database']
        print(f"  sky map: {len(summary['objects'])} objects in the images; compared with {database.get('catalog_stars')}; "
              f"positions vs Gaia: {database.get('pointing')}; NEW in the images: {len(database.get('new_objects', []))}")
        self.assertTrue(database['sky_map'] and database['field_check'])
        self.assertGreater(database['pointing']['matches'], 0)
        self.assertTrue(any(o['track'] for o in summary['objects'].values()))         # objects can be tracked


if __name__ == '__main__':
    unittest.main()
