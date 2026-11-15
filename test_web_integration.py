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


def wait_for(client, job, seconds=30):
    for _ in range(int(seconds * 20)):
        state = client.get(f'/api/analysis/{job}').get_json()
        if state['status'] in ('done', 'error'):
            return state
        time.sleep(0.05)
    raise AssertionError(f'job still {state["status"]} after {seconds} s')


def fake_zoom(mjd, detector, rotation_deg=0.0, star_east_px=0.0, size=60):
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
    image = 5 + rng.normal(0, 0.05, (size, size)) + 4 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 2)
    header = fits.Header()
    header['MSKN00'] = 'TRANSIENT'
    header['MSKN07'] = 'HOT'
    flags = np.zeros((size, size), np.int32)
    flags[int(centre[1]), int(centre[0])] = 1 << 7                   # a HOT pixel on the target
    return {'image': image, 'variance': np.full((size, size), 0.0025), 'bad': np.zeros((size, size), bool),
            'flags': flags, 'flags_header': header, 'wcs': w, 'catalog_xy': centre, 'mjd': mjd,
            'detector': detector, 'wavelength_um': 0.9 + detector / 2, 'snr': 12.5, 'flux_mjy': 1.0,
            'flux_err_mjy': 0.1, 'n_bad_in_aperture': 1, 'saturated': False, 'track': 'ok', 'astrometry_ok': True,
            'obs_id': f'fake_{mjd}'}


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

    def test_patch_size_is_read_at_run_time(self):
        frame = fake_zoom(60800.0, 1, size=120)
        frame['catalog_xy'] = (60.0, 60.0)
        self.assertEqual(pipeline.make_zooms([dict(frame)], False)[0]['image'].shape, (cfg.ZOOM_SIZE_PIX,) * 2)
        with settings_override({'ZOOM_SIZE_PIX': 80}):
            self.assertEqual(pipeline.make_zooms([dict(frame)], False)[0]['image'].shape, (80, 80))

    def test_web_settings_are_restored(self):
        before = {name: getattr(cfg, name) for name in web.run_settings(web.web_request(PAYLOAD))}
        with settings_override(web.run_settings(web.web_request({**PAYLOAD, 'patch': {'half_width_deg': 0.2}}))):
            self.assertTrue(cfg.SPREAD_IMAGES)
            self.assertFalse(cfg.FIELD_CHECK)
        self.assertEqual({name: getattr(cfg, name) for name in before}, before)


class Rendering(unittest.TestCase):
    """Offline: fake frames -> PNGs on one north-up grid, and the summary the browser reads."""

    def test_frames_line_up(self):
        zooms = [fake_zoom(60800.0, 1, rotation_deg=0.0, star_east_px=8), fake_zoom(60800.1, 2, 90.0, 8),
                 fake_zoom(60990.0, 4, 217.0, 8)]
        with tempfile.TemporaryDirectory() as tmp:
            shown = web.write_frames(zooms, tmp, 40)
            self.assertEqual([s['png'] for s in shown], ['frame_1.png', 'frame_2.png', 'frame_3.png'])
            for s in shown:
                pixels = np.asarray(Image.open(Path(tmp) / s['png']).convert('L'), dtype=float)
                self.assertEqual(pixels.shape, (40, 40))
                row, column = np.unravel_index(np.argmax(pixels), pixels.shape)
                # the star is 8 px EAST of the target: 8 px LEFT of the centre (19.5), same row - whatever the rotation
                self.assertLessEqual(abs(column - 11.5), 1, s['png'])
                self.assertLessEqual(abs(row - 19.5), 1, s['png'])
                self.assertEqual(s['flags_in_aperture'], ['HOT'])
                self.assertFalse(np.any(np.all(np.asarray(Image.open(Path(tmp) / s['png'])) == web.NO_DATA_RGB, axis=-1)),
                                 f"{s['png']}: empty pixels")                       # the 60 px frames cover the 40 px patch: no empty corners

    def test_summary(self):
        zooms = [fake_zoom(60800.0, 1), fake_zoom(60990.5, 3)]
        zooms[1]['snr'] = np.nan                                                    # not measurable -> null, not NaN
        request = web.web_request(PAYLOAD)
        with tempfile.TemporaryDirectory() as tmp:
            results = {'tracked': {'name': 'x', 'images_available': 7}, 'frames': zooms, 'zooms': zooms}
            summary = json.loads(json.dumps(web.summary_of(results, request, web.write_frames(zooms, tmp, 50))))
        self.assertEqual((summary['n_frames'], summary['images_available'], summary['patch_px']), (2, 7, 50))
        first, second = summary['frames']
        self.assertEqual(first['time_utc'], '2025-05-05 00:00:00')
        self.assertEqual((first['detector'], first['wavelength_um'], first['snr']), (1, 1.4, 12.5))
        self.assertIsNone(second['snr'])
        for key in ('n', 'png', 'mjd', 'flags_in_aperture', 'n_bad_in_aperture', 'saturated'):
            self.assertIn(key, first)

    def test_no_data_is_not_black(self):
        rgb = web.to_rgb(np.array([[0.0, np.nan], [1.0, 5.0]]))
        self.assertEqual(tuple(rgb[0, 1]), web.NO_DATA_RGB)
        self.assertEqual(rgb.dtype, np.uint8)


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
                self.assertEqual(png.size, (50, 50))


if __name__ == '__main__':
    unittest.main()
