"""Figures: a frame viewer for every image, movement and brightness over time, and the spectrum check."""

import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Slider

from . import config as cfg
from .frames import pause_measuring, zoom_original
from .irsa import fetch_cutout
from .measure import bad_pixel_mask, reliable_frames, robust_scatter, usable_frames
from .visual import PALETTE_STOPS, anchors, colour_of, find_partners, render, target_xy

MAX_GRID_PANELS = 30        # show_frames draws one panel per frame; for more frames use show_viewer
WAVELENGTH_RANGE = (0.75, 5.0)   # SPHEREx's wavelengths (um): fixed color scale for the live view


class LiveView:
    """A window that fills in while the frames are still downloading:
      left         the newest frame's zoom (original pixels; red = where the object was found, blue dashed =
                   where it was expected: the star's catalog position or the asteroid's predicted position)
      top right    brightness of every frame so far, against time (colored by wavelength)
      bottom right an asteroid's path on the sky (along the JPL Horizons line), or a star's distance from its
                   catalog position
    Pass `update` to load_all_frames(on_frame=...). It redraws at most every LIVE_REFRESH_SECONDS so drawing
    never slows the processing down; close() it when the full analysis is ready."""

    def __init__(self, tracked, total, t0_mjd):
        self.tracked, self.total, self.t0 = tracked, total, t0_mjd
        self.moving = 'ephemeris' in tracked
        self.frames, self.latest, self.done = [], None, 0
        self.start, self.last_draw, self.dirty = time.time(), 0.0, False

        self.fig = plt.figure(figsize=(14, 6.5))
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.35], hspace=0.5)
        self.ax_img = self.fig.add_subplot(grid[:, 0])
        self.ax_flux = self.fig.add_subplot(grid[0, 1])
        self.ax_pos = self.fig.add_subplot(grid[1, 1])
        self.fig.colorbar(ScalarMappable(Normalize(*WAVELENGTH_RANGE), 'turbo'), ax=self.ax_flux, label='Wavelength (µm)')
        plt.show(block=False)
        self._draw()

    def update(self, frame, done, total):
        """A newly measured frame (or None while waiting): redraw if it's time, and keep the window responsive."""
        self.done = done
        if frame is not None:
            self.frames.append(frame)
            self.latest, self.dirty = frame, True
        if self.dirty and time.time() - self.last_draw >= cfg.LIVE_REFRESH_SECONDS:
            self._draw()
        else:
            self.fig.canvas.flush_events()                      # let the window move / resize between redraws

    def close(self):
        plt.close(self.fig)        # (no last redraw: the frames' full pixels may already have been released)

    def _draw(self):
        with pause_measuring():                             # the measuring threads would starve the drawing
            self._draw_now()

    def _draw_now(self):
        for ax in (self.ax_img, self.ax_flux, self.ax_pos):
            ax.clear()
        if self.latest is not None:
            self._draw_newest(self.latest)
            self._draw_series()
        else:
            self.ax_img.text(0.5, 0.5, 'waiting for the first frame...', ha='center', transform=self.ax_img.transAxes)
        elapsed = time.time() - self.start
        self.fig.suptitle(f"{self.tracked['name']}: {self.done} of {self.total} images processed   "
                          f"({self.done / max(elapsed, 1e-9):.1f}/s, {elapsed:.0f} s)   - full analysis follows")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.last_draw, self.dirty = time.time(), False

    def _draw_newest(self, frame):
        zoom = zoom_original(frame)                         # full pixels are still in memory at this point
        pixels = zoom['image'][np.isfinite(zoom['image'])]
        sky, noise = np.median(pixels), robust_scatter(pixels)
        self.ax_img.imshow(zoom['image'], origin='lower', cmap='gray', vmin=sky - 3 * noise, vmax=sky + 15 * noise,
                           interpolation='nearest')
        for (ra, dec), style in (((frame['expected_ra'], frame['expected_dec']), dict(edgecolor='deepskyblue', ls='--')),
                                 ((frame['object_ra'], frame['object_dec']), dict(edgecolor='red'))):
            x, y = zoom['wcs'].world_to_pixel_values(ra, dec)
            self.ax_img.add_patch(Circle((float(x), float(y)), cfg.APERTURE_RADIUS_PIX, facecolor='none', lw=1.3, **style))
        self.ax_img.set(xticks=[], yticks=[])
        self.ax_img.set_title(f"newest: {frame['date_obs'][:16].replace('T', ' ')}   detector {frame['detector']}   "
                              f"λ = {frame['wavelength_um']:.3f} µm\nflux {frame['flux_mjy']:.1f} mJy   "
                              f"SNR {frame['snr']:.0f}", fontsize=9)

    def _draw_series(self):
        frames = self.frames
        days = np.array([f['mjd'] for f in frames]) - self.t0
        wavelength = np.array([f['wavelength_um'] for f in frames])
        color = dict(c=wavelength, cmap='turbo', vmin=WAVELENGTH_RANGE[0], vmax=WAVELENGTH_RANGE[1], s=12)
        self.ax_flux.scatter(days, [f['flux_mjy'] for f in frames], **color)
        self.ax_flux.set(xlabel='Days since first image', ylabel='Flux density (mJy)',
                         title=f'Brightness ({len(frames)} frames so far)')

        ra, dec = np.array([f['object_ra'] for f in frames]), np.array([f['object_dec'] for f in frames])
        if self.moving:                                     # an asteroid: its path on the sky, along the prediction
            ephemeris = self.tracked['ephemeris']
            mjd = days + self.t0
            on_path = np.degrees(np.interp(mjd, ephemeris.mjd, ephemeris.ra))
            ra = on_path + (ra - on_path + 180) % 360 - 180   # same 360-degree branch as the path line
            self.ax_pos.plot(np.degrees(ephemeris.ra), ephemeris.dec, color='0.8', lw=2, zorder=1)
            self.ax_pos.scatter(ra, dec, zorder=2, **color)
            self.ax_pos.invert_xaxis()
            self.ax_pos.set(xlabel='RA (degrees)', ylabel='Dec (degrees)', title='Path on the sky (gray = JPL Horizons)')
        else:                                               # a star: how far from its catalog position
            expected_ra = np.array([f['expected_ra'] for f in frames])
            expected_dec = np.array([f['expected_dec'] for f in frames])
            d_ra = ((ra - expected_ra + 180) % 360 - 180) * np.cos(np.radians(dec))   # RA difference on the sky
            offset = np.hypot(d_ra, dec - expected_dec) * 3600
            self.ax_pos.scatter(days, offset, **color)
            self.ax_pos.set(xlabel='Days since first image', ylabel='Offset from catalog position (arcsec)',
                            title='Position')


def fmt_angle(arcsec):
    """An angle for a title: arcseconds below 1', arcminutes below 1 degree, then degrees."""
    if arcsec < 60:
        return f'{arcsec:.2f}"'
    return f"{arcsec / 60:.1f}'" if arcsec < 3600 else f'{arcsec / 3600:.2f}°'


def draw_zoom(ax, zoom, index, total, t0, movement, brightness=None, compact=False, pretty=False, partner=None,
              anchor=None, factor=None, marks=None):
    """Draw one frame's zoom into `ax`.
    pretty=False: the original pixels in grayscale, each pixel a sharp square (what was measured).
    pretty=True: display-only colour (from the comparison with `partner`, the simultaneous frame at another
    wavelength - see visual.py), smoothed and upscaled. The measurements are not affected either way.
    Red circle = where the object was found in this frame. Blue dashed circle = the reference position:
    where a star was at T1, or where the ephemeris predicted an asteroid (drawn only if it is in the zoom).
    `t0` = time of T1 (MJD). `brightness` = the star's brightness analysis (None for an asteroid).
    `compact` = shorter title, for the grid. `anchor` (pretty only, see visual.anchors): draw the target at the
    same brightness and in the same colour in every frame. `marks` = {'red': (ra, dec) or None, 'blue': ...} puts
    the two circles on another object instead (one selected in the viewer). Returns the image (for a color bar)."""
    if pretty:
        rgb, extent = render(zoom, partner, factor=factor, anchor=anchor)
        im = ax.imshow(rgb, origin='lower', extent=extent, interpolation='bilinear')
    else:
        # each frame is at its own wavelength, so each gets its own brightness scale (sky - 3 / + 15 noise)
        pixels = zoom['image'][np.isfinite(zoom['image'])]
        sky, noise = np.median(pixels), robust_scatter(pixels)
        im = ax.imshow(zoom['image'], origin='lower', cmap='gray', vmin=sky - 3 * noise, vmax=sky + 15 * noise,
                       interpolation='nearest')       # each original pixel as a sharp square
    # circles placed by sky position, converted to this frame's pixels (so they stay round)
    if marks is None:                                       # the analysed object
        reference = movement['reference']
        reference = reference[index] if reference.shape else reference      # one position per frame, or one for all
        blue = (float(reference.ra.deg), float(reference.dec.deg))
        red = (zoom['object_ra'], zoom['object_dec'])
    else:                                                   # an object selected in the viewer (see show_viewer)
        blue, red = marks.get('blue'), marks.get('red')
    ny, nx = zoom['image'].shape
    if blue is not None:
        ref_x, ref_y = (float(v) for v in zoom['wcs'].world_to_pixel_values(*blue))
        if 0 <= ref_x < nx and 0 <= ref_y < ny:
            ax.add_patch(Circle((ref_x, ref_y), cfg.APERTURE_RADIUS_PIX + 0.6,
                                edgecolor='deepskyblue', facecolor='none', lw=1.5, ls='--', zorder=9))
    if red is not None:
        now_x, now_y = (float(v) for v in zoom['wcs'].world_to_pixel_values(*red))
        ax.add_patch(Circle((now_x, now_y), cfg.APERTURE_RADIUS_PIX, edgecolor='red', facecolor='none', lw=1.3,
                            zorder=9))

    moved = movement['flagged'][index]
    changed = bool(brightness['flagged'][index]) if brightness else False
    flag_label = movement.get('flag_label', 'MOVED')
    date = zoom['date_obs'][:16].replace('T', ' ')
    days = zoom['mjd'] - t0
    n_bad = zoom['n_bad_in_aperture']
    traveled = f"moved {fmt_angle(movement['offset_arcsec'][index])}"
    if compact:
        marks = (f'  {flag_label}' if moved else '') + ('  BRIGHTNESS' if changed else '') + ('  [bad px]' if n_bad else '')
        ax.set_title(f"T{index + 1}  {date}\n+{days:.2f} d   λ = {zoom['wavelength_um']:.3f} µm\n{traveled}{marks}",
                     fontsize=8)
    else:
        move_flag = f'  << {flag_label}' if moved else ''
        unreliable = f"   [{n_bad} bad pixel(s) in circle: not used]" if n_bad else ''
        if 'notes' in movement:                             # an asteroid: signal-to-noise and distance from ephemeris
            third = movement['notes'][index]
        else:                                               # a star: brightness against its own spectrum
            deviation = brightness['deviation'][index]      # NaN when too few frames at a similar wavelength
            versus = f'{deviation * 100:+.1f}%' if np.isfinite(deviation) else 'n/a (too few similar-wavelength frames)'
            third = f"vs spectrum {versus}{'  << BRIGHTNESS CHANGE' if changed else ''}"
        if zoom.get('saturated'):
            unreliable = ''
            deviation = brightness['deviation'][index] if brightness and brightness.get('source') == 'glow' else np.nan
            brightness_text = ('SATURATED: position from its outer glow, ' +
                               (f'glow vs spectrum {deviation * 100:+.1f}%{"  << BRIGHTNESS CHANGE" if changed else ""}'
                                if np.isfinite(deviation) else 'brightness not measurable'))
        else:
            brightness_text = f"flux {zoom['flux_mjy']:.2f} ± {zoom['flux_err_mjy']:.2f} mJy   {third}"
        ax.set_title(f"T{index + 1} of {total}   {date}   +{days:.2f} d   detector {zoom['detector']}{unreliable}\n"
                     f"λ = {zoom['wavelength_um']:.3f} µm   {traveled}{move_flag}\n{brightness_text}", fontsize=9)
    return im


def circle_legend(movement):
    """Legend handles for the two circles (the blue one means different things for a star and an asteroid)."""
    return [Line2D([], [], color='deepskyblue', ls='--', label=movement['reference_label']),
            Line2D([], [], color='red', label='object found in this frame')]


def show_frames(zooms, tracked, movement, brightness=None, return_axes=False, pretty=None):
    """All zooms side by side in a grid, one panel per frame (for a handful of frames, e.g.
    run(max_images=10); up to MAX_GRID_PANELS). Each panel is scaled to its own sky level.
    pretty: colour / smooth / upscale for display (default config.PRETTY_IMAGES; see draw_zoom).

    return_axes=True returns (fig, zoom_axes) instead of just fig: zoom_axes[i] is the axes of frame T{i+1},
    so the caller can add its own overlays to the panels."""
    n = len(zooms)
    if n > MAX_GRID_PANELS:
        raise ValueError(f'{n} frames is too many for a grid (max {MAX_GRID_PANELS}) - use show_viewer instead')
    pretty = cfg.PRETTY_IMAGES if pretty is None else pretty
    partners = find_partners(zooms) if pretty else [None] * n
    frame_anchors = anchors(zooms, partners) if pretty else [None] * n
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(3.8 * n_cols, 4.4 * n_rows))
    if pretty:
        fig.patch.set_facecolor('0.12')

    zoom_axes = []
    for i, zoom in enumerate(zooms):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=zoom['wcs'])
        partner = zooms[partners[i]] if partners[i] is not None else None
        draw_zoom(ax, zoom, i, n, zooms[0]['mjd'], movement, brightness, compact=True, pretty=pretty, partner=partner,
                  anchor=frame_anchors[i])
        if pretty:
            _dark_axes(ax)
        ax.coords[0].set_axislabel('RA' if i // n_cols == n_rows - 1 else ' ')
        ax.coords[1].set_axislabel('Dec' if i % n_cols == 0 else ' ')
        ax.coords[0].set_ticklabel(size=7)
        ax.coords[1].set_ticklabel(size=7)
        zoom_axes.append(ax)

    fig.legend(handles=circle_legend(movement), loc='upper right', fontsize=9)
    fig.suptitle(f"SPHEREx: {tracked['name']}   ({n} frames, "
                 + ('colour + smoothing for display only)' if pretty else 'original pixels)'),
                 color='white' if pretty else 'black')
    return (fig, zoom_axes) if return_axes else fig


def _dark_axes(ax):
    """White labels / ticks for panels on a dark figure background (and black where there is no image)."""
    ax.set_facecolor('black')
    ax.title.set_color('white')
    for coord in (ax.coords[0], ax.coords[1]):
        coord.set_ticklabel(color='white')
        coord.set_axislabel(coord.get_axislabel(), color='white')
        coord.set_ticks(color='0.7')


def _colour_key(cax, zoom, partner):
    """What the colours mean for this frame: a strip of the palette, labelled with the two wavelengths compared."""
    cax.clear()
    strip = colour_of(np.linspace(PALETTE_STOPS[0], PALETTE_STOPS[-1], 128))[:, None, :]
    cax.imshow(strip, origin='lower', aspect='auto', extent=(0, 1, PALETTE_STOPS[0], PALETTE_STOPS[-1]))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.tick_params(colors='white', labelsize=7)
    if partner is None:
        cax.set_yticks([0])
        cax.set_yticklabels(['no simultaneous\nframe: brightness\nonly'])
        return
    short, long = sorted((zoom['wavelength_um'], partner['wavelength_um']))
    cax.set_yticks([PALETTE_STOPS[0], 0, PALETTE_STOPS[-1]])
    cax.set_yticklabels([f'bluer: brighter\nat {short:.2f} µm', 'typical\nstar', f'redder: brighter\nat {long:.2f} µm'])


def load_wide(zoom, moving=False):
    """The wide view of a frame: a WIDE_VIEW_SIZE cutout of the same SPHEREx image around the target, downloaded
    on demand (cached on disk). Returned as a copy of the zoom with the wide pixels and coordinates in place of
    the zoomed ones, so it draws exactly like a zoom. Display only - nothing is measured on it."""
    centre = SkyCoord(zoom['expected_ra'], zoom['expected_dec'], unit='deg')
    with fetch_cutout(zoom['access_url'], centre, cfg.WIDE_VIEW_SIZE * u.arcmin) as hdulist:
        header = hdulist['IMAGE'].header
        image = hdulist['IMAGE'].data.astype(float) - hdulist['ZODI'].data
        flags = hdulist['FLAGS'].data.astype(np.int64)
        bad = bad_pixel_mask(image, flags, hdulist['FLAGS'].header, moving)
        variance = hdulist['VARIANCE'].data.astype(float)
    return dict(zoom, image=image, bad=bad, flags=flags, variance=variance, wcs=WCS(header), wide=True)


def _footprint(ax, display, zoom):
    """The zoom's outline, drawn on the wide view."""
    ny, nx = zoom['image'].shape
    corners_x, corners_y = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5]), np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    ra, dec = zoom['wcs'].pixel_to_world_values(corners_x, corners_y)
    x, y = display['wcs'].world_to_pixel_values(ra, dec)
    ax.add_patch(Polygon(np.c_[x, y], closed=True, edgecolor='0.85', facecolor='none', lw=0.9, ls=':'))


def overlay_catalog(ax, display, field, bodies=None):
    """Gaia DR3 / CatWISE sources (moved to this frame's date) on the image, and the SPHEREx sources that NO
    catalog explains; plus known asteroids / comets from JPL (`bodies`, see field.known_small_bodies).
    Returns a one-line summary."""
    from .field import compare_frame
    result = compare_frame(display, field)
    small = display['image'].shape[0] > 2 * cfg.ZOOM_SIZE_PIX            # wide view: smaller markers
    plus, cross = (12, 9) if small else (30, 22)
    # only the catalog sources SPHEREx actually detects here (a crowded field has thousands of fainter ones)
    gaia = (result['cat_source'] == 'Gaia') & result['cat_seen']
    wise_only = (result['cat_source'] == 'CatWISE') & ~result['cat_gaia_match'] & result['cat_seen']
    ax.scatter(result['cat_x'][gaia], result['cat_y'][gaia], marker='+', s=plus, color='#4fd2ff', lw=0.9, zorder=5)
    ax.scatter(result['cat_x'][wise_only], result['cat_y'][wise_only], marker='x', s=cross, color='#ff9a3c', lw=0.9,
               zorder=5)
    unknown = ~result['matched'] & result['peak_clean'] & (result['peak_sigma'] >= cfg.FIELD_MIN_SIGMA)
    for x, y in zip(result['peak_x'][unknown], result['peak_y'][unknown]):
        ax.add_patch(Circle((x, y), 1.8, edgecolor='yellow', facecolor='none', lw=1.2, zorder=6))
    text = (f"SPHEREx sources: {int(result['matched'].sum())} known - Gaia DR3 (+) {int(gaia.sum())}, "
            f"CatWISE only (x) {int(wise_only.sum())}; in no catalog (yellow): {int(unknown.sum())}")
    if bodies:
        ra, dec = np.array([b['ra'] for b in bodies]), np.array([b['dec'] for b in bodies])
        x, y = display['wcs'].world_to_pixel_values(ra, dec)
        ny, nx = display['image'].shape
        inside = (x > -0.5) & (x < nx - 0.5) & (y > -0.5) & (y < ny - 0.5)
        ax.scatter(x[inside], y[inside], marker='D', s=40, facecolor='none', edgecolor='magenta', lw=1.2, zorder=7)
        for k in np.flatnonzero(inside):
            ax.annotate(bodies[k]['name'].split('(')[0].strip(), (x[k], y[k]), xytext=(5, 5), textcoords='offset points',
                        color='magenta', fontsize=7, zorder=7)
        text += f'; JPL: {int(inside.sum())} known asteroid/comet(s) (magenta)'
    elif bodies is not None:
        text += '; JPL: no known asteroid/comet here'
    ny, nx = display['image'].shape
    ax.set_xlim(-0.5, nx - 0.5)                            # markers just outside mustn't widen the view
    ax.set_ylim(-0.5, ny - 0.5)
    return text


_track_request = {}


def request_track(spec):
    """Ask main to track this object next (after the current windows close)."""
    _track_request['spec'] = spec


def take_track_request():
    """The object picked in the viewer to track next (see request_track), or None; clears it."""
    return _track_request.pop('spec', None)


MAP_STYLE = {   # how each kind of mapped object is drawn (see skymap.py)
    'new': dict(color='#ffe600'), 'small body': dict(color='magenta'), 'planet': dict(color='#7CFC00'),
    'artifact': dict(color='0.6'), 'Gaia': dict(color='#4fd2ff'), 'CatWISE': dict(color='#ff9a3c'),
    'Hipparcos': dict(color='white'),
}
NEW_MARKER_SIZE = 420           # points^2: the ring around a NEW object


def map_legend():
    """Legend handles for the sky map: what every marker means."""
    def handle(label, **style):
        return Line2D([], [], linestyle='none', label=label, **style)
    new = MAP_STYLE['new']['color']
    return [handle('NEW object: in no catalog\n(not previously mapped)', marker='o', markersize=14, markerfacecolor='none',
                   markeredgecolor=new, markeredgewidth=2.6),
            handle('named star (Hipparcos +\nStellarium name)', marker='*', markersize=10, markerfacecolor='none',
                   markeredgecolor='white'),
            handle('Gaia DR3 star', marker='+', markersize=8, color=MAP_STYLE['Gaia']['color']),
            handle('CatWISE only (red / dusty)', marker='x', markersize=7, color=MAP_STYLE['CatWISE']['color']),
            handle('known asteroid / comet (JPL)', marker='D', markersize=7, markerfacecolor='none',
                   markeredgecolor=MAP_STYLE['small body']['color']),
            handle('planet / Moon (DE440s)', marker='D', markersize=7, markerfacecolor='none',
                   markeredgecolor=MAP_STYLE['planet']['color']),
            handle('flagged detector artifact', marker='s', markersize=7, markerfacecolor='none',
                   markeredgecolor=MAP_STYLE['artifact']['color'])]


def draw_map(ax, display, entries):
    """The sky map of one frame (frame['map'], see skymap.map_frames) on `display`: every star with its catalog
    (named / Hipparcos stars labelled), known asteroids / comets and planets (labelled), flagged artifacts, and NEW
    objects (yellow, labelled with what they probably are). Returns [(x, y, entry)] of what was drawn and a summary."""
    ny, nx = display['image'].shape
    small = ny > 2 * cfg.ZOOM_SIZE_PIX
    drawn, counts = [], {}
    for entry in entries:
        x, y = (float(v) for v in display['wcs'].world_to_pixel_values(entry['ra'], entry['dec']))
        if not (-0.5 <= x < nx - 0.5 and -0.5 <= y < ny - 0.5):
            continue
        kind = entry['kind']
        key = entry.get('source', kind) if kind == 'star' else kind
        counts[key] = counts.get(key, 0) + 1
        color = MAP_STYLE.get(key, {'color': 'white'})['color']
        if kind == 'star':
            if entry.get('named'):
                ax.scatter([x], [y], marker='*', s=60 if not small else 25, facecolor='none', edgecolor='white', lw=0.9, zorder=6)
                ax.annotate(entry['label'], (x, y), xytext=(6, 4), textcoords='offset points', color='white',
                            fontsize=7, zorder=8)
            else:
                ax.scatter([x], [y], marker='+' if entry['source'] == 'Gaia' else 'x', s=12 if small else 28,
                           color=color, lw=0.9, zorder=5)
        elif kind == 'artifact':
            ax.scatter([x], [y], marker='s', s=30, facecolor='none', edgecolor=color, lw=0.8, zorder=5)
        elif kind == 'new':                                 # not in ANY catalog: impossible to miss
            # (sizes in screen points, so it is just as visible in the wide view)
            ax.scatter([x], [y], s=NEW_MARKER_SIZE * 2.2, facecolor='none', edgecolor=color, lw=7, alpha=0.25, zorder=7)
            ax.scatter([x], [y], s=NEW_MARKER_SIZE, facecolor='none', edgecolor=color, lw=2.6, zorder=8)
            ax.annotate(entry['label'].replace(' (', '\n('), (x, y), xytext=(14, 10), textcoords='offset points',
                        color='black', fontsize=8, fontweight='bold', zorder=9,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='none', alpha=0.95),
                        arrowprops=dict(arrowstyle='-', color=color, lw=1.4))
        else:                                               # small body / planet
            ax.scatter([x], [y], marker='D', s=45, facecolor='none', edgecolor=color, lw=1.3, zorder=7)
            name = entry['label'].split('(')[0].strip() or entry['label']
            ax.annotate(name + ('' if entry.get('detected') else ' (not detected)'), (x, y), xytext=(6, 5),
                        textcoords='offset points', color=color, fontsize=7, zorder=8)
        drawn.append((x, y, entry))
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    parts = [f"{counts[k]} {label}" for k, label in (('Hipparcos', 'Hipparcos'), ('Gaia', 'Gaia DR3 (+)'),
             ('CatWISE', 'CatWISE (x)'), ('small body', 'known asteroid/comet'), ('planet', 'planet'),
             ('artifact', 'flagged artifact'), ('new', 'NEW (yellow)')) if counts.get(k)]
    return drawn, 'mapped: ' + (', '.join(parts) or 'nothing detected') + '   - click an object, T = track it'


def _object_key(entry):
    """What identifies one mapped object across frames (its catalog row, JPL designation or NEW id)."""
    kind = entry['kind']
    return kind, {'star': entry.get('catalog_row'), 'small body': entry.get('body'),
                  'new': entry.get('new_id')}.get(kind, entry['label'])


def object_positions(zooms, entry):
    """{frame index: (ra, dec)} of the mapped object `entry` in every frame where the sky map has it."""
    key, positions = _object_key(entry), {}
    for i, zoom in enumerate(zooms):
        for other in zoom.get('map', []):
            if _object_key(other) == key:
                positions[i] = (other['ra'], other['dec'])
                break
    return positions


def track_spec(entry, tracked, frame_mjd):
    """How to track a mapped object (for request_track), or a spec with 'reason' if it can't be tracked."""
    sky = tracked.get('skymap') or {}
    kind = entry['kind']
    if kind == 'star':
        from .skymap import star_to_track
        return {'kind': 'star', 'star': star_to_track(sky['catalog'][entry['catalog_row']]), 'name': entry['label']}
    if kind == 'small body':
        from astropy.time import Time
        window = [Time(frame_mjd - 45, format='mjd').iso[:10], Time(frame_mjd + 45, format='mjd').iso[:10]]
        return {'kind': 'asteroid', 'id': entry['body'], 'window': window, 'name': entry['label']}
    if kind == 'new':
        for obj in sky.get('new_objects', []):
            if obj['id'] == entry['new_id']:
                return obj['track']
    if kind == 'artifact':
        return {'kind': 'none', 'name': entry['label'], 'reason': 'a flagged detector artifact, not a source'}
    return {'kind': 'none', 'name': entry['label'], 'reason': 'planets are far too bright for SPHEREx to measure'}


def show_viewer(zooms, tracked, movement, brightness=None, pretty=None):
    """Step through every frame. Red circle = where the object was found in this frame; blue dashed circle =
    the reference position (see draw_zoom). pretty (default config.PRETTY_IMAGES): colour, sharpening and
    upscaling for display only - the target is drawn at the same brightness and colour in every frame;
    otherwise the original pixels in grayscale.
    Controls: slider, left/right arrow keys, F = jump to the next flagged frame,
              W = wide view (WIDE_VIEW_SIZE arcmin of the same image, downloaded on demand) / back to the zoom,
              G = Gaia DR3 + CatWISE overlay (and SPHEREx sources no catalog explains),
              J = ask JPL which known asteroids / comets are in view (~20 s, in the background).
    With a sky map (coordinates: tracked['skymap']), G shows it - every object named, NEW ones in yellow; click an
    object to see what it is, then T to track it (the windows close and its analysis starts)."""
    from .field import identified, identify_async, query_field
    n = len(zooms)
    pretty = cfg.PRETTY_IMAGES if pretty is None else pretty
    partners = find_partners(zooms) if pretty else [None] * n
    frame_anchors = anchors(zooms, partners) if pretty else [None] * n
    colour = frame_anchors[0]['colour'] if pretty else np.nan
    moving = 'ephemeris' in tracked
    flagged = np.flatnonzero(movement['flagged'] | (brightness['flagged'] if brightness else False))
    plt.rcParams['keymap.grid'] = [k for k in plt.rcParams['keymap.grid'] if k != 'g']   # G is ours here
    has_map = any('map' in z for z in zooms)
    if has_map:     # wider window: the sky-map key gets its own column on the right
        layout = {'size': (11.5, 9.4), 'image': [0.08, 0.15, 0.56, 0.7], 'cbar': [0.66, 0.2, 0.02, 0.6],
                  'slider': [0.11, 0.035, 0.53, 0.025], 'status_x': 0.37, 'legend_x': 0.775}
    else:
        layout = {'size': (8.5, 9.4), 'image': [0.12, 0.15, 0.68, 0.7], 'cbar': [0.86, 0.2, 0.025, 0.6],
                  'slider': [0.14, 0.035, 0.66, 0.025], 'status_x': 0.5, 'legend_x': None}
    fig = plt.figure(figsize=layout['size'])
    if pretty:
        fig.patch.set_facecolor('0.12')
    slider = Slider(fig.add_axes(layout['slider']), 'Frame', 1, n, valinit=1, valstep=1, valfmt='T%d')
    cbar_ax = fig.add_axes(layout['cbar'])
    status = fig.text(layout['status_x'], 0.075, '', ha='center', fontsize=8, color='0.8' if pretty else '0.3')
    state = {'ax': None, 'i': 0, 'wide': False, 'catalog': has_map, 'waiting': set(), 'drawn': [], 'selected': None,
             'selected_positions': {}}
    pool = ThreadPoolExecutor(max_workers=3)
    jobs = {}

    def job(key, func, *args):
        if key not in jobs:
            jobs[key] = pool.submit(func, *args)
        return jobs[key]

    def result(key):
        """A finished job's answer; None while running; False if it failed."""
        future = jobs.get(key)
        if future is None or not future.done():
            return None
        if future.exception() is not None:
            return False
        return future.result()

    def draw(i):
        state['i'], state['waiting'] = i, set()
        zoom = zooms[i]
        partner = zooms[partners[i]] if partners[i] is not None else None
        display, display_partner, factor, notes = zoom, partner, None, []
        if state['wide']:
            keys = [('wide', i)] + ([('wide', partners[i])] if partner is not None else [])
            job(('wide', i), load_wide, zoom, moving)
            if partner is not None:
                job(('wide', partners[i]), load_wide, partner, moving)
            answers = [result(k) for k in keys]
            if answers[0] is False:
                notes.append('wide view: download failed')
            elif any(a is None for a in answers):
                notes.append(f'downloading the {cfg.WIDE_VIEW_SIZE}\' wide view...')
                state['waiting'].update(keys)
            else:
                display = answers[0]
                display_partner = answers[1] if len(answers) > 1 and answers[1] else None
                factor = cfg.WIDE_VIEW_UPSCALE
        anchor = {'xy': target_xy(display), 'colour': colour} if pretty else None
        marks = None
        if state['selected'] is not None:                  # the circles follow the selected object
            positions = state['selected_positions']
            first = min(positions) if positions else None
            marks = {'red': positions.get(i),
                     'blue': positions[first] if first is not None else (state['selected']['ra'], state['selected']['dec'])}
            label = state['selected']['label']
            legend_text = [f"{label} at its first sighting" + (f" (T{first + 1})" if first is not None else ''),
                           f"{label} in this frame"]
        else:
            legend_text = [movement['reference_label'], 'object found in this frame']
        for text, value in zip(state['circle_legend'].get_texts() if 'circle_legend' in state else [], legend_text):
            text.set_text(value)
        if state['ax'] is not None:
            state['ax'].remove()
        ax = state['ax'] = fig.add_axes(layout['image'], projection=display['wcs'])
        im = draw_zoom(ax, display, i, n, zooms[0]['mjd'], movement, brightness, pretty=pretty,
                       partner=display_partner, anchor=anchor, factor=factor, marks=marks)
        if display is not zoom:
            _footprint(ax, display, zoom)
        ax.coords[0].set_axislabel('RA')
        ax.coords[1].set_axislabel('Dec')

        state['drawn'] = []
        if state['catalog'] and 'map' in zoom:
            state['drawn'], text = draw_map(ax, display, zoom['map'])
            notes.append(text)
            selected = state['selected']
            if selected is not None:
                notes.append(f"SELECTED: {selected['label']}" + (f" - {selected['info']}" if selected.get('info') else '')
                             + ('' if selected['kind'] in ('artifact', 'planet') else '   (T = track it)')
                             + ('' if marks and marks['red'] else '   - not detected in this frame')
                             + '   (click empty sky to go back to the target)')
        elif state['catalog']:
            ny, nx = display['image'].shape
            centre = display['wcs'].pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
            radius = np.hypot(nx, ny) / 2 * cfg.PIXEL_ARCSEC * u.arcsec
            field_key = ('field', round(centre.ra.deg, 3), round(centre.dec.deg, 3), round(radius.to_value(u.arcsec)))
            field = (tracked.get('field') if display is zoom and tracked.get('field') is not None
                     else result(field_key))
            if field is None:
                job(field_key, query_field, centre, radius, True)
                state['waiting'].add(field_key)
                notes.append('loading Gaia / CatWISE...')
            elif field is False:
                notes.append('catalogs unavailable')
            else:
                jpl_key = ('jpl', i, display is not zoom)
                bodies = identified(jpl_key)
                if jpl_key in jobs and bodies is None:
                    notes.append('asking JPL (~20 s)...')
                    state['waiting'].add(jpl_key)
                notes.append(overlay_catalog(ax, display, field, bodies))
        if pretty:
            _dark_axes(ax)
            _colour_key(cbar_ax, zoom, partner)
        else:
            cbar_ax.clear()
            fig.colorbar(im, cax=cbar_ax, label='MJy/sr (zodi subtracted)')
        status.set_text('\n'.join(textwrap.fill(note, 120) for note in notes))
        fig.canvas.draw_idle()

    def check_waiting():
        """Redraw once whatever the current frame is waiting for (download, catalog, JPL) has arrived."""
        if not state['waiting']:
            return
        done = all((identified(k) is not None) if k[0] == 'jpl' else (result(k) is not None) for k in state['waiting'])
        if done:
            draw(state['i'])

    def on_key(event):
        i = int(slider.val) - 1
        key = (event.key or '').lower()
        if key == 'right':
            slider.set_val(min(i + 2, n))
        elif key == 'left':
            slider.set_val(max(i, 1))
        elif key == 'f' and len(flagged):
            later = flagged[flagged > i]
            slider.set_val((later[0] if len(later) else flagged[0]) + 1)
        elif key == 'w':
            state['wide'] = not state['wide']
            draw(i)
        elif key == 'g':
            state['catalog'] = not state['catalog']
            draw(i)
        elif key == 'j':
            state['catalog'] = True
            wide = bool(state['wide'] and result(('wide', i)))
            half = (cfg.WIDE_VIEW_SIZE / 2 / 60) if wide else (cfg.ZOOM_SIZE_PIX / 2 * cfg.PIXEL_ARCSEC / 3600)
            jpl_key = ('jpl', i, wide)
            jobs[jpl_key] = True                       # (marks the lookup as asked for)
            identify_async(jpl_key, zooms[i]['mjd'], zooms[i]['expected_ra'], zooms[i]['expected_dec'], half)
            draw(i)
        elif key == 't' and state['selected'] is not None:
            spec = track_spec(state['selected'], tracked, zooms[i]['mjd'])
            if spec['kind'] == 'none':
                status.set_text(f"Can't track {spec['name']}: {spec['reason']}")
                fig.canvas.draw_idle()
            else:
                print(f"\nTracking {spec.get('name', state['selected']['label'])} next...")
                request_track(spec)
                plt.close('all')                           # main starts its analysis when the windows are closed

    def on_click(event):
        """Select the mapped object nearest the click (within 3 original pixels)."""
        if event.inaxes is not state['ax'] or not state['drawn'] or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        distance = [np.hypot(dx - x, dy - y) for dx, dy, _ in state['drawn']]
        k = int(np.argmin(distance))
        state['selected'] = state['drawn'][k][2] if distance[k] <= 3 else None
        state['selected_positions'] = object_positions(zooms, state['selected']) if state['selected'] else {}
        draw(state['i'])

    slider.on_changed(lambda value: draw(int(value) - 1))
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    timer = fig.canvas.new_timer(interval=400)
    timer.add_callback(check_waiting)
    timer.start()
    fig.canvas.mpl_connect('close_event', lambda event: (timer.stop(), pool.shutdown(wait=False, cancel_futures=True)))
    state['circle_legend'] = fig.legend(handles=circle_legend(movement), loc='upper right', fontsize=9)
    if pretty:                                              # (moved below the two-line title)
        state['circle_legend'].set_bbox_to_anchor((0.99, 0.925), transform=fig.transFigure)
    if has_map:                                             # what every sky-map marker means
        key = fig.legend(handles=map_legend(), loc='center left', bbox_to_anchor=(layout['legend_x'], 0.5),
                         fontsize=8, title='Sky map', title_fontsize=9, labelspacing=1.1, borderpad=0.8,
                         facecolor='0.18' if pretty else 'white', edgecolor='0.5', framealpha=1)
        for text in key.get_texts() + [key.get_title()]:
            text.set_color('white' if pretty else 'black')
    fig.suptitle(f"SPHEREx: {tracked['name']}   ←/→ step, F next flagged ({len(flagged)}), W wide view, "
                 + (f"G sky map, click + T track" if has_map else f"G Gaia/CatWISE, J JPL asteroids")
                 + ('\ncolour, sharpening and resolution are for display only - measurements use the original pixels'
                    if pretty else ''), fontsize=10, y=0.995, color='white' if pretty else 'black')
    draw(0)
    fig._slider, fig._timer = slider, timer                # keep them alive (matplotlib holds weak refs)
    return fig


def show_time_series(frames, tracked, movement, brightness):
    """Left top: offset from the T1 position vs time. Left bottom: brightness vs the object's own spectrum
    vs time. Right: the path on the sky. Points are colored by wavelength; flagged frames are ringed red."""
    days = np.array([f['mjd'] for f in frames]) - frames[0]['mjd']
    wavelength = np.array([f['wavelength_um'] for f in frames])

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    ax_move = fig.add_subplot(grid[0, 0])
    ax_bright = fig.add_subplot(grid[1, 0], sharex=ax_move)
    ax_sky = fig.add_subplot(grid[:, 1])
    color = dict(cmap='turbo', vmin=wavelength.min(), vmax=wavelength.max(), s=14, zorder=3)

    # movement: each frame's flag threshold depends on its detector (gray dashes)
    threshold = cfg.MOVEMENT_SIGMA * movement['scatter_arcsec']
    limit = float(np.nanmedian(threshold))
    ax_move.scatter(days, threshold, marker='_', s=40, color='0.65', zorder=1,
                    label=f'flag threshold ({cfg.MOVEMENT_SIGMA}x its detector\'s scatter)')
    reliable = reliable_frames(frames)
    positioned = usable_frames(frames)          # reliable AND detected: the frames a position means something in
    if not positioned.any():
        positioned = reliable

    def dots(ax, x, y, yerr=None, use=reliable, unused_label='bad pixel in circle (not used)'):
        """Used frames as dots colored by wavelength; the others as gray x's (shown for completeness - they
        aren't used, so their odd values aren't events)."""
        if yerr is not None:
            ax.errorbar(x[use], y[use], yerr=yerr[use], fmt='none', ecolor='0.75', lw=0.6)
        artist = ax.scatter(x[use], y[use], c=wavelength[use], **color)
        ax.scatter(x[~use], y[~use], marker='x', s=14, color='0.6', lw=0.8, zorder=2, label=unused_label)
        return artist

    unused = 'not detected or bad pixel (not used)'
    offset = movement['offset_arcsec']
    sc = dots(ax_move, days, offset, movement['error_arcsec'], use=positioned, unused_label=unused)
    if 'expected_offset_arcsec' in movement:                # a named star: the catalog's predicted motion
        order = np.argsort(days)
        ax_move.plot(days[order], movement['expected_offset_arcsec'][order], color='deepskyblue', lw=1.5, zorder=2,
                     label='predicted from SIMBAD/Gaia proper motion + parallax')
    ax_move.axhline(cfg.PIXEL_ARCSEC, color='red', ls=':', lw=1, label=f'1 SPHEREx pixel ({cfg.PIXEL_ARCSEC}")')
    top = np.nanmax(np.r_[offset[positioned], movement.get('expected_offset_arcsec', [0])])
    ax_move.set(ylabel='Offset from T1 (arcsec)', title='Movement over time',
                ylim=(0, max(cfg.PIXEL_ARCSEC * 1.15, top * 1.15)))
    ax_move.legend(fontsize=8, loc='upper left')

    # brightness
    deviation = brightness['deviation'] * 100
    measured = brightness.get('used', reliable & np.isfinite(deviation))
    from_glow = brightness.get('source') == 'glow'
    ax_bright.set(xlabel='Days since T1', ylabel='Brightness vs own spectrum (%)',
                  title='Brightness over time' + (' - relative, from the saturated star\'s outer glow' if from_glow else ''))
    if measured.any():
        flux_err_pct = (np.zeros(len(frames)) if from_glow else          # (abs: a faint reference can be < 0)
                        np.abs(np.array([f['flux_err_mjy'] for f in frames]) / brightness['reference_mjy'] * 100))
        band = cfg.BRIGHTNESS_SIGMA * brightness['scatter'] * 100
        ax_bright.axhspan(-band, band, color='0.88', label=f'±{cfg.BRIGHTNESS_SIGMA}x typical deviation ({band:.1f}%)')
        ax_bright.axhline(0, color='0.5', lw=0.8)
        dots(ax_bright, days, deviation, flux_err_pct, use=measured, unused_label='brightness not usable')
        span = max(1.5 * band, np.nanpercentile(np.abs(deviation[measured]), 99) * 1.15)   # (a rare outlier may be off-scale)
        ax_bright.set(ylim=(-span, span))
        ax_bright.legend(fontsize=8, loc='upper left')
    else:
        saturated = sum(bool(f.get('saturated')) for f in frames)
        ax_bright.text(0.5, 0.5, 'brightness not measurable' + (f'\n({saturated} of {len(frames)} frames saturated: '
                       'the object is too bright for SPHEREx)' if saturated else ''),
                       ha='center', va='center', transform=ax_bright.transAxes, color='0.4')

    for ax, values, flags in ((ax_move, offset, movement['flagged']), (ax_bright, deviation, brightness['flagged'])):
        if flags.any():
            ax.scatter(days[flags], values[flags], s=90, facecolor='none', edgecolor='red', lw=1.5, zorder=4)

    # path on the sky
    d_ra, d_dec = movement['d_ra_arcsec'], movement['d_dec_arcsec']
    ax_sky.add_patch(Circle((0, 0), limit, color='0.88', zorder=1))
    dots(ax_sky, d_ra, d_dec, use=positioned, unused_label=unused)
    if 'expected_d_ra_arcsec' in movement:
        order = np.argsort(days)
        ax_sky.plot(movement['expected_d_ra_arcsec'][order], movement['expected_d_dec_arcsec'][order],
                    color='deepskyblue', lw=1.5, zorder=2, label='predicted (catalog)')
    ax_sky.scatter([0], [0], marker='+', s=150, color='deepskyblue', zorder=4, label='T1 position')
    extent = max(1.0, np.nanmax(np.abs(np.r_[d_ra[positioned], d_dec[positioned],
                                            movement.get('expected_d_ra_arcsec', [0]),
                                            movement.get('expected_d_dec_arcsec', [0])])) * 1.2)
    ax_sky.set(xlim=(extent, -extent), ylim=(-extent, extent), aspect='equal',     # east (+RA) to the left
               xlabel='ΔRA (arcsec, east ←)', ylabel='ΔDec (arcsec, north ↑)', title='Path on the sky')
    ax_sky.legend(fontsize=8, loc='upper right')
    fig.colorbar(sc, ax=[ax_move, ax_bright, ax_sky], label='Wavelength (µm)', fraction=0.025, pad=0.02)
    fig.suptitle(f"{tracked['name']}: {len(frames)} frames over {days[-1]:.0f} days")
    return fig


def show_motion(frames, tracked, movement):
    """Where the asteroid went (detected frames only; undetected ones are gray x's).
    Top left: its path across the sky - JPL Horizons' path as a line, detections as dots colored by date.
    Top right: distance travelled since the first detection, over time (colored by wavelength).
    Bottom left: how far each detection is from where the ephemeris said it would be (1 pixel = 6.15").
    Bottom right: its spectrum - brightness against wavelength, colored by date."""
    detected = movement['detected']
    mjd = np.array([f['mjd'] for f in frames])
    days = mjd - mjd[0]
    wavelength = np.array([f['wavelength_um'] for f in frames])
    flux = np.array([f['flux_mjy'] for f in frames])
    ra, dec = np.array([f['object_ra'] for f in frames]), np.array([f['object_dec'] for f in frames])

    ephemeris = tracked['ephemeris']
    on_path =np.degrees(np.interp(mjd, ephemeris.mjd, ephemeris.ra))
    ra = on_path + (ra - on_path + 180) % 360 - 180           # same 360-degree branch as the path (no wrap-around)

    fig, ((ax_sky, ax_dist), (ax_res, ax_spec)) = plt.subplots(2, 2, figsize=(14, 9))

    ax_sky.plot(np.degrees(ephemeris.ra), ephemeris.dec, color='0.75', lw=2, zorder=1, label='JPL Horizons path')
    sc = ax_sky.scatter(ra[detected], dec[detected], c=days[detected], cmap='viridis', s=16, zorder=3)
    ax_sky.scatter(ra[~detected], dec[~detected], marker='x', color='0.5', s=16, zorder=2, label='not detected')
    ax_sky.invert_xaxis()                                      # east (larger RA) to the left, as on the sky
    ax_sky.set(xlabel='RA (degrees)', ylabel='Dec (degrees)', title='Path across the sky')
    ax_sky.legend(fontsize=8)
    fig.colorbar(sc, ax=ax_sky, label='Days since first frame')

    sc = ax_dist.scatter(days[detected], movement['offset_arcsec'][detected] / 3600, c=wavelength[detected],
                         cmap='turbo', s=16)
    ax_dist.set(xlabel='Days since first frame', ylabel='Distance travelled since first detection (degrees)',
                title='Movement over time')
    fig.colorbar(sc, ax=ax_dist, label='Wavelength (µm)')

    residual = movement['residual_arcsec']
    ax_res.scatter(days[detected], residual[detected], c='black', s=10)
    ax_res.axhline(cfg.PIXEL_ARCSEC, color='red', ls=':', lw=1, label=f'1 SPHEREx pixel ({cfg.PIXEL_ARCSEC}")')
    ax_res.axhline(cfg.RESIDUAL_LIMIT_PX * cfg.PIXEL_ARCSEC, color='red', lw=1,
                   label=f'flag limit ({cfg.RESIDUAL_LIMIT_PX} px)')
    ax_res.set(xlabel='Days since first frame', ylabel='Distance from ephemeris position (arcsec)',
               title=f'Found where predicted: median {np.median(residual[detected]):.1f}"',
               ylim=(0, max(1.3 * cfg.RESIDUAL_LIMIT_PX * cfg.PIXEL_ARCSEC, 1.1 * np.max(residual[detected]))))
    ax_res.legend(fontsize=8)

    shown = detected & (flux > 0)
    sc = ax_spec.scatter(wavelength[shown], flux[shown], c=days[shown], cmap='viridis', s=16)
    ax_spec.set(xlabel='Wavelength (µm)', ylabel='Flux density (mJy)', yscale='log', title='Its spectrum')
    fig.colorbar(sc, ax=ax_spec, label='Days since first frame')

    fig.suptitle(f"{tracked['name']}: {int(detected.sum())} detections over {days[detected].max() - days[detected].min():.0f} days")
    fig.tight_layout()
    return fig


def show_spectrum(check, tracked):
    """SPHEREx measurements (every frame) vs the object's IRSA catalog brightness."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    wl, flux, err = check['points'].T
    ax.errorbar(wl, flux, yerr=err, fmt='o', color='black', ms=2.5, lw=0.5, label=f'SPHEREx ({len(wl)} frames)')
    for band, band_wl, band_flux, band_err in check['catalog']:
        ax.errorbar(band_wl, band_flux, yerr=band_err, fmt='s', color='red', mfc='none', ms=9, mew=1.5)
        ax.annotate(band, (band_wl, band_flux), textcoords='offset points', xytext=(6, 6), fontsize=8, color='red')
    ax.plot([], [], 's', color='red', mfc='none', label='IRSA catalog (2MASS, AllWISE)')
    ax.set(xlabel='Wavelength (µm)', ylabel='Flux density (mJy)',
           title=f"Spectrum check: {tracked['name']} - {'same object' if check['same_object'] else 'MISMATCH'}\n"
                 f"(SPHEREx/catalog = {check['scale']:.2f}, shape within {check['scatter'] * 100:.0f}%)")
    ax.legend()
    fig.tight_layout()
    return fig
