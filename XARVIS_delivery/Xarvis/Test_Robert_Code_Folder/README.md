# XARVIS viewer (website)

The XARVIS website: a 3D view of the solar system with NASA's SPHEREx telescope, what it can see, any patch of sky by
its coordinates, and the known objects there. Official agency data only (one labelled exception: Stellarium constellation
figures).

## Run

```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 (about page: http://127.0.0.1:5000/about). The first start downloads NASA/JPL's DE440s
ephemeris (32 MB, cached by astropy) from naif.jpl.nasa.gov. After updating the code, stop the server (Ctrl+C) and start
it again; the browser picks up changed JavaScript and CSS by itself (file versions are in the URLs).

## Files

| Path | What |
|---|---|
| `app.py` | Flask backend: pages, and the data API (DE440s planets, SPHEREx from JPL Horizons, JPL SBDB and SB Ident lookups). Its docstring lists every route |
| `templates/index.html` | The viewer page (markup only) |
| `static/css/viewer.css` | Viewer styles |
| `static/js/*.js` | Viewer code, loaded in this order (plain scripts sharing one global scope; each file starts with a short summary) |
| | `core.js`: constants, helpers, clock, maths, JPL mean elements |
| | `scene.js`: server data and the scene state for each frame; star motion |
| | `draw.js`: projection, star backdrop, constellations and labels, real asteroids |
| | `tracker.js`: Track objects panel |
| | `view3d.js`: what SPHEREx can see, the 3D scene, cursor read-out |
| | `analysis.js`: Go to sky coordinates and the sky point analysis popup |
| | `app.js`: main loop, status panel, UI events, boot |
| `templates/about.html`, `templates/404.html`, `static/site.css` | About and 404 pages |
| `static/sky.json`, `static/small_bodies.json`, `static/targets.json` | Data files (built by `tools/`) |
| `tools/` | Data builders and `verify_motion.py` (accuracy checks) |
| `TEST_Robert_code.py` | Stand-alone IRSA test script (not used by the website) |

## Features (unchanged in this clean-up)
- **Main view:**
  - the solar system in 3D, with SPHEREx's real orbit (NASA/JPL Horizons), the planets and Moon (DE440s), about 12,700
    real small bodies (JPL SBDB) and the stars (ESA Hipparcos);
  - a plain sentence about SPHEREx, with **Advanced orbital parameters** and accuracy notes collapsed underneath.
- **What SPHEREx can see:** the sky at least 91 degrees from the Sun and not behind Earth, shaded on the star backdrop.
- **Go to sky coordinates:**
  - type RA/Dec or pick a challenge test target (ESA Gaia DR3, `static/targets.json`);
  - patch size is an Advanced setting in SPHEREx pixels (default 50 x 50 at 6.15");
  - double-clicking the sky opens the same popup.
- **Sky point analysis popup:**
  - whether SPHEREx can see the point, and when it next can;
  - the target's motion during the survey;
  - known asteroids in the patch (JPL SB Ident) and a patch map;
  - the image box only appears inside SPHEREx's view and data dates.
- **Track objects:** search, plus Show (known classes or unknown SPHEREx candidates) and Only filters. Any other object
  can be looked up live at JPL. Tracked objects are highlighted.
- **Time:** limited to the SPHEREx data span (2025-05-01 00:00 UTC to the latest observation), with Jump to first or
  latest data, up to 1 year per second.
- **Interface:** landing panel, About and 404 pages, links to every data source.

## Hooks for the backend (search `TODO(backend)`)
- `requestAnalysis(payload)` in `static/js/analysis.js` receives everything the popup knows about a patch:
  - target RA/Dec and patch size;
  - the time;
  - the Gaia target (proper motion, epoch);
  - SPHEREx's position.

  It should start the image analysis and fill the image box (`#anaViz`).
- `XARVIS_addCandidates(rows)` in `static/js/tracker.js` adds unknown objects from the SPHEREx analysis to Track objects.

## Data sources
| What | Source | Official? |
|---|---|---|
| Stars | ESA Hipparcos new reduction (ESA archive) -> `static/sky.json` | yes (ESA) |
| Planets, Moon | NASA/JPL DE440s (NAIF), 1849-12-26 .. 2150-01-22; JPL mean orbital elements outside | yes (NASA/JPL) |
| SPHEREx orbit | NASA/JPL Horizons, spacecraft -163182, 2-minute track, interpolated (within ~2" of Horizons); orbit model with the official parameters when Horizons has no data | yes (NASA/JPL) |
| SPHEREx facts (launch, science dates, orbit table, field of view, pixel scale, 91 deg Sun limit) | IPAC SPHEREx Explanatory Supplement v2.1, Horizons entry | yes (IPAC, NASA/JPL) |
| Small bodies | NASA/JPL SBDB -> `static/small_bodies.json`; live lookups via the SBDB API with Horizons elements; SB Ident for patches | yes (NASA/JPL) |
| Test targets | ESA Gaia DR3 -> `static/targets.json`; WISE 0855-0714 from the challenge text | yes (ESA, NASA) |
| Constellation figures, borders, names | Stellarium "modern" sky culture (CC BY-SA 4.0) | **no** (labelled in the viewer) |

## Rebuilding the data files (not needed to run)
```
python tools/build_sky.py --source esa --stellarium index.json --out static/sky.json --check
python tools/build_small_bodies.py --out static/small_bodies.json
python tools/build_targets.py --out static/targets.json
```

## Accuracy checks
`python tools/verify_motion.py` (needs the internet): 45 checks against real events and independent code. They cover:
- transits of Venus (2012) and Mercury (2019), and solar and lunar eclipses;
- equinoxes and solstices, perihelion and aphelion, orbital periods, and the Moon's months;
- star motion against astropy;
- the SPHEREx orbit against the official parameters;
- small bodies against JPL Horizons.

Known limits:
- Star motion is proper motion only. Hipparcos has no radial velocities.
- Planets can't be predicted beyond about ±5,000 years, though the time range is now limited to the SPHEREx data anyway.
- SPHEREx is drawn on a circular orbit.
- The Moon and flight-direction avoidance angles are not published, so they are not drawn.

## Notes for merging into the team repository
- **Path changes in this folder:**
  - the page's styles and scripts moved out of `templates/index.html` into `static/css/` and `static/js/`, which gives
    smaller files and fewer merge conflicts;
  - the old `dwarf_cache.json` is gone, since nothing uses it.
- **Repository root still needs** a `.gitignore` (`__pycache__/`, `*.pyc`, `.DS_Store`, `results/`) and a
  `.gitattributes` (`* text=auto eol=lf`). Mixed Windows and Mac line endings caused whole-file diffs.
- **API:** `app.py` keeps the same routes, so the pipeline team can build against them. The popup's `requestAnalysis()`
  hook is where the SPHEREx image job plugs in.
