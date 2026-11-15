# Test clean: what changed (from git HEAD 57f44c6)

This is a trial merge of the separate code into one website and one pipeline. Try it on a new branch first.

## Fixes
- `xarvis_pipeline/main.py`: removed the committed merge-conflict markers. Before this, the pipeline could not be imported.
- Line endings: all text files are LF. `.gitattributes` keeps them that way on Windows and Mac.
- `.gitignore`: `__pycache__`, `.DS_Store` and `results/` are no longer committed.
- Removed files:
  - `test_config.json` and `load_config.py` (unused second config);
  - `dwarf_cache.json` (stale);
  - `TEST_Robert_code.py` (superseded by `xarvis_pipeline/irsa.py`);
  - `Test_Jonah_code.py` (use `python -m xarvis_pipeline`).
- Requirements: one `requirements.txt`, without the unused `firefly_client`.

## New layout
| Before | After |
|---|---|
| `Jonah_Code_File/` | `xarvis_pipeline/` (same modules; `test_algorithm.py` moved to `tests/`) |
| `Test_Robert_Code_Folder/app.py` | `webapp/server.py` (start with `python run.py`) |
| `Test_Robert_Code_Folder/templates`, `static` | `webapp/templates`, `webapp/static` |
| `Test_Robert_Code_Folder/tools` | `tools/` |
| `results/433_Eros.ecsv` | `results_examples/433_Eros.ecsv` (`results/` is now generated output) |

## The website now runs the pipeline
- `xarvis_pipeline/api.py`: `analyse(request, folder, on_progress)` runs the pipeline with no windows.
- `xarvis_pipeline/render.py`: makes PNGs:
  - every frame, resampled north up and east left on one grid, so frames line up for blinking;
  - inverse-variance **stacks per detector**;
  - **last minus first visit** differences.
- `webapp/analysis.py` and `webapp/jobs.py`: `/api/analysis` runs it as a background job, one at a time, and reuses finished results.
- Analysis popup (`index.html`):
  - **Load SPHEREx images** button with a progress bar;
  - frame player (play, slider), flags and saturation per frame;
  - stack and difference views;
  - spectrum chart across the 102 channels.
- Shareable links: `/?target=Barnard&date=2025-05-01`, `/?ra=08:55:10.83&dec=-07:14:42.5&px=50`.

## Small pipeline changes, made to work on the web
- `main.PROGRESS_HOOK`: a progress callback for headless runs (the desktop live window is unchanged).
- `config.SPREAD_IMAGES`: with `max_images`, the website spreads images over the whole survey instead of taking the first ones. Desktop default is unchanged.
- `make_zooms` reads the patch size at run time, so the website's patch-size setting is used.
- `stars.gaia_star()`, plus an `epoch_mjd` field in `positions_at`: stars from ESA Gaia DR3 (epoch 2016.0) without SIMBAD.
- `skymap.hipparcos()` now uses the ESA archive, and `star_names()` uses the names shipped in `webapp/static/sky.json`. These replace the VizieR mirror and the raw GitHub download.
- Still to decide: the desktop app's `--star NAME` still uses SIMBAD to resolve names.

## Tested
- `python -m unittest discover tests`: 8 tests pass (the team's 5 algorithm tests plus 3 new website/rendering tests).
- `python tools/verify_motion.py`: 45 of 45 pass.
- Real SPHEREx data through the website:
  - Barnard's Star: 30 images in 18 s;
  - RW Aurigae: 100 images in 32 s;
  - WISE 0855: 8 images in 13 s.
- Desktop app: `python -m xarvis_pipeline --coords "08h55m10.8s -07d14m42s" --max-images 4 --no-show` runs.

## How to try it
1. `git checkout -b test-clean`
2. Delete `Jonah_Code_File/`, `Test_Robert_Code_Folder/`, `Test_Jonah_code.py`, `test_config.json` and `results/` from the branch.
3. Unzip this folder's contents into the repo root.
4. Run `git add -A` and `git status` (the moves show as renames).
5. Run `pip install -r requirements.txt`, then `python run.py`.
