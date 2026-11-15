# Xarvis

This is a test commit

Nick was here
m

Who it is for, and what each group can do with what it shows: [AUDIENCE.md](AUDIENCE.md).

## Run the website (viewer + SPHEREx image time-lapse)

From this folder, with the virtual environment active (`.venv\Scripts\Activate.ps1`) and `pip install -r requirements.txt` done:

```
python run_web.py
```

Then open http://127.0.0.1:5000. Double-click the sky (or use "Go to sky coordinates"): when SPHEREx can see the point on the date shown, the popup loads the SPHEREx images of that patch from NASA/IPAC IRSA and plays them.

- Website: `Test_Robert_Code_Folder/` (`app.py`, `analysis_jobs.py`, `templates/`, `static/`)
- Pipeline: `Jonah_Code_File/` (`web.py` is the website's entry point; desktop app: `python -m Jonah_Code_File`)
- Settings: `config.json`
- Web results: `results/jobs/<job>/` (`summary.json`, `frame_<n>.png`)

Tests:

```
python -m unittest Jonah_Code_File.test_algorithm
python -m unittest discover -s Jonah_Code_File -t . -p "test_*.py"
python -m unittest test_web_integration
python Test_Robert_Code_Folder/tools/verify_motion.py
```

`$env:XARVIS_ONLINE=1; python -m unittest test_web_integration` also runs one real analysis against IRSA.
