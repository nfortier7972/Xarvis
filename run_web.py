"""Start the XARVIS website with the SPHEREx image pipeline connected:

    python run_web.py            then open http://127.0.0.1:5000

Works from any folder: the paths are taken from this file's location. The website is Test_Robert_Code_Folder/app.py;
the pipeline (Jonah_Code_File) is imported as a package from this folder. The desktop app is unchanged:
python -m Jonah_Code_File
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")                # the web server never opens matplotlib windows
for folder in (ROOT, ROOT / "Test_Robert_Code_Folder"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from app import main                                       # noqa: E402  (Test_Robert_Code_Folder/app.py)

if __name__ == "__main__":
    main()
