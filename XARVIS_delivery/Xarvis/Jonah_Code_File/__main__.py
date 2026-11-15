"""Run the analysis: just run this file (e.g. the Run button in VS Code), or: python -m Jonah_Code_File

Settings are in the project-level config.json (the asteroid, star or sky position to track, the time window, how many images),
or on the command line, e.g.:  python -m Jonah_Code_File --coords "08h55m10.8s -07d14m42s"   (--help for all)
"""

import sys
from pathlib import Path

if not __package__:
    # Run directly as a file: Python puts THIS folder first on the import path, where our own config.py,
    # irsa.py, main.py ... could shadow other modules, and the package-relative imports below can't work.
    # Replace it with the folder that contains the package.
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
    from Jonah_Code_File.main import cli
else:
    from .main import cli

if __name__ == '__main__':
    cli()
