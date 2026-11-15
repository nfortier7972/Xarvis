"""Start the XARVIS website:  python run.py   then open http://127.0.0.1:5000

The desktop analysis app (matplotlib windows) is still available:  python -m xarvis_pipeline
"""
from webapp.server import main

if __name__ == "__main__":
    main()
