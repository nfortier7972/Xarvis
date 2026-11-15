"""Background jobs for the SPHEREx image time-lapse (the bridge between this website and Jonah_Code_File).

A run takes seconds to minutes (IRSA downloads), far longer than one web request should, so it runs on a worker thread
and the browser polls for progress. ONE worker: the pipeline's settings are module-wide, so runs go one at a time.
Each job writes into <results>/jobs/<id>/ (summary.json + frame_<n>.png); the id is a hash of the request, so the
same request finds its finished results again - also after a server restart.

    submit(request)  -> job id                 (starts the run, or reuses a queued / running / finished one)
    status(job_id)   -> dict, or None           {'status': queued|running|done|error, 'done', 'total', 'message',
                                                 'summary' (when done), 'error' (when failed)}
    folder(job_id)   -> where its files are

The pipeline is imported on first use, in the worker - the viewer itself starts without it (and still works if the
pipeline's packages are missing: the popup then shows the error).
"""
import hashlib
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent              # the Xarvis folder: Jonah_Code_File is imported from here
os.environ.setdefault("MPLBACKEND", "Agg")                # before anything imports matplotlib: never open a window

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spherex-analysis")
_jobs = {}
_lock = threading.Lock()


def pipeline():
    """The pipeline's web module (Jonah_Code_File.web), wherever the server was started from."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Jonah_Code_File import web
    return web


def job_id(request):
    return hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]


def folder(jid):
    return pipeline().results_folder() / jid


def _saved_summary(jid):
    try:
        return json.loads((folder(jid) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(jid):
    with _lock:
        job = dict(_jobs[jid]) if jid in _jobs else None
    if job is None or job["status"] == "done":
        summary = _saved_summary(jid)
        if summary is None:
            return None if job is None else {**job, "status": "error", "error": "The saved results are missing."}
        job = {**(job or {"done": summary["n_frames"], "total": summary["n_frames"], "message": "Saved results"}),
               "status": "done", "summary": summary}
    return job


def finished():
    """Every finished analysis (newest first), as the viewer draws it on the sky: where it is, what it was, which
    SPHEREx images it holds and the request that made it (to open it again). A tracked object that moves (an asteroid,
    a NEW mover) has the path its images followed; anything else one centre."""
    try:
        folders = [f for f in pipeline().results_folder().iterdir() if JOB_FOLDER(f.name)]
    except OSError:
        return []
    out = []
    for path in folders:
        try:
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            frames, request, target = summary["frames"], summary["request"], summary["target"]
            points = [[f["expected_ra"], f["expected_dec"]] for f in frames
                      if f.get("expected_ra") is not None and f.get("expected_dec") is not None]
            if not points:                                  # (saved before positions were kept: the patch centre)
                points = [[target["ra_deg"], target["dec_deg"]]]
            moving = bool(target.get("moving"))
            ra, dec = points[len(points) // 2] if moving else points[0]
            database = summary.get("database") or {}
            out.append({
                "job": path.name, "name": target.get("name") or "analysis", "tracking": summary.get("tracking"),
                "moving": moving, "ra_deg": ra, "dec_deg": dec, "path": points if moving else None,
                "patch_px": summary["patch_px"], "pixel_arcsec": summary["pixel_arcsec"],
                "n_frames": summary["n_frames"], "images_available": summary.get("images_available"),
                "first_utc": frames[0]["time_utc"] if frames else None, "last_utc": frames[-1]["time_utc"] if frames else None,
                "new_objects": len(database.get("new_objects") or []), "request": request,
                "saved": (path / "summary.json").stat().st_mtime})
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue                                        # an unreadable folder is skipped, not fatal
    return sorted(out, key=lambda a: -a["saved"])


def JOB_FOLDER(name):
    return len(name) == 16 and all(c in "0123456789abcdef" for c in name)


def friendly(error):
    """One sentence for the popup from whatever stopped the run."""
    text = " ".join(str(error).split())[:600] or type(error).__name__
    if isinstance(error, ImportError):
        return f"The analysis pipeline could not be loaded on the server ({text}). Check that requirements.txt is installed."
    if "attempts failed" in text or "HTTP 5" in text or "unreachable" in text:
        return f"NASA/IPAC IRSA did not answer ({text}). The archive is often briefly overloaded: try again in a minute."
    return text


def submit(request, run=None):
    """Start the job for `request` (see Jonah_Code_File.web.web_request) unless it is already queued, running or
    finished. run(request, folder, on_progress): the work - the pipeline's analyse by default (tests pass a stand-in)."""
    jid = job_id(request)
    with _lock:
        known = _jobs.get(jid)
        if known and known["status"] in ("queued", "running"):
            return jid
        if _saved_summary(jid) is not None:
            return jid
        _jobs[jid] = {"status": "queued", "done": 0, "total": 0, "message": "Waiting for another analysis to finish..."}

    def progress(done, total, message):
        with _lock:
            _jobs[jid].update(status="running", done=int(done), total=int(total), message=str(message))

    def work():
        progress(0, 0, "Starting...")
        try:
            (run or pipeline().analyse)(request, folder(jid), progress)
            with _lock:
                _jobs[jid].update(status="done", message="Done")
        except BaseException as e:                         # every failure reaches the browser as a message
            traceback.print_exc()
            with _lock:
                _jobs[jid].update(status="error", error=friendly(e))

    _pool.submit(work)
    return jid
