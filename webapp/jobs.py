"""Background jobs for the SPHEREx image analysis (a run takes from seconds to minutes, longer than a web request).

One worker thread runs jobs one at a time (the pipeline's settings are module globals, and IRSA downloads are already
done 16 at a time inside a run). Each job writes into results/jobs/<id>/; a finished job's folder is reused when the
same request comes again, also after a server restart. For a public deployment, swap this for RQ/Celery + Redis.
"""

import hashlib
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'jobs'
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='spherex-job')
_jobs = {}
_lock = threading.Lock()


def job_id(request):
    """The same request always gets the same id (so its results can be reused)."""
    return hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]


def folder(jid):
    return JOBS_DIR / jid


def status(jid):
    """{'status': queued|running|done|error, 'done', 'total', 'message', 'summary'?, 'error'?} or None."""
    with _lock:
        job = dict(_jobs[jid]) if jid in _jobs else None
    if job is None and (folder(jid) / 'summary.json').exists():                  # finished in an earlier session
        job = {'status': 'done', 'done': 0, 'total': 0, 'message': 'Saved results'}
    if job and job['status'] == 'done':
        job['summary'] = json.loads((folder(jid) / 'summary.json').read_text())
    return job


def submit(request, run):
    """Start (or reuse) the job for `request`. run(request, folder, on_progress) does the work. Returns the id."""
    jid = job_id(request)
    with _lock:
        known = _jobs.get(jid)
        if known and known['status'] in ('queued', 'running', 'done'):
            return jid
        if (folder(jid) / 'summary.json').exists():
            _jobs[jid] = {'status': 'done', 'done': 0, 'total': 0, 'message': 'Saved results', 'started': time.time()}
            return jid
        _jobs[jid] = {'status': 'queued', 'done': 0, 'total': 0, 'message': 'Waiting for the previous analysis to finish',
                      'started': time.time()}

    def progress(done, total, message):
        with _lock:
            _jobs[jid].update(status='running', done=done, total=total, message=message)

    def work():
        progress(0, 0, 'Starting...')
        try:
            run(request, folder(jid), progress)
            with _lock:
                _jobs[jid].update(status='done', message='Done')
        except Exception as e:                                                    # reported to the browser, not lost
            traceback.print_exc()
            with _lock:
                _jobs[jid].update(status='error', error=f'{type(e).__name__}: {e}'[:400])

    _pool.submit(work)
    return jid
