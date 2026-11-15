"""/api/analysis: run the SPHEREx image pipeline (xarvis_pipeline) for a patch of sky, as a background job.

    POST /api/analysis                      body: see _request() - returns 202 {"job": id}
    GET  /api/analysis/<id>                 {"status", "done", "total", "message", "summary" (when done)}
    GET  /api/analysis/<id>/<file>.png      frame_<n>.png, stack_d<d>.png, diff_d<d>.png
"""

from flask import Blueprint, abort, jsonify, request, send_from_directory

from . import jobs

bp = Blueprint('analysis', __name__)
MAX_IMAGES_LIMIT = 300


def _number(value, lo, hi, name):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number')
    if not lo <= v <= hi:
        raise ValueError(f'{name} must be between {lo} and {hi}')
    return v


def _request(body):
    """Validate the browser's request and keep only what the pipeline needs (this is also the job's identity)."""
    mode = body.get('mode')
    out = {'mode': mode, 'patch_px': int(_number(body.get('patch_px', 50), 10, 400, 'patch_px')),
           'max_images': int(_number(body.get('max_images', 30), 1, MAX_IMAGES_LIMIT, 'max_images'))}
    if mode == 'coordinates':
        out['ra'] = round(_number(body.get('ra'), 0, 360, 'ra'), 6)
        out['dec'] = round(_number(body.get('dec'), -90, 90, 'dec'), 6)
    elif mode == 'gaia_star':
        t = body.get('target') or {}
        out['target'] = {'name': str(t.get('name', 'target'))[:80], 'ra': _number(t.get('ra'), 0, 360, 'ra'),
                         'dec': _number(t.get('dec'), -90, 90, 'dec')}
        for k in ('pmra', 'pmdec', 'parallax'):
            if t.get(k) is not None:
                out['target'][k] = _number(t[k], -20000, 20000, k)
        if t.get('gaia_dr3'):
            out['target']['gaia_dr3'] = str(t['gaia_dr3'])[:30]
    elif mode == 'asteroid':
        out['asteroid'] = str(body.get('asteroid', ''))[:40]
        if not out['asteroid']:
            raise ValueError('asteroid id missing')
        if body.get('window'):
            out['window'] = [str(w)[:10] for w in body['window']][:2]
    else:
        raise ValueError("mode must be 'coordinates', 'gaia_star' or 'asteroid'")
    return out


def _run(req, folder, on_progress):
    from xarvis_pipeline.api import analyse              # imported on first use: the viewer starts fast
    analyse(req, folder, on_progress)


@bp.post('/api/analysis')
def start():
    try:
        req = _request(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'job': jobs.submit(req, _run), 'request': req}), 202


@bp.get('/api/analysis/<jid>')
def state(jid):
    if not jid.isalnum():
        abort(404)
    job = jobs.status(jid)
    if job is None:
        return jsonify({'error': 'unknown job'}), 404
    return jsonify(job)


@bp.get('/api/analysis/<jid>/<name>.png')
def image(jid, name):
    if not jid.isalnum() or not name.replace('_', '').isalnum():
        abort(404)
    return send_from_directory(jobs.folder(jid), f'{name}.png', max_age=3600)
