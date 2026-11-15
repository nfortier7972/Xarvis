"""Every web request the program makes, in one place: timeouts, retries, disk caching and each service's limits.

Each service gets the retry policy that suits it (measured):
  irsa          IRSA answers "502 Proxy Error" for a moment and then succeeds: many quick retries, then back off
  jpl-sb-ident  JPL SB Ident allows 20 requests per 5 minutes (its HTTP 429 reply says so): requests are spaced to
                stay under that - across runs too - and a 429's Retry-After is honoured
  jpl           other JPL APIs (SBDB, Horizons): a few retries with back-off
  cds           SIMBAD / VizieR mirrors: a few retries with back-off
  github        reference files (Stellarium sky culture): a few retries
A refusal that retrying can't fix (HTTP 400, 403, 404 ...) is raised at once as Refused.
"""

import json
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter

from . import cache
from . import config as cfg

log = logging.getLogger(__name__)


class BadResponse(Exception):
    """The service answered, but not with what we asked for (e.g. an error page): worth retrying."""


class Refused(ValueError):
    """The service refused the request in a way retrying can't fix (HTTP 4xx other than 408 / 429)."""


def _retry_errors():
    errors = [requests.RequestException, OSError, BadResponse]
    try:                                        # pyvo / astroquery wrap IRSA's errors in their own
        from pyvo.dal.exceptions import DALAccessError
        errors.append(DALAccessError)
    except ImportError:
        pass
    return tuple(errors)


RETRY_ERRORS = _retry_errors()


@dataclass(frozen=True)
class Policy:
    attempts: int
    first_wait: float           # seconds before the first retry ...
    growth: float               # ... multiplied by this each time ...
    max_wait: float             # ... up to this
    fast_attempts: int = 0      # this many quick retries (first_wait each) before backing off
    rate: tuple = None          # (requests, seconds): never more than this many requests per window


def _policies():
    return {
        'irsa': Policy(cfg.MAX_ATTEMPTS, cfg.WAIT_SECONDS, 2.0, cfg.MAX_WAIT_SECONDS, fast_attempts=cfg.FAST_ATTEMPTS),
        'jpl-sb-ident': Policy(4, 5.0, 2.0, 60.0, rate=(18, 300)),
        'jpl': Policy(5, 1.0, 2.0, 20.0),
        'cds': Policy(4, 1.0, 2.0, 10.0),
        'github': Policy(4, 1.0, 2.0, 10.0),
    }


def policy(service):
    return _policies()[service]


def wait_before_retry(service, attempt):
    """Seconds to wait before retry number `attempt` (0-based)."""
    p = policy(service)
    if attempt < p.fast_attempts:
        return p.first_wait
    if p.fast_attempts:                         # after the quick retries: 0.1 s, 0.2 s, 0.4 s ... (IRSA outages)
        return min(0.1 * p.growth ** (attempt - p.fast_attempts), p.max_wait)
    return min(p.first_wait * p.growth ** attempt, p.max_wait)


# ----------------------------------------------------------------------------
# Connections
# ----------------------------------------------------------------------------

class TimeoutAdapter(HTTPAdapter):
    """Gives every request a timeout. astroquery/pyvo send theirs with none, so a stalled reply would make
    Python wait forever; with a timeout it fails and gets retried instead."""
    def send(self, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = cfg.REQUEST_TIMEOUT
        return super().send(request, **kwargs)


_thread_local = threading.local()


def session():
    """One reusable connection pool per thread (reusing connections is much faster than reconnecting)."""
    if not hasattr(_thread_local, 'session'):
        s = requests.Session()
        adapter = TimeoutAdapter(pool_maxsize=cfg.MAX_CONCURRENT_DOWNLOADS)
        s.mount('https://', adapter)
        s.mount('http://', adapter)
        _thread_local.session = s
    return _thread_local.session


# ----------------------------------------------------------------------------
# Rate limits (kept on disk, so back-to-back runs respect them too)
# ----------------------------------------------------------------------------

_rate_lock = threading.Lock()


def _rate_slot(service):
    """Wait until one more request to `service` fits within its rate limit, and record it."""
    limit = policy(service).rate
    if not limit:
        return
    count, window = limit
    path = cache.cache_dir().parent / f'rate_{service}.json'
    with _rate_lock:
        try:
            recent = [t for t in json.loads(path.read_text()) if time.time() - t < window]
        except (OSError, ValueError):
            recent = []
        if len(recent) >= count:
            wait = window - (time.time() - min(recent)) + 1
            log.info(f'  {service}: {len(recent)} requests in the last {window} s (limit) - waiting {wait:.0f} s...')
            time.sleep(max(wait, 0))
            recent = [t for t in recent if time.time() - t < window]
        recent.append(time.time())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(recent))
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------------

def get(url, params=None, *, service='irsa', label=None, cache_kind=None, cache_key=None, check=None,
        slots=None, timeout=None, quiet=True):
    """The body of a GET request (bytes), with the service's retries; from / into the disk cache when
    `cache_kind` is given (key: `cache_key`, or one made from the URL and parameters).
    check(content): raise BadResponse if the content isn't what was asked for (e.g. an HTML error page).
    slots: a semaphore limiting how many of these run at once. Raises Refused (permanent refusal) or
    RuntimeError (no answer after all attempts)."""
    label = label or service
    if cache_kind:
        cache_key = cache_key or cache.make_key(url, *sorted((params or {}).items()))
        cached = cache.get(cache_kind, cache_key)
        if cached is not None:
            return cached
    p = policy(service)
    last = None
    for attempt in range(p.attempts):
        _rate_slot(service)
        try:
            with slots or nullcontext():
                response = session().get(url, params=params, timeout=timeout or cfg.REQUEST_TIMEOUT)
            status = response.status_code
            if status == 200:
                content = response.content
                if check:
                    check(content)
                if cache_kind:
                    cache.put(cache_kind, cache_key, content)
                return content
            if status == 429:                       # over the service's limit: wait as long as it asks
                wait = min(float(response.headers.get('Retry-After') or 60), 300)
                log.info(f'  {label}: the service asks us to slow down - waiting {wait:.0f} s...')
                time.sleep(wait)
                last = 'HTTP 429'
                continue
            if 400 <= status < 500 and status != 408:
                raise Refused(f'{label}: refused (HTTP {status})')
            last = f'HTTP {status}'
        except RETRY_ERRORS as e:
            last = str(e).splitlines()[0] if str(e) else type(e).__name__
        if not quiet:
            log.info(f'{label}: attempt {attempt + 1}/{p.attempts} failed: {str(last)[:120]}')
        time.sleep(wait_before_retry(service, attempt))
    raise RuntimeError(f'{label}: all {p.attempts} attempts failed ({str(last)[:120]})')


def retry(label, func, *args, service='irsa', quiet=False, **kwargs):
    """Call func(*args, **kwargs) - e.g. an astroquery query - retrying on network errors with the service's
    policy."""
    p = policy(service)
    for attempt in range(p.attempts):
        try:
            return func(*args, **kwargs)
        except RETRY_ERRORS as e:
            if not quiet:
                message = str(e).splitlines()[0] if str(e) else type(e).__name__
                log.info(f'{label}: attempt {attempt + 1}/{p.attempts} failed: {message[:120]}')
            time.sleep(wait_before_retry(service, attempt))
    raise RuntimeError(f'{label}: all {p.attempts} attempts failed.')
