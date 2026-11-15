// XARVIS viewer · scene.js
// Data from the server (/api/track, /api/spherex, /api/orbits) and updateScene(), which builds `state` (planet,
// Moon and SPHEREx positions, orbits, accuracy notes) for every frame; star proper motion and constellation lines.
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// ============================================================================ server data ===
let meta = null, track = null, snap = null, orbitsData = null, sky = null;
const busy = {track: false, snap: false, orbits: false};
const cr4 = (a, b, c, d, f) => 0.5*(2*b + (c - a)*f + (2*a - 5*b + 4*c - d)*f*f + (3*b - a - 3*c + d)*f*f*f);     // Catmull-Rom
function trackSpec(rate) { return Math.abs(rate) <= 1/24 ? {half: 16, step: 0.05} : {half: 90, step: 0.25}; }
function trackCovers(jd, t = track) { return !!t && jd >= t.jd0 + 2*t.step && jd <= t.jd0 + (t.n - 3)*t.step; }
async function loadTrack(center, spec) {
  busy.track = true;
  try {
    const n = Math.round(2*spec.half/spec.step) + 1, jd0 = center - spec.half;
    const d = await api(`/api/track?jd0=${jd0}&step=${spec.step}&n=${n}`);
    track = {jd0: d.jd0, step: d.step, n: d.n, pos: Float64Array.from(d.pos), moon: d.moon ? Float64Array.from(d.moon) : null};
  } catch (e) { console.warn('track', e); retryAt.track = performance.now() + 3000; } finally { busy.track = false; }
}
async function loadSnap(jd) {
  busy.snap = true;
  try { const d = await api('/api/spherex?jd=' + jd); d.w = cross(d.n, d.u); snap = d; ringCache.key = null; } catch (e) { console.warn('spherex', e); retryAt.snap = performance.now() + 3000; } finally { busy.snap = false; }
}
async function loadOrbits(jd) {
  busy.orbits = true;
  try { const d = await api('/api/orbits?jd=' + jd); orbitsData = {jd: d.jd, orbits: d.orbits}; } catch (e) { console.warn('orbits', e); retryAt.orbits = performance.now() + 3000; } finally { busy.orbits = false; }
}
// Keep the data the scene needs ahead of the clock (single request in flight per kind)
const retryAt = {track: 0, snap: 0, orbits: 0, meta: 0};
const deOK = () => !!(meta && meta.de_start_jd);
const outsideDE = jd => !deOK() || jd < meta.de_start_jd + 20 || jd > meta.de_end_jd - 20;     // JPL DE440s: 1849-12-26 .. 2150-01-22
const isFar = jd => Math.abs(jd - J2000) > 5000*365.25;                                         // beyond this nobody can say where planets are
function ensureData() {
  const t = performance.now(), rate = simRate(), r = Math.abs(rate), jd = sim.jd, out = outsideDE(jd);
  if (!deOK() && t > retryAt.meta) { retryAt.meta = t + 4000; api('/api/meta').then(m => { meta = m; }).catch(() => {}); }   // the kernel may still be loading
  if (r <= 1 && !out && !busy.track && t > retryAt.track) {
    const spec = trackSpec(rate), want = jd + rate*4;
    if (!track || track.step !== spec.step || !trackCovers(want) || !trackCovers(jd)) loadTrack(want, spec);
  }
  const maxAge = clamp(r*2, 0.04, 5);
  if (!busy.snap && t > retryAt.snap && (!snap || Math.abs(jd - snap.jd) > maxAge)) loadSnap(jd);
  if (r <= 1 && !out && !busy.orbits && t > retryAt.orbits && (!orbitsData || Math.abs(jd - orbitsData.jd) > 150)) loadOrbits(jd);
}

// ================================================================================== the scene ===
// `state` is what the drawing code reads. It is rebuilt every frame from the clock.
let state = null;
const kepOrbitCache = {key: null, orbits: null};
const ringCache = {key: null, ring: null, moonJD: -1e12, moon: null};
function circle(u, w, a, n = 120) {                                           // SPHEREx orbit: circle of radius a (AU) in the plane (u, w)
  const out = []; for (let i = 0; i <= n; i++) { const t = 2*Math.PI*i/n, c = Math.cos(t)*a, s = Math.sin(t)*a; out.push([c*u[0]+s*w[0], c*u[1]+s*w[1], c*u[2]+s*w[2]]); }
  return out;
}
function closeLoop(pts) {                                                     // same drift correction as the server's close_loop
  const n = pts.length - 1, d = sub(pts[n], pts[0]);
  const out = pts.map((p, i) => { const w = (i - n/2)/(n/2); return [p[0]-0.5*d[0]*w, p[1]-0.5*d[1]*w, p[2]-0.5*d[2]*w]; });
  out[n] = out[0].slice(); return out;
}
function trackEval(jd, out) {                                                 // planets -> out.P[name], moon offset
  const t = track, u = (jd - t.jd0)/t.step, i = Math.floor(u), f = u - i;
  if (i < 1 || i > t.n - 3) return false;
  const P = t.pos;
  for (let b = 0; b < 8; b++) { const pos = out.P[PLANETS[b]];
    for (let k = 0; k < 3; k++) pos[k] = cr4(P[((i-1)*8+b)*3+k], P[(i*8+b)*3+k], P[((i+1)*8+b)*3+k], P[((i+2)*8+b)*3+k], f); }
  out.moon = null;
  if (t.moon) { const M = t.moon, mo = out.moonBuf; for (let k = 0; k < 3; k++) mo[k] = cr4(M[(i-1)*3+k], M[i*3+k], M[(i+1)*3+k], M[(i+2)*3+k], f); out.moon = mo; }
  return true;
}
function moonRingOffsets(jd) {                                                // the Moon's path over one month, around `jd`
  if (!track || !trackCovers(jd - 13.7) || !trackCovers(jd + 13.7) || !track.moon) return null;
  if (Math.abs(jd - ringCache.moonJD) < 1 && ringCache.moon) return ringCache.moon;
  const tmp = {P: Object.fromEntries(PLANETS.map(n => [n, [0,0,0]])), moonBuf: [0,0,0]}, pts = [];
  for (let k = 0; k <= 60; k++) { if (!trackEval(jd - 13.66 + 27.32*k/60, tmp) || !tmp.moon) return null; pts.push(tmp.moon.slice()); }
  ringCache.moonJD = jd; ringCache.moon = closeLoop(pts); return ringCache.moon;
}
const scratch = {P: Object.fromEntries(PLANETS.map(n => [n, [0,0,0]])), moonBuf: [0,0,0], moon: null};
function updateScene() {
  if (!snap) return false;
  const jd = sim.jd, T = (jd - J2000)/36525, rate = simRate();
  const far = isFar(jd), out = outsideDE(jd), fast = Math.abs(rate) > 1 || out;     // too fast / outside JPL's DE440s: use JPL's mean orbital elements
  const warns = [];
  let from = 'track';
  if (fast || !trackCovers(jd) || !trackEval(jd, scratch)) {
    from = 'kepler';
    for (const n of PLANETS) scratch.P[n] = kepler(n, T);
    scratch.moon = null;
  }
  const planets = PLANETS.map(n => ({name: n, pos: scratch.P[n], radius_km: RADIUS_KM[n]}));
  const E = scratch.P.Earth;
  const byName = Object.fromEntries(planets.map(p => [p.name, p]));
  if (scratch.moon) { const m = scratch.moon, mp = {name: 'Moon', pos: [E[0]+m[0], E[1]+m[1], E[2]+m[2]], radius_km: RADIUS_KM.Moon}; planets.push(mp); byName.Moon = mp; }
  // SPHEREx: interpolated along the real NASA/JPL Horizons track when available, otherwise a circular orbit from the snapshot
  const a = snap.a_km/AU_KM, phi = 2*Math.PI*(jd - snap.jd)*1440/snap.period_min, cp = Math.cos(phi), sp_ = Math.sin(phi);
  let off = [0,1,2].map(k => a*(cp*snap.u[k] + sp_*snap.w[k]));
  const tr = snap.track;                                                     // real Horizons positions every 2 minutes: interpolate along them
  if (tr && tr.length > 1 && jd >= tr[0][0] && jd <= tr[tr.length - 1][0]) {
    let i = Math.min(tr.length - 2, Math.floor((jd - tr[0][0])/(tr[1][0] - tr[0][0])));
    while (i > 0 && tr[i][0] > jd) i--; while (i < tr.length - 2 && tr[i + 1][0] < jd) i++;
    const A = tr[i], B = tr[i + 1], f = (jd - A[0])/(B[0] - A[0]), ra = Math.hypot(A[1], A[2], A[3]), rb = Math.hypot(B[1], B[2], B[3]);
    const ua = [A[1]/ra, A[2]/ra, A[3]/ra], ub = [B[1]/rb, B[2]/rb, B[3]/rb], om = Math.acos(clamp(dot(ua, ub), -1, 1)), so = Math.sin(om);
    const k1 = so > 1e-9 ? Math.sin((1 - f)*om)/so : 1 - f, k2 = so > 1e-9 ? Math.sin(f*om)/so : f, rr = (ra + (rb - ra)*f)/AU_KM;
    off = [0, 1, 2].map(k => rr*(k1*ua[k] + k2*ub[k]));
  }
  const pos = [E[0]+off[0], E[1]+off[1], E[2]+off[2]], r = Math.hypot(...pos);
  if (ringCache.key !== snap) { ringCache.key = snap; ringCache.ring = circle(snap.u, snap.w, a); }
  const eclp = eclLonLat(pos), offKm = Math.hypot(...off)*AU_KM;
  const spherex = {pos, off, ring: ringCache.ring, helio_dist_au: r, ecl_lon_deg: eclp[0], ecl_lat_deg: eclp[1], earth_dist_km: offKm, altitude_km: offKm - EARTH_R_KM,
                   period_min: snap.period_min, live: snap.live, source: snap.source, showMarker: Math.abs(rate) <= 1/24};
  state = {planets, byName, E, sun_dir: norm([-E[0], -E[1], -E[2]]), spherex, moonOff: scratch.moon ? moonRingOffsets(jd) : null, from, bodies: null};
  state.bodies = planets;
  // honest warnings about what is approximate at this time / speed
  const yr = 2000 + yearsSince2000(jd);
  if (meta && jd < meta.launch_jd) warns.push('Before SPHEREx launched (12 Mar 2025): its orbit is hypothetical.');
  else if (meta && jd < meta.science_start_jd) warns.push('SPHEREx was still in orbit checkout (science began 1 May 2025).');
  else if (meta && jd > meta.science_end_jd) warns.push("After SPHEREx's nominal 25-month science mission (to mid-2027).");
  if (!snap.live) warns.push('No live NASA/JPL Horizons data for SPHEREx at this date: its orbit is modelled from the official parameters (right height and plane, position along the orbit unknown).');
  if (far) warns.push('Beyond ±5,000 years nobody can predict where the planets are: they are shown at arbitrary points on their mean orbits.');
  else if (out) warns.push("Outside NASA/JPL's DE440 ephemeris (1850–2150): planets use JPL's mean orbital elements (about 0.2°), and the Moon is hidden.");
  else if (from === 'kepler' && fast) warns.push('Fast playback: planets are on mean orbits (≤ 0.2° error), the Moon is hidden.');
  if (Math.abs(yr - 2000) > 10000) warns.push('Star motion uses proper motion only; a few nearby stars are approximate this far away.');
  if (Math.abs(rate) > 1/24 && Math.abs(rate) <= 1 && !out) warns.push('SPHEREx marker hidden at this speed (it orbits every 97 minutes).');
  if (from === 'kepler' && !fast && busy.track) warns.push('Loading ephemeris…');
  state.warns = warns;
  // orbit lines
  if (!fast && orbitsData) state.orbits = orbitsData.orbits;
  else if (fast) { const key = Math.round(T/0.5); if (kepOrbitCache.key !== key) { kepOrbitCache.key = key; kepOrbitCache.orbits = Object.fromEntries(PLANETS.map(n => [n, kepOrbit(n, key*0.5)])); } state.orbits = kepOrbitCache.orbits; }
  else state.orbits = null;
  return true;
}

// ======================================================================= star motion ===
// A star moves in a straight line at constant velocity. With only its proper motion known (Hipparcos has no
// radial velocity) the new direction is  normalize(u0 + t * w),  where w = mu_alpha* e_alpha + mu_delta e_delta
// (radians per year) and u0 is the J2000 direction. The distance cancels out.
const MAS = Math.PI/180/3600/1000;
let starYears = null;
function updateStars(force) {
  if (!sky) return;
  const yrs = yearsSince2000(sim.jd);
  if (!force && starYears !== null && Math.abs(yrs - starYears) < 0.005) return;
  starYears = yrs;
  const n = sky.n, u0 = sky.u0, w = sky.w, v = sky.v;
  for (let i = 0; i < n; i++) {
    const x = u0[i*3] + yrs*w[i*3], y = u0[i*3+1] + yrs*w[i*3+1], z = u0[i*3+2] + yrs*w[i*3+2], l = 1/Math.hypot(x, y, z);
    v[i*3] = x*l; v[i*3+1] = y*l; v[i*3+2] = z*l;
  }
  if (Math.abs(yrs - sky.linesYears) > 0.02 || force) rebuildLines(yrs);
}
// Unit vectors along the great-circle arc between two points, <= 1.5 degrees apart (segments can be 30+ degrees)
function arcPoints(a, b, out) {
  const d = clamp(dot(a, b), -1, 1), th = Math.acos(d), n = Math.max(1, Math.ceil(th/(1.5*D2R))), sn = Math.sin(th);
  for (let k = 0; k < n; k++) { const t = k/n, w0 = sn < 1e-9 ? 1 - t : Math.sin((1 - t)*th)/sn, w1 = sn < 1e-9 ? t : Math.sin(t*th)/sn;
    out.push(w0*a[0] + w1*b[0], w0*a[1] + w1*b[1], w0*a[2] + w1*b[2]); }
}
function rebuildLines(yrs) {
  const v = sky.v, star = i => [v[i*3], v[i*3+1], v[i*3+2]], lines = [];
  for (const poly of sky.figures) { const pts = [];
    for (let k = 0; k + 1 < poly.length; k++) arcPoints(star(poly[k]), star(poly[k+1]), pts);
    pts.push(...star(poly[poly.length - 1])); lines.push(Float32Array.from(pts)); }
  sky.lines = lines; sky.linesYears = yrs;
}
const BIN_COL = ['#9bb0ff','#cad8ff','#f8f7ff','#fff4e8','#ffe7bf','#ffd2a1','#ffb878','#ff9e5e'];
async function loadSky() {
  const d = await api('/static/sky.json');
  const n = d.stars.length, u0 = new Float64Array(n*3), w = new Float64Array(n*3), v = new Float64Array(n*3), r = new Float32Array(n), bins = BIN_COL.map(() => []);
  const hipIdx = new Map(), bright = [];
  d.stars.forEach(([ra, dec, mag, bv, hip, pmra, pmdec], i) => {
    const a = ra*D2R, dc = dec*D2R;
    const ea = eqVec2ecl([-Math.sin(a), Math.cos(a), 0]), ed = eqVec2ecl([-Math.sin(dc)*Math.cos(a), -Math.sin(dc)*Math.sin(a), Math.cos(dc)]), p = eq2ecl(a, dc);
    for (let k = 0; k < 3; k++) { u0[i*3+k] = p[k]; v[i*3+k] = p[k]; w[i*3+k] = (pmra*ea[k] + pmdec*ed[k])*MAS; }
    r[i] = Math.max(0.55, 2.8 - 0.42*mag);
    bins[clamp(Math.round((bv + 0.3)/0.3), 0, 7)].push(i);
    hipIdx.set(hip, i); if (mag <= 4.6) bright.push(i);
  });
  const nameOf = new Map(), names = [];
  d.names.forEach(([hip, name]) => { const i = hipIdx.get(hip); if (i !== undefined) { nameOf.set(i, name); names.push({i, name}); } });
  names.sort((p, q) => d.stars[p.i][2] - d.stars[q.i][2]);          // brightest first, so they win label space
  const figures = []; for (const abbr in d.cl) for (const poly of d.cl[abbr]) figures.push(poly);
  const bounds = d.bounds.map(flat => { const o = new Float32Array(flat.length/2*3);
    for (let k = 0; k < flat.length; k += 2) o.set(eq2ecl(flat[k]*D2R, flat[k+1]*D2R), k/2*3); return o; });
  sky = {n, u0, w, v, r, bins: bins.map(b => Int32Array.from(b)), bright, nameOf, names, figures, lines: [], linesYears: 1e9, bounds,
         mag: d.stars.map(s => s[2]), bv: d.stars.map(s => s[3]), hip: d.stars.map(s => s[4]), pm: d.stars.map(s => [s[5], s[6]]),
         labels: d.labels.map(([ra, dec, name, en]) => ({v: eq2ecl(ra*D2R, dec*D2R), name, en}))};
  $('sources').textContent = d.attribution + ' Planets and Moon: NASA/JPL DE440s ephemeris. SPHEREx orbit: NASA/JPL Horizons. Asteroids, Trojans and Kuiper belt: NASA/JPL Small-Body Database. SPHEREx facts: IPAC SPHEREx Explanatory Supplement v2.1.';
  updateStars(true);
}
