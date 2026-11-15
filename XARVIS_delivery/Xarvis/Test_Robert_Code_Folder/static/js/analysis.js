// XARVIS viewer · analysis.js
// Go to sky coordinates (test targets from ESA Gaia DR3, typed RA/Dec) and the sky point analysis popup: visibility,
// target motion, known asteroids in the patch (JPL SB Ident), the patch map and the SPHEREx image time-lapse
// (requestAnalysis() sends the payload to the backend pipeline and plays the images it returns).
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// ======================================================================= go to sky coordinates ===
// Type RA/Dec (or pick a test target) to look at a patch of sky: the view turns to it, a marker shows the patch, and the
// analysis popup opens for it. Test targets come from ESA Gaia DR3 (static/targets.json), and are moved with their
// proper motion to the date shown; WISE 0855-0714 is too faint for Gaia and uses the challenge's coordinates.
let targets = [], skyTarget = null, viewAnim = null;
const SPHEREX_PIX = 6.15;
const patchPix = () => clamp(Math.round(+$('gPix').value || 50), 10, 2000), patchHW = () => patchPix()*SPHEREX_PIX/3600/2;   // half-width in degrees
function showPatchInfo() { const px = patchPix(), am = px*SPHEREX_PIX/60; setText('gPixInfo', `${px} × ${px} px at 6.15″ per pixel = ${am < 60 ? am.toFixed(1) + '′' : (am/60).toFixed(2) + '°'} square. Larger patches show faster-moving objects but take longer to load.`); }                                         // arcsec per pixel (IPAC Explanatory Supplement, Table 2)
function parseRA(t) {
  t = t.trim().toLowerCase().replace(/[−–]/g, '-'); if (!t) return null;
  const n = t.match(/[\d.]+/g); if (!n) return null;
  if (/[h:]/.test(t) || /\s/.test(t) || n.length > 1) { const [h, m = 0, s = 0] = n.map(Number); const v = (h + m/60 + s/3600)*15; return h < 24 && m < 60 && s < 60 ? v : null; }
  const v = Number(n[0]); return v >= 0 && v < 360 ? v : null;            // a single number: degrees
}
function parseDec(t) {
  t = t.trim().replace(/[−–]/g, '-'); if (!t) return null;
  const neg = /^-/.test(t), n = t.match(/[\d.]+/g); if (!n) return null;
  const [d, m = 0, s = 0] = n.map(Number), v = (neg ? -1 : 1)*(d + m/60 + s/3600);
  return Math.abs(v) <= 90 && m < 60 && s < 60 ? v : null;
}
function targetEqAt(T, jd) {                                      // [ra, dec] in degrees at date jd (linear proper motion, like the star field)
  if (T.pmra == null || T.epoch == null) return [T.ra, T.dec];
  const a = T.ra*D2R, d = T.dec*D2R, u = [Math.cos(d)*Math.cos(a), Math.cos(d)*Math.sin(a), Math.sin(d)], e = [-Math.sin(a), Math.cos(a), 0], nn = [-Math.sin(d)*Math.cos(a), -Math.sin(d)*Math.sin(a), Math.cos(d)];
  const yrs = (jd - 2451545.0)/365.25 + 2000 - T.epoch, k = D2R/3.6e6*yrs, v = norm([0, 1, 2].map(i => u[i] + k*(T.pmra*e[i] + T.pmdec*nn[i])));
  return [(Math.atan2(v[1], v[0])/D2R + 360) % 360, Math.asin(v[2])/D2R];
}
const targetDir = (T, jd = sim.jd) => { const [ra, dec] = targetEqAt(T, jd); return eq2ecl(ra*D2R, dec*D2R); };
async function loadTargets() {
  try { targets = (await api('/static/targets.json')).targets;
    $('gTarget').innerHTML = '<option value="">Choose a target…</option>' + targets.map((t, i) => `<option value="${i}">${esc(t.name)}</option>`).join(''); }
  catch (e) { console.warn('targets', e); }
}
function lookToward(f) {                                          // turn the view smoothly so direction f is in the middle of the screen
  spin = false; $('spin').classList.remove('on'); $('spinPanel').hidden = true;
  const f0 = norm([-R[2][0], -R[2][1], -R[2][2]]), ang = Math.acos(clamp(dot(f0, f), -1, 1)), t0 = performance.now();
  viewAnim = t => { const k = Math.min(1, (t - t0)/1300), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2)/2;
    let v = f; if (ang > 1e-6 && ang < Math.PI - 1e-3) { const sn = Math.sin(ang); v = [0, 1, 2].map(i => (Math.sin((1 - e)*ang)*f0[i] + Math.sin(e*ang)*f[i])/sn); }
    R = lookAt(norm(v), [0, 0, 1]); return k >= 1; };
}
function showSkyTarget(T, hw) {
  skyTarget = {T, hw}; lookToward(targetDir(T));
  $('gotoChip').innerHTML = `<span class="chip"><b>${esc(T.name)}</b><button class="ghost cf" id="gReopen" title="Open the analysis again">Open</button><button class="ghost cx" id="gClear" title="Remove the marker">✕</button></span>`;
  $('gReopen').onclick = () => openAnalysisDir(targetDir(T), T, hw); $('gClear').onclick = () => { skyTarget = null; $('gotoChip').innerHTML = ''; };
  setTimeout(() => openAnalysisDir(targetDir(T), T, hw), 1350);
}
function goToTyped() {
  const i = $('gTarget').value, hw = patchHW(), ra = parseRA($('gRA').value), dec = parseDec($('gDec').value);
  if (ra == null || dec == null) { $('gMsg').className = 'err'; $('gMsg').textContent = ra == null ? 'Right ascension not understood: try 08h55m10.8s, 08 55 10.8 or 133.795 (degrees).' : 'Declination not understood: try -7d14m42.5s, -07 14 42.5 or -7.245.'; return; }
  $('gMsg').className = 'dim';
  const T0 = i !== '' ? targets[+i] : null;
  const same = T0 && Math.abs(ra - T0.ra) < 1e-4 && Math.abs(dec - T0.dec) < 1e-4;      // untouched preset: keep its Gaia motion
  showSkyTarget(same ? T0 : {name: `RA ${fmtRA(ra, false)} · Dec ${fmtDec(dec, false)}`, ra, dec, typed: true, source: 'typed coordinates (J2000)'}, hw);
}
function drawSkyTarget() {
  if (!skyTarget) return;
  const SK = skyScale(), T = skyTarget.T, d = targetDir(T), q = skyXY(toView(d), SK); if (!q) return;
  const half = Math.max(9, skyTarget.hw*D2R*SK), pulse = 3*Math.sin(performance.now()/300);
  ctx.strokeStyle = '#5fd6f0'; ctx.lineWidth = 1.6; ctx.strokeRect(q[0] - half, q[1] - half, 2*half, 2*half);
  ctx.beginPath(); for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.moveTo(q[0] + dx*(half + 4 + pulse), q[1] + dy*(half + 4 + pulse)); ctx.lineTo(q[0] + dx*(half + 14 + pulse), q[1] + dy*(half + 14 + pulse)); } ctx.stroke();
  ctx.font = 'bold 12px system-ui'; const w = measure('bold 12px system-ui', T.name); ctx.fillStyle = '#02040acc'; ctx.fillRect(q[0] + half + 16, q[1] - half - 20, w + 8, 17);
  ctx.fillStyle = '#9fe6f7'; ctx.fillText(T.name, q[0] + half + 20, q[1] - half - 7);
  addHit(q[0], q[1], half + 6, 0.6, () => tipHTML(esc(T.name), 'sky patch', [['RA / Dec (J2000)', fmtRADec(d)], ['Patch', (2*skyTarget.hw*60).toFixed(0) + '′ × ' + (2*skyTarget.hw*60).toFixed(0) + '′']], 'Double-click or use "Open" to analyse it.'));
}

// ========================================================================= analysis popup ===
// Double-click: take the sky direction under the cursor (as seen from Earth: the stars are effectively at infinity, so
// the geocentric direction equals the direction drawn), package it, show it and send it to the backend (requestAnalysis).
function analysisPayload(dir, T = null, hw = 0.25) {
  const eq = ecl2eqVec(dir), [ra, dec] = ecl2eq(dir), [lam, beta] = eclLonLat(dir), [l, b] = ecl2gal(dir);
  const sp = state.spherex, elong = Math.acos(clamp(dot(dir, state.sun_dir), -1, 1))/D2R;
  const r6 = v => +v.toFixed(6);
  return {
    request: 'sky_point_analysis', version: 1,
    time: {utc: fmtDate(sim.jd) + ' UTC', jd_utc: +sim.jd.toFixed(6), simulated: !sim.live},
    target: {frame: 'geocentric, J2000 equatorial (ICRS-aligned)', ra_deg: r6(ra), dec_deg: r6(dec), unit_vector: eq.map(r6),
             spherex_visibility: visibilityOf(dir), ecliptic_lon_deg: r6(lam), ecliptic_lat_deg: r6(beta), galactic_l_deg: r6(l), galactic_b_deg: r6(b), sun_elongation_deg: +elong.toFixed(3)},
    patch: {half_width_deg: hw, size: (2*hw*60).toFixed(0) + ' x ' + (2*hw*60).toFixed(0) + ' arcmin'},
    named_target: T ? {name: T.name, source: T.source || null, gaia_dr3: T.gaia_dr3 || null, ra_deg_at_epoch: T.ra, dec_deg_at_epoch: T.dec, epoch: T.epoch ?? null,
                       pmra_mas_yr: T.pmra ?? null, pmdec_mas_yr: T.pmdec ?? null, parallax_mas: T.parallax ?? null, gaia_g_mag: T.g ?? null} : null,
    spherex: {geocentric_km_j2000_equatorial: ecl2eqVec(sp.off).map(v => +(v*AU_KM).toFixed(1)), altitude_km: +sp.altitude_km.toFixed(1), orbit_source: sp.source},
  };
}
// ------------------------------------------------------------------ SPHEREx image time-lapse ---
// The backend (Jonah_Code_File, through app.py) downloads the SPHEREx images of the patch from NASA/IPAC IRSA, measures
// them and returns one PNG per image, all on one grid centred on the target (north up, east left), so they line up.
// That takes seconds to minutes, so it is a background job: POST /api/analysis starts it (the popup's payload plus
// max_images), GET /api/analysis/<job> is polled once a second for progress, and when it is done its summary lists the
// frames: time, detector, wavelength, signal-to-noise and flags for each. Nothing is requested when the image box is
// hidden (outside SPHEREx's view, or a date with no data).
const tl = {req: 0, job: null, frames: [], imgs: [], i: 0, playing: true, timer: null, fps: 4, state: 'idle', tipTimer: null, tip: 0};
// While the data comes from NASA: a loading card with the five steps, a progress bar, the pipeline's own messages and
// tips (what each group of users can do with what is shown - see AUDIENCE.md). "Minimize" keeps it loading in a badge.
const TL_STEPS = ['Search the SPHEREx archive', 'Download and measure the images', 'Compare with the catalogs and map the sky', 'Make the images', 'Load them here'];
const TL_TIPS = [
  'SPHEREx maps the whole sky every six months in 102 infrared colours (0.75–5 µm). XARVIS shows you every image it took of the point you picked, in time order.',
  'Red circle: where the target was found in this image. Dashed blue: where it was in the first image (T1). If they drift apart, it moved.',
  'Yellow rings are NEW objects: in no catalog (ESA Gaia DR3, CatWISE2020, Hipparcos), not a known asteroid, comet or planet, and not a flagged detector artifact. Click one, then "Track this object".',
  'Colour compares two wavelengths SPHEREx took at the same instant: redder means relatively brighter in the infrared (cool stars, brown dwarfs, dust); bluer means hotter.',
  'Students and teachers: nearby stars such as Barnard\'s Star move several arcseconds a year. Pick one under "Go to sky coordinates → Test targets" and watch it shift across the survey.',
  'Asteroid hunters: a NEW object classed "moving" has a fitted path across the images. Track it to collect every SPHEREx image along that path.',
  'Researchers: the per-image flags (TRANSIENT, OUTLIER, saturated, bad pixels) tell you which measurements to trust before you download the data from IRSA.',
  'Variable stars: each image\'s brightness is compared with the object\'s own spectrum; a "BRIGHTNESS CHANGE" in the frame title marks an image that stands out.',
  'Every position is checked against ESA Gaia DR3: the "Positions vs Gaia" line shows how well SPHEREx\'s pointing matches, typically well under one 6.15″ pixel.',
  'The first look at a patch downloads from NASA/IPAC IRSA (seconds to a minute or two). Asking again is instant: the results are kept.',
  'NASA/IPAC IRSA is sometimes busy and answers "502". XARVIS retries automatically; if it still fails, "Try again" a minute later usually works.',
  'More images (the "Images" menu) cover more of the survey\'s visits but take longer to collect. 8 is quickest; 24 is a good balance.',
  'Keyboard, once the images are in: ← → step through them, Space plays and pauses, Home / End jump to the first / last.'];
// How long a NEW analysis can take (measured: 8 to 50 images up to about a minute, 100 or more up to about 3 minutes;
// a tracked asteroid searches a whole time window, so it can take longer). Repeats are instant: results are kept.
function tlTimeText(count) {
  const high = count === 'all' || +count >= 100;
  return high ? `With ${count === 'all' ? 'all the' : count} images, a new analysis can take up to about 3 minutes.`
              : `With ${count} images, a new analysis can take up to about a minute.`;
}
function tlTimeHint() {
  $('tlTime').textContent = tlTimeText($('tlCount').value) + ' Low counts (8 to 50) take up to about a minute; high counts (100 or more) up to about 3. Opening one again is instant.';
}
function tlTipShow(k) { tl.tip = ((k % TL_TIPS.length) + TL_TIPS.length) % TL_TIPS.length; $('tlTip').textContent = TL_TIPS[tl.tip]; }
function tlTips(on) {
  clearInterval(tl.tipTimer); tl.tipTimer = null;
  if (on) { if (!$('tlTip').textContent) tlTipShow(Math.floor(Math.random()*TL_TIPS.length)); tl.tipTimer = setInterval(() => tlTipShow(tl.tip + 1), 9000); }
}
function tlStep(msg, done, total) {                           // which of the five steps a progress message belongs to
  if (/^Loading the images/.test(msg)) return 4;
  if (/^Making the images/.test(msg)) return 3;
  if (total && done < total) return 1;
  if (total && done >= total) return 2;
  return 0;
}
function tlReset() {
  clearInterval(tl.timer); tl.timer = null; tlTips(false); tl.frames = []; tl.imgs = []; tl.state = 'idle';
  $('tlSel').innerHTML = ''; $('tlDb').innerHTML = ''; $('tlPlayer').hidden = $('tlData').hidden = $('tlLoad').hidden = $('tlErr').hidden = true; $('tlMsg').textContent = '';
}
function tlProgress(msg, done, total) {                         // no total yet: a moving bar, so the page never looks frozen
  const step = tlStep(msg, done, total);
  tl.state = 'loading'; $('tlErr').hidden = true; $('tlLoad').hidden = false; tlTips(true);
  $('tlLoadTitle').textContent = tl.payload && tl.payload.track ? `Collecting SPHEREx data on ${tl.payload.track.name}` : 'Collecting SPHEREx data from NASA';
  $('tlLoadTime').textContent = tlTimeText($('tlCount').value) + (tl.payload && tl.payload.track && tl.payload.track.kind === 'asteroid' ? ' Tracking an asteroid searches its whole time window, so it can take a little longer.' : '') + ' You can minimize this and keep exploring.';
  [...$('tlSteps').children].forEach((li, k) => { li.className = k < step ? 'done' : k === step ? 'on' : ''; });
  $('tlMsg').textContent = msg; $('tlBar').classList.toggle('wait', !total || step === 2 || step === 3);
  $('tlFill').style.width = total && step !== 2 && step !== 3 ? (100*done/total).toFixed(1) + '%' : '';
  tl.miniText = `${TL_STEPS[step]}${total && (step === 1 || step === 4) ? ` · ${done} of ${total}` : ''}`; tlMini(tl.miniText);
}
function tlFail(msg) {
  clearInterval(tl.timer); tl.timer = null; tlTips(false); tl.state = 'error';
  $('tlPlayer').hidden = $('tlData').hidden = $('tlLoad').hidden = true; $('tlErr').hidden = false; $('tlErrMsg').textContent = msg;
  tlMini(msg);
}
// ---- minimized: a badge with a spinner while it loads, a check mark when the images are ready; click to open it again
function tlMini(text) {
  const icon = $('anaMiniIcon'), done = tl.state === 'done', bad = tl.state === 'error';
  icon.className = 'spin' + (done ? ' ok' : bad ? ' bad' : '');
  $('anaMini').className = done ? 'done' : bad ? 'err' : '';
  $('anaMiniTitle').textContent = done ? 'SPHEREx images ready' : bad ? 'The analysis failed' : 'Collecting SPHEREx data…';
  $('anaMiniText').textContent = done ? `${tl.frames.length} images · ${$('anaTitle').textContent} · click to open` : text;
}
function minimizeAnalysis() { if ($('anaBack').hidden) return; $('anaBack').hidden = true; $('anaMini').hidden = false; tlMini(tl.state === 'error' ? $('tlErrMsg').textContent : tl.miniText || ''); }
function restoreAnalysis() { $('anaMini').hidden = true; $('anaBack').hidden = false; $('anaClose').focus({preventScroll: true}); }
async function tlJSON(url, opt) {                               // JSON from our server, or a readable error
  let r; try { r = await fetch(url, opt); } catch (e) { throw new Error('The XARVIS server did not answer. Check that it is still running, then try again.'); }
  let d = null; try { d = await r.json(); } catch (e) { /* not JSON: handled below */ }
  if (!r.ok || !d) throw new Error((d && d.error) || `The server could not start the analysis (HTTP ${r.status}).`);
  return d;
}
async function requestAnalysis(payload) {
  const id = ++tl.req; tlReset();
  if (!payload || $('anaViz').hidden) return null;
  payload = {...payload}; tl.payload = payload;
  if (!payload.track) { tl.base = payload; tl.sel = null; }            // the patch itself (not an object picked in it)
  tlTrackBar();
  const count = $('tlCount').value; payload.max_images = count === 'all' ? null : +count;
  $('anaJSON').textContent = JSON.stringify(payload, null, 2);
  tlProgress(payload.track ? `Asking for the SPHEREx images of ${payload.track.name}…` : 'Asking for the SPHEREx images of this patch…', 0, 0);
  const t0 = performance.now();
  try {
    const start = await tlJSON('/api/analysis', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    for (;;) {
      if (id !== tl.req) return null;                           // the popup was closed or another point was opened
      const d = await tlJSON('/api/analysis/' + start.job);
      if (id !== tl.req) return null;
      if (d.status === 'error') throw new Error(d.error || 'The analysis failed.');
      if (d.status === 'done') { await tlShow(id, start.job, d.summary, payload); return d.summary; }
      const slow = !d.total && performance.now() - t0 > 20000 ? ' The archive is slow to answer; still trying.' : '';
      tlProgress((d.message || 'Working…') + slow, d.done, d.total);
      await new Promise(res => setTimeout(res, 1000));
    }
  } catch (e) { if (id === tl.req) tlFail(e.message); return null; }
}
async function tlShow(id, job, S, payload) {
  const n = S.frames.length;
  if (!n) throw new Error('SPHEREx has no usable image of this point yet.');
  let loaded = 0; tlProgress(`Loading the images: 0 of ${n}`, 0, n);
  const imgs = await Promise.all(S.frames.map(f => new Promise(res => { const im = new Image();
    im.onload = im.onerror = () => { if (id === tl.req) tlProgress(`Loading the images: ${++loaded} of ${n}`, loaded, n); res(im); }; im.src = `/api/analysis/${job}/${f.png}`; })));
  if (id !== tl.req) return;
  tl.job = job; tl.frames = S.frames; tl.imgs = imgs; tl.circles = S.circles || {red: 'object found in this frame', blue: 'object at T1'};
  tl.S = S; tl.sel = null; $('tlSvg').setAttribute('viewBox', `0 0 ${S.image_px} ${S.image_px}`);
  const hasMap = S.frames.some(f => f.objects); $('tlMapBar').hidden = !hasMap; $('tlMapKey').hidden = !hasMap || !$('tlMapOn').checked;
  tlDatabase(S);
  const mjd = payload.time.jd_utc - 2400000.5; let near = 0;      // start on the image taken closest to the date shown
  S.frames.forEach((f, i) => { if (Math.abs(f.mjd - mjd) < Math.abs(S.frames[near].mjd - mjd)) near = i; });
  tlTips(false); $('tlLoad').hidden = true; $('tlMsg').textContent = ''; $('tlPlayer').hidden = $('tlData').hidden = false; $('tlSlide').max = n - 1;
  tl.state = 'done'; tlMini(''); loadAnalysed();             // (the viewer's "Analysed patches" layer gains this one)
  const asked = Math.round(2*payload.patch.half_width_deg*3600/SPHEREX_PIX), am = (S.patch_px*S.pixel_arcsec/60).toFixed(1);
  $('tlNote').innerHTML = `${n} of the ${S.images_available ?? n} SPHEREx images that cover this point${S.images_available > n ? ', spread over all its survey dates' : ''}, in time order. ` +
    `Each is ${S.patch_px} × ${S.patch_px} SPHEREx pixels (${am}′, 6.15″ each) centred on ${S.target.moving ? 'its predicted position in that image (JPL Horizons / fitted path), so the object stays in the middle' : 'the target'}, N ↑ · E ←, shown like the XARVIS desktop viewer: ` +
    `colour from the simultaneous image on the other detector arm, sharpened and upscaled, the target at the same brightness and colour in every frame (display only; the measurements use the original pixels).` +
    (asked > S.patch_px ? ` The patch is ${asked} pixels wide; the images are limited to ${S.patch_px}.` : '') + `<br>${esc(S.source)}.`;
  tlFrame(near); tlPlay(n > 1);
}
function tlFrame(i) {
  const n = tl.frames.length; if (!n) return; tl.i = i = ((i % n) + n) % n; const f = tl.frames[i];
  $('tlImg').src = tl.imgs[i].src; $('tlSlide').value = i; $('tlNum').textContent = `T${i + 1} / ${n}`;
  $('tlTitle').textContent = (f.title || [`T${i + 1} of ${n}`]).join('\n');           // the desktop viewer's frame title
  const c = f.colour_um;
  $('tlKey').innerHTML = `<span><i class="ring"></i>${esc(tl.circles.red)}</span><span><i class="ring ref"></i>${esc(tl.circles.blue)}</span>` +
    (c ? `<span>bluer: brighter at ${c[0].toFixed(2)} µm<i class="bar"></i>redder: brighter at ${c[1].toFixed(2)} µm</span>` : '<span>no simultaneous frame: brightness only</span>');
  const flags = [...(f.flags_in_aperture || []), ...(f.saturated ? ['saturated'] : [])];
  const rows = [['Time (UTC)', esc(f.time_utc)], ['Detector', f.detector ?? '—'], ['Wavelength', f.wavelength_um == null ? '—' : f.wavelength_um.toFixed(3) + ' µm'],
                ['Signal-to-noise at the target', f.snr == null ? 'not measurable' : f.snr.toFixed(1)],
                ['Flags at the target', flags.length ? esc(flags.join(', ')) + (f.n_bad_in_aperture ? ` · ${f.n_bad_in_aperture} bad pixel${f.n_bad_in_aperture === 1 ? '' : 's'}` : '') : 'none']];
  $('tlInfo').innerHTML = rows.map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
  $('tlMapText').textContent = f.map_text || '';
  tlDraw();
}
// ---- what the backend found in the images: the sky map (every object named; NEW = in no catalog), the circles, and
// picking an object to track it (like the desktop viewer: click an object, then track it; click empty sky to go back)
const SVGNS = 'http://www.w3.org/2000/svg', MAP_COLOUR = {new: '#ffe600', 'small body': '#ff4dff', planet: '#7cfc00', artifact: '#999999', Gaia: '#4fd2ff', CatWISE: '#ff9a3c', Hipparcos: '#ffffff'};
function svgEl(tag, attrs, parent) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }
function tlDraw() {
  const svg = $('tlSvg'), f = tl.frames[tl.i]; if (!f) return;
  svg.replaceChildren();
  if ($('tlMapOn').checked && f.objects) for (const o of f.objects) {
    const g = svgEl('g', {class: 'o'}, svg), c = o.kind === 'star' ? MAP_COLOUR[o.named ? 'Hipparcos' : o.source] || '#fff' : MAP_COLOUR[o.kind] || '#fff', x = o.x, y = o.y;
    const line = {stroke: c, 'stroke-width': 1.6, fill: 'none'};
    if (o.kind === 'star' && o.named) svgEl('path', {d: starPath(x, y, 7), ...line}, g);
    else if (o.kind === 'star' && o.source === 'CatWISE') svgEl('path', {d: `M${x-4} ${y-4}L${x+4} ${y+4}M${x+4} ${y-4}L${x-4} ${y+4}`, ...line}, g);
    else if (o.kind === 'star') svgEl('path', {d: `M${x-5} ${y}H${x+5}M${x} ${y-5}V${y+5}`, ...line}, g);
    else if (o.kind === 'artifact') svgEl('rect', {x: x - 5, y: y - 5, width: 10, height: 10, ...line, 'stroke-width': 1.2}, g);
    else if (o.kind === 'new') { svgEl('circle', {cx: x, cy: y, r: 13, stroke: c, 'stroke-width': 7, 'stroke-opacity': 0.25, fill: 'none'}, g); svgEl('circle', {cx: x, cy: y, r: 13, stroke: c, 'stroke-width': 2.6, fill: 'none'}, g); }
    else svgEl('path', {d: `M${x} ${y-7}L${x+7} ${y}L${x} ${y+7}L${x-7} ${y}Z`, ...line}, g);        // small body / planet
    if (o.kind === 'new') {                                                // labelled, impossible to miss
      const t = o.label, w = 6.6*t.length + 8, lx = Math.min(x + 16, +tl.S.image_px - w - 2), ly = Math.max(y - 26, 2);
      svgEl('line', {x1: x, y1: y, x2: lx, y2: ly + 9, stroke: c, 'stroke-width': 1.4}, g);
      svgEl('rect', {x: lx, y: ly, width: w, height: 17, rx: 3, fill: c, 'fill-opacity': 0.95}, g);
      svgEl('text', {x: lx + 4, y: ly + 12.5, class: 'tag'}, g).textContent = t;
    } else if ((o.kind === 'star' && o.named) || o.kind === 'small body' || o.kind === 'planet') {
      const name = o.kind === 'star' ? o.label : (o.label.split('(')[0].trim() || o.label) + (o.detected ? '' : ' (not detected)');
      svgEl('text', {x: x + 9, y: y - 7, fill: c}, g).textContent = name;
    }
    svgEl('title', {}, g).textContent = o.label;
  }
  // the circles: the analysed object, or the object picked in the images (red = this frame, blue = its first sighting)
  let red = f.marks && f.marks.red, blue = f.marks && f.marks.blue;
  if (tl.sel) { const O = tl.S.objects[tl.sel], here = (f.objects || []).find(o => o.id === tl.sel);
    red = here ? [here.x, here.y] : null;
    const first = O && tl.frames[O.frames[0] - 1], at = first && (first.objects || []).find(o => o.id === tl.sel); blue = at ? [at.x, at.y] : null; }
  const R = (f.marks && f.marks.radius) || 16, RB = (f.marks && f.marks.blue_radius) || 21;
  if (blue) svgEl('circle', {cx: blue[0], cy: blue[1], r: RB, stroke: '#00bfff', 'stroke-width': 2.2, 'stroke-dasharray': '7 5', fill: 'none', 'pointer-events': 'none'}, svg);
  if (red) svgEl('circle', {cx: red[0], cy: red[1], r: R, stroke: '#ff0000', 'stroke-width': 2.2, fill: 'none', 'pointer-events': 'none'}, svg);
}
function tlClick(e) {                                         // the nearest object within reach, or empty sky: back to the target
  const f = tl.frames[tl.i], svg = $('tlSvg'), r = svg.getBoundingClientRect(), W = +tl.S.image_px;
  const x = (e.clientX - r.left)/r.width*W, y = (e.clientY - r.top)/r.height*W, reach = 14*W/r.width;
  let best = null, bd = reach;
  if ($('tlMapOn').checked) for (const o of f.objects || []) { const d = Math.hypot(o.x - x, o.y - y); if (d < bd) { bd = d; best = o; } }
  tlSelect(best ? best.id : null);
}
function starPath(x, y, r) { let d = ''; for (let k = 0; k < 10; k++) { const a = -Math.PI/2 + k*Math.PI/5, q = k % 2 ? r*0.45 : r; d += (k ? 'L' : 'M') + (x + q*Math.cos(a)).toFixed(1) + ' ' + (y + q*Math.sin(a)).toFixed(1); } return d + 'Z'; }
const KIND_TEXT = {Gaia: 'star in ESA Gaia DR3', CatWISE: 'source in CatWISE2020 only (red / dusty, no Gaia counterpart)', Hipparcos: 'named star (ESA Hipparcos)',
  'small body': 'known asteroid / comet (NASA/JPL SB Ident, SBDB, Horizons)', planet: 'planet / Moon (NASA/JPL DE440s)', artifact: 'flagged detector artifact (SPHEREx FLAGS plane)', new: 'NEW object: in no catalog, not a known asteroid, comet or planet, not a flagged artifact'};
function tlSelect(id) {
  tl.sel = id && tl.S.objects[id] ? id : null; tlDraw();
  for (const tr of document.querySelectorAll('#tlDb tr.pick')) tr.classList.toggle('on', tr.dataset.id === tl.sel);
  const O = tl.sel && tl.S.objects[tl.sel];
  if (!O) { $('tlSel').innerHTML = ''; $('tlKey').querySelector('span') && tlKeyText(); return; }
  const what = O.kind === 'star' ? KIND_TEXT[O.named ? 'Hipparcos' : O.source] : KIND_TEXT[O.kind];
  const ts = O.frames.map(n => 'T' + n), seen = O.detected_in.length;
  $('tlSel').innerHTML = `<b>${esc(O.label)}</b><div class="dim">${esc(what || O.kind)}</div>` + (O.info ? `<div>${esc(O.info)}</div>` : '') +
    `<div class="dim">In view in ${ts.length} of ${tl.frames.length} images (${esc(ts.slice(0, 12).join(', '))}${ts.length > 12 ? ' …' : ''}); detected in ${seen}. Red circle: this image; blue dashed: its first sighting.</div>` +
    `<div class="row">${O.track ? `<button id="tlTrackGo" class="primary">Track this object</button>` : `<span class="dim">Can't be tracked: ${esc(O.reason || 'not supported')}.</span>`}<button id="tlUnsel" class="ghost">Back to the target</button></div>`;
  if (O.track) $('tlTrackGo').onclick = () => { tlPlay(false); requestAnalysis({...tl.base, track: O.track}); };
  $('tlUnsel').onclick = () => tlSelect(null);
  tlKeyText();
}
function tlKeyText() {                                         // the circle legend follows what the circles show
  const O = tl.sel && tl.S.objects[tl.sel], spans = $('tlKey').querySelectorAll('span');
  if (spans.length < 2) return;
  spans[0].lastChild.textContent = O ? `${O.label} in this image` : tl.circles.red;
  spans[1].lastChild.textContent = O ? `${O.label} at its first sighting` : tl.circles.blue;
}
function tlTrackBar() {                                        // "Tracking X" with the way back to the patch
  const P = tl.payload, on = !!(P && P.track);
  $('tlTrack').hidden = !on; if (!on) return;
  $('tlTrack').innerHTML = `<span>Tracking <b>${esc(P.track.name)}</b>, picked in the images of this patch</span><button id="tlBackBase">← Back to the patch</button>`;
  $('tlBackBase').onclick = () => requestAnalysis(tl.base);
}
function tlDatabase(S) {                                       // what the images were compared with, and the NEW objects
  const D = S.database || {}, rows = [];
  if (!D.sky_map && !D.field_check) {
    $('tlDb').innerHTML = `<div class="sbh">Compared with the catalogs</div><div class="dim">${S.target.moving ? 'This object moves, so each image is centred on its predicted position; the sky map and catalog comparison run for targets fixed on the sky (as in the desktop viewer).' : 'The catalogs could not be reached for this run.'}</div>` +
      (D.notes || []).map(n => `<div class="dim">${esc(n)}</div>`).join('');
    return;
  }
  const cs = D.catalog_stars || {};
  rows.push(['Stars in the field', ['Gaia', 'CatWISE', 'Hipparcos'].filter(k => k in cs).map(k => `${cs[k]} ${{Gaia: 'ESA Gaia DR3', CatWISE: 'CatWISE2020', Hipparcos: 'ESA Hipparcos'}[k]}`).join(' · ') || '—']);
  if (D.pointing) rows.push(['Positions vs Gaia DR3', D.pointing.matches ? `${D.pointing.matches} matches · median offset ${D.pointing.median_offset_arcsec.toFixed(2)}″` : 'no matches']);
  if (D.sky_map) {
    const sb = D.known_small_bodies || [];
    rows.push(['Known asteroids / comets seen', sb.length ? sb.map(b => esc(b.name) + ` (${b.frames.map(n => 'T' + n).join(', ')})`).join('; ') : 'none (NASA/JPL SB Ident)']);
    rows.push(['Planets / Moon', (D.planets || []).join(', ') || 'none']);
  }
  let html = '<div class="sbh">Compared with the catalogs</div>' + rows.map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('') +
    (D.notes || []).map(n => `<div class="dim">${esc(n)}</div>`).join('');
  if (D.sky_map) {
    const nw = D.new_objects || [];
    html += `<div class="sbh">NEW objects · ${nw.length}</div>` + (nw.length ? '<table><tr><th>Object</th><th>Class</th><th>First seen (UTC)</th><th>What it looks like</th></tr>' +
      nw.map(o => `<tr class="pick" data-id="${esc(o.object)}"><td class="new">${esc(o.id)}</td><td>${esc(o.class)}</td><td>${esc(o.first_seen_utc)}</td><td>${esc(o.description || '')}</td></tr>`).join('') + '</table>' +
      '<div class="dim" style="margin-top:4px">Click one to show it in the images and track it.</div>'
      : '<div class="dim">None in these images: every source SPHEREx detects here is a known star, asteroid, comet or planet.</div>') +
      (D.new_outside ? `<div class="dim" style="margin-top:4px">${D.new_outside} more just outside this patch: make the patch larger ("Go to sky coordinates" → Advanced) to see ${D.new_outside === 1 ? 'it' : 'them'}.</div>` : '');
  }
  $('tlDb').innerHTML = html;
  for (const tr of $('tlDb').querySelectorAll('tr.pick')) tr.onclick = () => {           // show it where it was first seen
    const O = S.objects[tr.dataset.id]; if (!O) return; tlPlay(false); tlFrame(O.frames[0] - 1); tlSelect(tr.dataset.id); };
}
const ICON_PLAY = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>', ICON_PAUSE = '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
function tlPlay(on) {
  tl.playing = on; clearInterval(tl.timer); tl.timer = on && tl.frames.length > 1 ? setInterval(() => tlFrame(tl.i + 1), 1000/tl.fps) : null;
  $('tlPlay').innerHTML = on ? ICON_PAUSE : ICON_PLAY; $('tlPlay').setAttribute('aria-label', on ? 'Pause' : 'Play');
}
function tlKeys(e) {                                            // the player's keys while the popup is open (from app.js)
  if ($('tlPlayer').hidden || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || (e.target.tagName === 'BUTTON' && e.key === ' ')) return false;
  const n = tl.frames.length, go = i => { tlPlay(false); tlFrame(i); };
  if (e.key === 'ArrowRight') go(tl.i + 1); else if (e.key === 'ArrowLeft') go(tl.i - 1);
  else if (e.key === 'Home') go(0); else if (e.key === 'End') go(n - 1);
  else if (e.key === ' ') tlPlay(!tl.playing); else return false;
  e.preventDefault(); return true;
}
function tlInit() {                                             // called once from app.js
  $('tlPlay').onclick = () => tlPlay(!tl.playing);
  $('tlFirst').onclick = () => { tlPlay(false); tlFrame(0); }; $('tlLast').onclick = () => { tlPlay(false); tlFrame(tl.frames.length - 1); };
  $('tlPrev').onclick = () => { tlPlay(false); tlFrame(tl.i - 1); }; $('tlNext').onclick = () => { tlPlay(false); tlFrame(tl.i + 1); };
  $('tlSpeed').onchange = () => { tl.fps = 2*+$('tlSpeed').value; if (tl.playing) tlPlay(true); };
  $('tlTipNext').onclick = () => { tlTipShow(tl.tip + 1); tlTips(true); };
  $('anaMin').onclick = $('tlMinBtn').onclick = minimizeAnalysis;
  $('anaMiniOpen').onclick = restoreAnalysis; $('anaMiniX').onclick = closeAnalysis;
  $('tlSlide').oninput = () => { tlPlay(false); tlFrame(+$('tlSlide').value); };
  $('tlCount').onchange = $('tlRetry').onclick = () => { tlTimeHint(); requestAnalysis(tl.payload || anaPayloadNow); };
  tlTimeHint();
  $('tlSvg').onclick = tlClick;
  $('tlMapOn').onchange = () => { $('tlMapKey').hidden = !$('tlMapOn').checked; tlDraw(); };
}
let anaPayloadNow = null;
// When will SPHEREx be able to see direction d? Behind Earth: step SPHEREx around its orbit minute by minute.
// Too close to the Sun: step the Sun along Earth's orbit day by day (JPL mean elements) until it is 91 deg away.
function spxOffAt(jd) { const a = snap.a_km/AU_KM, phi = 2*Math.PI*(jd - snap.jd)*1440/snap.period_min; return [0, 1, 2].map(k => a*(Math.cos(phi)*snap.u[k] + Math.sin(phi)*snap.w[k])); }
function nextVisible(d) {
  const sunOK = jd => { const E = kepler('Earth', (jd - J2000)/36525); return dot(d, norm([-E[0], -E[1], -E[2]])) <= COS_SUN; };
  if (sunOK(sim.jd)) {
    for (let m = 1; m <= 200; m++) { const jd = sim.jd + m/1440, o = spxOffAt(jd), r = Math.hypot(...o), toE = [-o[0]/r, -o[1]/r, -o[2]/r];
      if (dot(d, toE) < Math.cos(Math.asin(Math.min(1, EARTH_R_KM/(r*AU_KM))))) return `It comes back into view in about ${m} minute${m === 1 ? '' : 's'}, as SPHEREx moves around Earth (one orbit takes ${snap.period_min.toFixed(0)} minutes).`; }
    return 'Earth blocks it for the whole of this orbit.';
  }
  for (let dd = 1; dd <= 366; dd++) if (sunOK(sim.jd + dd)) return `As Earth moves around the Sun, it comes into SPHEREx's view around ${fmtDate(sim.jd + dd, false).slice(0, 10)} (in ${dd} day${dd === 1 ? '' : 's'}).`;
  return 'It stays too close to the Sun for the next year.';
}
let clickMark = null;
function drawClickMark() {                                     // brief feedback where you double-clicked: green ring = SPHEREx can see it, red = it can't
  if (!clickMark) return; const age = (performance.now() - clickMark.t)/1000; if (age > 3) { clickMark = null; return; }
  const q = skyXY(toView(clickMark.dir), skyScale()); if (!q) return;
  ctx.globalAlpha = Math.min(1, 3 - age); ctx.lineWidth = 2; ctx.strokeStyle = clickMark.ok ? '#5fe0a0' : '#ff6b5a';
  ctx.beginPath(); ctx.arc(q[0], q[1], 12 + age*4, 0, 7); ctx.stroke();
  if (!clickMark.ok) { ctx.beginPath(); ctx.moveTo(q[0] - 7, q[1] - 7); ctx.lineTo(q[0] + 7, q[1] + 7); ctx.moveTo(q[0] + 7, q[1] - 7); ctx.lineTo(q[0] - 7, q[1] + 7); ctx.stroke(); }
  ctx.globalAlpha = 1;
}
function openAnalysis(x, y) {
  if (!state || !sky) return;
  const A = analysedAt(x, y);                                  // double-click on an analysed patch: open that analysis again
  if (A) openSaved(A); else openAnalysisDir(skyDirAt(x, y), null, patchHW());
}
// ---- where SPHEREx images have been analysed: every finished analysis projected on the sky (layer "Analysed patches")
let analysed = [], analysedSpots = [];
async function loadAnalysed() {
  try { analysed = (await api('/api/analyses')).analyses || []; } catch (e) { analysed = []; }
  $('cAnalysedN').textContent = analysed.length ? `(${analysed.length})` : '';
}
const eqDir = (ra, dec) => eq2ecl(ra*D2R, dec*D2R);
function drawAnalysed() {
  analysedSpots = [];
  if (!opt.cAnalysed || !analysed.length) return;
  const SK = skyScale(), col = '#5fe0a0', font = '11px system-ui', rects = [];
  ctx.save(); ctx.lineWidth = 1.4; ctx.font = font;
  for (const A of analysed) {
    const hw = A.patch_px*A.pixel_arcsec/3600/2, at = (ra, dec) => skyXY(toView(eqDir(ra, dec)), SK);
    let q = at(A.ra_deg, A.dec_deg), half = 6;
    if (A.moving && A.path && A.path.length > 1) {            // a moving object: the path its images followed
      ctx.strokeStyle = col; ctx.setLineDash([5, 4]); ctx.beginPath(); let on = false;
      for (const [ra, dec] of A.path) { const p = at(ra, dec); if (!p) { on = false; continue; } on ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); on = true; }
      ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = col;
      for (const [ra, dec] of A.path) { const p = at(ra, dec); if (p) { ctx.beginPath(); ctx.arc(p[0], p[1], 2.6, 0, 7); ctx.fill(); } }
    } else if (q) {                                            // a patch: its real size on the sky (at least a few pixels)
      half = Math.max(4, hw*D2R*SK);
      ctx.fillStyle = col + '26'; ctx.fillRect(q[0] - half, q[1] - half, 2*half, 2*half);
      ctx.strokeStyle = col; ctx.strokeRect(q[0] - half, q[1] - half, 2*half, 2*half);
    }
    if (!q || q[0] < -50 || q[0] > W + 50 || q[1] < -50 || q[1] > H + 50) continue;
    const label = `${A.tracking || A.name} · ${A.n_frames} image${A.n_frames === 1 ? '' : 's'}`, w = measure(font, label);
    for (const [lx, ly] of [[q[0] + half + 5, q[1] - half + 9], [q[0] + half + 5, q[1] + half + 12], [q[0] - half - 5 - w, q[1] - half + 9]]) {
      const r = [lx, ly - 11, lx + w, ly + 3]; if (hitsRect(r, rects)) continue;
      rects.push(r); ctx.fillStyle = '#02040acc'; ctx.fillRect(lx - 3, ly - 12, w + 6, 16); ctx.fillStyle = col; ctx.fillText(label, lx, ly); break;
    }
    analysedSpots.push({A, x: q[0], y: q[1], r: half + 6});
    const span = A.first_utc ? (A.first_utc.slice(0, 10) + (A.last_utc && A.last_utc.slice(0, 10) !== A.first_utc.slice(0, 10) ? ' → ' + A.last_utc.slice(0, 10) : '')) : '—';
    addHit(q[0], q[1], half + 6, 0.65, () => tipHTML(esc(A.tracking || A.name), A.tracking ? 'tracked with XARVIS' : 'analysed with XARVIS',
      [['SPHEREx images', `${A.n_frames}` + (A.images_available ? ` of ${A.images_available}` : '')], ['Imaged', span],
       ['Patch', `${A.patch_px} × ${A.patch_px} px (${(A.patch_px*A.pixel_arcsec/60).toFixed(1)}′)`], ...(A.moving ? [['Motion', 'moving: the line is the path its images followed']] : []),
       ['NEW objects in the images', String(A.new_objects)]],
      'Double-click it to open this analysis again (the images show when SPHEREx can see it on the date shown).'));
  }
  ctx.restore();
}
function analysedAt(x, y) {                                    // the analysed patch under the cursor, nearest first
  if (!opt.cAnalysed) return null; let best = null, bd = 1e9;
  for (const s of analysedSpots) { const d = Math.max(Math.abs(x - s.x), Math.abs(y - s.y)); if (d <= s.r && d < bd) { bd = d; best = s.A; } }
  return best;
}
function openSaved(A) {                                       // the popup for a finished analysis, asking for exactly its request (so it is reused, not run again)
  const R = A.request, count = R.max_images == null ? 'all' : String(R.max_images);
  if ([...$('tlCount').options].some(o => o.value === count)) { $('tlCount').value = count; tlTimeHint(); }
  const S = R.star, known = S && targets.find(t => String(t.gaia_dr3) === String(S.gaia_dr3));   // a test target: its own entry
  const T = known || {name: S ? S.name : `RA ${fmtRA(R.ra_deg, false)} · Dec ${fmtDec(R.dec_deg, false)}`, ra: R.ra_deg, dec: R.dec_deg, typed: true, source: 'saved XARVIS analysis'};
  openAnalysisDir(eqDir(R.ra_deg, R.dec_deg), T, R.patch_px*SPHEREX_PIX/3600/2, A);
}
function savedPayload(P, A) {
  const R = A.request, S = R.star;
  const base = {...P, target: {...P.target, ra_deg: R.ra_deg, dec_deg: R.dec_deg},
    named_target: S ? {name: S.name, source: 'ESA Gaia DR3', gaia_dr3: S.gaia_dr3, ra_deg_at_epoch: S.ra, dec_deg_at_epoch: S.dec, epoch: S.epoch, pmra_mas_yr: S.pmra, pmdec_mas_yr: S.pmdec, parallax_mas: S.parallax} : null};
  if (!R.track) return base;
  tl.base = base; return {...base, track: R.track};             // a tracked object: "Back to the patch" goes to the patch it was picked in
}
function openAnalysisDir(dir, T, hw, saved = null) {
  if (!state || !sky) return;
  const P = anaPayloadNow = analysisPayload(dir, T, hw), t = P.target;
  setText('anaTitle', T ? T.name : `RA ${fmtRA(t.ra_deg, false)} · Dec ${fmtDec(t.dec_deg, false)}`);
  const sg = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3) + '°';
  const rows = [['Time (UTC)', P.time.utc + (P.time.simulated ? ' · simulated' : '')], ['Right ascension / declination', fmtRA(t.ra_deg) + ' / ' + fmtDec(t.dec_deg)],
                ['', t.ra_deg.toFixed(5) + '° / ' + sg(t.dec_deg)],
                ['Ecliptic longitude / latitude', t.ecliptic_lon_deg.toFixed(3) + '° / ' + sg(t.ecliptic_lat_deg)], ['Galactic longitude / latitude', t.galactic_l_deg.toFixed(3) + '° / ' + sg(t.galactic_b_deg)],
                ['Angle from the Sun', t.sun_elongation_deg.toFixed(1) + '°'], ['Can SPHEREx see it now?', t.spherex_visibility.visible_now ? 'yes' : 'no · ' + t.spherex_visibility.reason], ['Patch', P.patch.size.replace(' x ', '′ × ').replace(' arcmin', '′')]];
  $('anaRows').innerHTML = rows.map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
  // the named target: who measured it and how far it moves during the SPHEREx survey
  if (T && T.pmra != null) {
    const pm = Math.hypot(T.pmra, T.pmdec)/1000, yrs = (SCIENCE_END - SCIENCE_START)/365.25, mv = pm*yrs;
    $('anaTarget').innerHTML = [['Catalogue', `ESA Gaia DR3 ${esc(T.gaia_dr3)} · G = ${T.g.toFixed(2)}`], ['Note', esc(T.note || '')], ['Proper motion', pm.toFixed(3) + '″ per year'],
      ['Distance', T.parallax > 0 ? (1000/T.parallax).toFixed(T.parallax > 100 ? 2 : 0) + ' pc · ' + (3261.56/T.parallax).toFixed(T.parallax > 100 ? 2 : 0) + ' light-years' : '—'],
      ['Moves during the SPHEREx survey', `${mv.toFixed(1)}″ ≈ ${(mv/SPHEREX_PIX).toFixed(1)} SPHEREx pixels (6.15″)`]].map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
  } else if (T) $('anaTarget').innerHTML = `<div class="v"><span>Source</span><span>${esc(T.source || '')}</span></div>` + (T.note ? `<div class="v"><span>Note</span><span>${esc(T.note)}</span></div>` : '') + (T.typed ? '' : '<div class="v"><span>Proper motion</span><span>not in Gaia: position not moved</span></div>');
  else $('anaTarget').innerHTML = '';
  $('anaJSON').textContent = JSON.stringify(P, null, 2);
  $('anaCopy').textContent = 'Copy coordinates';
  // the most important answer first: can SPHEREx see it, and if not, when
  const vis = t.spherex_visibility, nx = vis.visible_now ? null : nextVisible(dir);
  $('anaVis').className = vis.visible_now ? 'ok' : 'no';
  $('anaVis').innerHTML = vis.visible_now ? `<b>✓ SPHEREx can see this point right now</b><span>It is ${t.sun_elongation_deg.toFixed(0)}° from the Sun (the limit is 91°) and not behind Earth.</span>`
    : `<b>✕ Outside SPHEREx's view right now</b><span>${vis.behind_earth && vis.sun_limit_ok ? 'Earth is in the way at this moment.' : `It is only ${t.sun_elongation_deg.toFixed(0)}° from the Sun; SPHEREx never points closer than 91°.`} ${nx || ''}</span>`;
  const inSpan = inDataSpan(sim.jd);
  $('anaViz').hidden = !vis.visible_now || !inSpan;          // no image box outside SPHEREx's view or outside the dates it has data for
  $('ana').classList.toggle('noviz', $('anaViz').hidden);      // then the data alone, in one column
  $('anaMini').hidden = true;                                  // (a new point replaces a minimized analysis)
  if (!inSpan) $('anaVis').innerHTML += `<span class="span">No SPHEREx images exist for this date: its data run from ${utc(DATA_FIRST)} to ${utc(dataLatest())}. Use “Jump to…” in the time bar to go there.</span>`;
  else if (sim.jd > nowJD() - 60) $('anaVis').innerHTML += '<span class="span">Images from the last 60 days may not be released by IRSA yet.</span>';
  P.target.spherex_next_visible = nx;
  $('anaJSON').textContent = JSON.stringify(P, null, 2);
  $('anaBack').hidden = false; $('ana').scrollTop = 0; $('anaPayload').open = false; $('anaClose').focus({preventScroll: true});
  clickMark = {dir, ok: vis.visible_now, t: performance.now()};
  requestAnalysis(saved && !$('anaViz').hidden ? savedPayload(P, saved) : P);   // the SPHEREx image time-lapse (only when the image box is shown)
}
// SPHEREx science data: from 1 May 2025 (start of the survey) to now, until the mission ends; IRSA releases each week's
// images within 60 days of observation (IPAC SPHEREx Explanatory Supplement, sec. 2.1-2.2).
const utc = jd => fmtDate(jd, false) + ' UTC';                     // 'YYYY-MM-DD hh:mm UTC'
const DATA_FIRST = 2460796.5, dataLatest = () => Math.min(nowJD(), 2461557.5), inDataSpan = jd => jd >= DATA_FIRST && jd <= dataLatest();
const SCIENCE_START = 2460796.5, SCIENCE_END = 2461557.5;
function closeAnalysis() { $('anaBack').hidden = true; $('anaMini').hidden = true; tl.req++; tlReset(); tl.payload = tl.base = tl.sel = null; $('tlTrack').hidden = true; }   // stops the polling and the player; the server keeps finished results
