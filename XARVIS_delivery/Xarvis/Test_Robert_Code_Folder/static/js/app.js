// XARVIS viewer · app.js
// Main loop (one frame per screen refresh), the status panel and advanced orbital parameters, all UI event wiring,
// and boot. Loaded last: it starts everything.
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// ================================================================================ main loop ===
let last = performance.now(), ema = 16.7, slowFrames = 0, lastAdapt = 0, lastUI = 0, lastCoord = 0;
const perf = {n: 0, ms: 0, max: 0}, noAdapt = location.search.includes('noadapt'), showPerf = location.search.includes('perf');
if (showPerf) $('perf').style.display = 'block';
let spin = false, spinTilt = 0, spinEcl = false;
function frame(t) {
  const dtRaw = t - last, dtm = Math.min(0.1, dtRaw/1000); last = t; const t0 = performance.now();
  advanceTime(dtm); ensureData();
  if (!anim) {
    const d = ztGoal - zt; if (Math.abs(d) > 1e-5) { zt += d*(1 - Math.exp(-dtm*30)); if (Math.abs(zt - +$('zoom').value) > 2e-4) $('zoom').value = zt; }
  }
  ctx.clearRect(0, 0, W, H); hits = [];
  if (viewAnim && viewAnim(t)) viewAnim = null;
  if (spin && !dragging) {
    const a = 0.25*dtm;
    if (spinEcl) { const c = Math.cos(a), sn = Math.sin(a); R = R.map(r => [c*r[0] + sn*r[1], -sn*r[0] + c*r[1], r[2]]); orthonormalize(); }   // about the ecliptic north pole
    else { const t = spinTilt*D2R, n = [Math.sin(t), Math.cos(t), 0], k = a, c = Math.cos(k), sn = Math.sin(k), v = 1 - c;
      applyView([[c + n[0]*n[0]*v, n[0]*n[1]*v, n[1]*sn], [n[0]*n[1]*v, c + n[1]*n[1]*v, -n[0]*sn], [-n[1]*sn, n[0]*sn, c]]); }   // Rodrigues rotation about the chosen screen-plane axis
  }
  const ready = sky && updateScene();
  if (ready) {
    updateStars();
    draw3D(); pickHit();
    if (t - lastCoord > 50) { lastCoord = t; updateCoords(); }
    if (t - lastUI > 120) { lastUI = t; updateUI(); if (t - trackListT > 2000) refreshTrackList(); }
    const ld = $('loading'); if (ld) { ld.remove(); $('fade').classList.add('gone'); setTimeout(() => { const f = $('fade'); if (f) f.remove(); }, 1600); }
  }
  const ms = performance.now() - t0; perf.n++; perf.ms += ms; perf.max = Math.max(perf.max, ms);
  if (dtRaw < 250) ema += (dtRaw - ema)*0.08;
  slowFrames = ema > 34 ? slowFrames + 1 : 0;
  if (!noAdapt && slowFrames > 60 && K > 0.61 && t - lastAdapt > 1200) { resScale *= 0.8; applyRes(); lastAdapt = t; slowFrames = 0; }
  if (showPerf && perf.n % 20 === 0) $('perf').textContent = `${(1000/ema).toFixed(0)} fps · draw ${(perf.ms/perf.n).toFixed(1)} ms (max ${perf.max.toFixed(0)}) · canvas x${K.toFixed(2)}`;
  if (perf.n % 120 === 0) { perf.ms = 0; perf.max = 0; perf.n = 1; }
  requestAnimationFrame(frame);
}

// ======================================================================================== UI ===
function updateUI() {
  const jd = sim.jd, rate = simRate();
  setText('clock', fmtDate(jd) + ' UTC');
  const dyr = (jd - nowJD())/365.25, dd = jd - nowJD();
  const badge = $('badge'); badge.textContent = sim.live ? 'LIVE' : (rate === 0 ? 'PAUSED' : 'SIMULATED'); badge.className = 'badge ' + (sim.live ? 'live' : 'sim');
  let note = '';
  if (!sim.live) note = Math.abs(dd) < 1.5 ? (dd*24).toFixed(1) + ' h from now' : Math.abs(dyr) < 2 ? dd.toFixed(1) + ' days from now' : (dyr >= 0 ? '+' : '−') + Math.abs(dyr).toLocaleString(undefined, {maximumFractionDigits: Math.abs(dyr) < 100 ? 1 : 0}) + ' years from now';
  setText('clockNote', note);
  const sp = state.spherex;
  setHTML('status', `NASA's infrared sky-mapping telescope is <b>${Math.round(sp.altitude_km).toLocaleString()} km</b> above Earth, circling it every <b>${sp.period_min.toFixed(0)} minutes</b>.`);
  setHTML('warns', state.warns.length ? '<div class="wh">Note</div>' + state.warns.map(w => `<div>${w}</div>`).join('') : '');
  setText('advWarn', state.warns.length ? '· ' + state.warns.length + (state.warns.length > 1 ? ' notes' : ' note') : '');
  if (document.activeElement !== $('dateIn')) $('dateIn').value = fmtDate(jd, false);
  setText('tSpan', 'SPHEREx data ' + fmtDate(DATA_FIRST, false) + ' → ' + utc(dataLatest()) + ' ·');
  if (!window._jl || performance.now() - window._jl > 60000) { window._jl = performance.now(); $('jumpLatest').textContent = 'Latest data · ' + utc(dataLatest()); }
  setText('tHint', sim.live ? 'live' : rate === 0 ? 'paused' : (rate < 0 ? 'rewinding · ' : 'playing · ') + RATES[sim.rateIdx][0]);
  for (const [id, b] of [['rev', !sim.live && sim.dir < 0], ['fwd', !sim.live && sim.dir > 0], ['pause', !sim.live && sim.dir === 0]]) $(id).classList.toggle('on', b);
  $('live').classList.toggle('on', sim.live);
  if (!$('advScroll').hidden) setHTML('adv', advancedHTML(sp));
}
// ---- advanced orbital parameters (for the curious): the orbit as an astronomer would list it
function advancedHTML(sp) {
  const mod = (x, m) => ((x % m) + m) % m, deg = x => x.toFixed(3) + '°', f = (x, d) => x.toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
  const nE = ecl2eqVec(snap.n), pE = ecl2eqVec(norm(sp.off)), sunE = ecl2eqVec(state.sun_dir);        // equatorial J2000 (the frame orbit elements are quoted in)
  const inc = Math.acos(clamp(nE[2], -1, 1))/D2R, node = norm([-nE[1], nE[0], 0]);
  const raan = mod(Math.atan2(node[1], node[0])/D2R, 360), argLat = mod(Math.atan2(dot(cross(node, pE), nE), dot(node, pE))/D2R, 360);
  const beta = Math.asin(clamp(dot(nE, sunE), -1, 1))/D2R, sunRA = mod(Math.atan2(sunE[1], sunE[0])/D2R, 360), ltan = mod(12 + (raan - sunRA)/15, 24);
  const aKm = snap.a_km, v = 2*Math.PI*aKm/(snap.period_min*60), E = state.E, y = yearsSince2000(sim.jd), vis = skyFractions();
  const hm = h => { const H = Math.floor(h), M = Math.round((h - H)*60); return String(M === 60 ? H + 1 : H).padStart(2, '0') + ':' + String(M === 60 ? 0 : M).padStart(2, '0'); };
  const grp = (t, rows) => `<div class="grp">${t}</div>` + rows.map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
  return grp('Orbit around Earth · J2000 equatorial', [
      ['Semi-major axis  a', f(aKm, 1) + ' km'], ['Eccentricity  e', '≈ 0 (circular model)'], ['Inclination  i', deg(inc)], ['RA of ascending node  Ω', deg(raan)],
      ['Argument of latitude  u', deg(argLat)], ['Period', f(snap.period_min, 2) + ' min'], ['Mean motion', f(1440/snap.period_min, 3) + ' rev/day'], ['Orbital speed', f(v, 3) + ' km/s'], ['Altitude', f(sp.altitude_km, 1) + ' km']])
    + grp('Sun-synchronous geometry', [['Beta angle  β', deg(beta)], ['Ascending node, local time', hm(ltan)]])
    + grp('What SPHEREx can see', [
      ['One exposure (field of view)', FOV_DEG[0] + '° × ' + FOV_DEG[1] + '°'], ['One exposure (solid angle)', f(FOV_SR*SR2DEG2, 1) + ' deg² · ' + f(FOV_SR*1000, 1) + ' msr'],
      ['Visible now', f(vis.now*100, 1) + '% of the sky · ' + f(vis.now*4*Math.PI, 2) + ' sr'],
      ['Hidden by Earth now', f((vis.sun - vis.now)*100, 1) + '% · disc radius ' + vis.rhoDeg.toFixed(1) + '°'],
      ['Visible overall (Sun limit)', f(vis.sun*100, 1) + '% · ' + f(vis.sun*4*Math.PI, 2) + ' sr'],
      ['Over six months', 'the whole sky']])
    + grp('Position · J2000 ecliptic', [
      ['Distance from Sun', f(sp.helio_dist_au, 5) + ' AU'], ['Distance from Earth centre', f(sp.earth_dist_km, 0) + ' km'], ['Longitude / latitude', sp.ecl_lon_deg.toFixed(3) + '° / ' + sp.ecl_lat_deg.toFixed(3) + '°'],
      ['X (heliocentric)', f(sp.pos[0], 5) + ' AU'], ['Y', f(sp.pos[1], 5) + ' AU'], ['Z', f(sp.pos[2], 5) + ' AU'], ['Earth–Sun distance', f(Math.hypot(...E), 5) + ' AU']])
    + grp('Time and data', [['Julian date (UTC)', f(sim.jd, 5)], ['Planet positions', state.from === 'track' ? (meta ? meta.ephemeris : '…') : 'JPL mean orbital elements'], ['SPHEREx orbit', sp.source], ['Star positions', 'epoch J2000 ' + (y >= 0 ? '+' : '−') + Math.abs(y).toLocaleString(undefined, {maximumFractionDigits: 1}) + ' yr']])
    + '<div class="dim" style="font-size:11px;margin-top:6px;line-height:1.35">The design node time of SPHEREx is 06:00 (IPAC). Visibility uses the official 91° Sun limit and the real disc of Earth; SPHEREx also avoids the Moon and its flight direction, but those angles are not published, so they are not included. The eccentricity is not measured here: the orbit is drawn as a circle (the real one is within about 0.001 of circular). Orbit shape and phase come from the latest NASA/JPL Horizons state; the elements above are derived from it.</div>';
}
function animateTo(target, ms) {
  const from = zt, t0 = performance.now();
  anim = t => { const k = Math.min(1, (t - t0)/ms), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2)/2; zt = ztGoal = from + (target - from)*e; $('zoom').value = zt; return k >= 1; };
  (function step(t) { if (anim && !anim(t)) requestAnimationFrame(step); else anim = null; })(performance.now());
}
function zoomBy(delta) { anim = null; ztGoal = clamp(ztGoal + delta*0.9, 0, 1); }
function goDate() { const jd = parseDate($('dateIn').value); if (jd === null || jd < DATA_FIRST - 1 || jd > dataLatest() + 1) { $('dateIn').style.borderColor = '#ff6b6b'; $('dateIn').title = 'Pick a UTC date between ' + utc(DATA_FIRST) + ' and ' + utc(dataLatest()) + ' (the SPHEREx data)'; return; } $('dateIn').style.borderColor = ''; const wasRealTime = sim.live || (sim.dir === 1 && sim.rateIdx === 0); setJD(jd); if (wasRealTime) sim.dir = 0; $('dateIn').blur(); }   // a jump from real time pauses on the chosen instant
function initUI() {
  $('speed').innerHTML = RATES.map((r, i) => `<option value="${i}">${r[0]}</option>`).join('');
  $('speed').onchange = () => { sim.rateIdx = +$('speed').value; if (sim.rateIdx !== 0 || sim.dir !== 1) sim.live = false; if (sim.dir === 0) sim.dir = 1; };
  $('fwd').onclick = () => { sim.dir = 1; if (sim.live && sim.rateIdx === 0) return; sim.live = false; };
  $('rev').onclick = () => { sim.dir = -1; sim.live = false; };
  $('pause').onclick = () => { if (sim.live) { sim.live = false; sim.dir = 0; } else sim.dir = sim.dir === 0 ? 1 : 0; };
  $('live').onclick = () => { sim.live = true; sim.dir = 1; sim.rateIdx = 0; $('speed').value = 0; sim.jd = nowJD(); };
  $('go').onclick = goDate;
  $('jump').onchange = () => { const v = $('jump').value; $('jump').value = ''; if (!v) return;
    setJD(v === 'first' ? DATA_FIRST : dataLatest()); sim.dir = 0; $('jump').blur(); };   // jump there and pause
  $('dateIn').addEventListener('keydown', e => { if (e.key === 'Enter') goDate(); e.stopPropagation(); });
  $('reset').onclick = () => { R = matFromYawPitch(-0.5, 1.0); focusId = null; renderChips(); };
  $('spin').onclick = () => { spin = !spin; $('spin').classList.toggle('on', spin); $('spinPanel').hidden = !spin; };
  const setTilt = (v, ecl = false) => { spinTilt = v; spinEcl = ecl; $('spinAxis').value = v; setText('spinVal', ecl ? 'ecliptic pole' : v + '°'); $('spinE').classList.toggle('on', ecl); };
  $('spinAxis').oninput = () => setTilt(+$('spinAxis').value);
  $('spinV').onclick = () => setTilt(0); $('spinH').onclick = () => setTilt(90); $('spinE').onclick = () => setTilt(0, true);
  $('gotoBtn').onclick = () => { const h = $('gotoPanel').hidden; $('gotoPanel').hidden = !h; $('gotoArrow').textContent = h ? '▴' : '▾'; if (h) $('gRA').focus(); };
  $('gTarget').onchange = () => { const T = targets[+$('gTarget').value]; if (!T) return; const [ra, dec] = [T.ra, T.dec]; $('gRA').value = ra.toFixed(6); $('gDec').value = dec.toFixed(6);
    $('gMsg').className = 'dim'; $('gMsg').textContent = T.pmra != null ? `ESA Gaia DR3 position at J${T.epoch}, moved with its proper motion to the date shown.` : 'Coordinates from the challenge text; not in Gaia, so not moved.'; };
  for (const id of ['gRA', 'gDec']) $(id).addEventListener('keydown', e => { if (e.key === 'Enter') goToTyped(); e.stopPropagation(); });
  $('gGo').onclick = goToTyped; $('gPix').oninput = showPatchInfo; showPatchInfo();
  $('trackBtn').onclick = () => { const h = $('trackPanel').hidden; $('trackPanel').hidden = !h; $('tools').classList.toggle('wide', h); $('trackArrow').textContent = h ? '▴' : '▾'; if (h) { refreshTrackList(); $('tq').focus(); } };
  $('tq').oninput = refreshTrackList; for (const id of ['fShow', 'fOnly']) $(id).onchange = refreshTrackList;
  $('tq').addEventListener('keydown', e => { if (e.key === 'Enter') { const first = $('tList').querySelector('.ti'); if (first && !first.classList.contains('on')) toggleTrack(first.dataset.id); else if (!first && $('tq').value.trim()) jplLookup($('tq').value.trim()); } e.stopPropagation(); });
  $('tList').onclick = e => { const t = e.target.closest('.ti'); if (t) toggleTrack(t.dataset.id); };
  $('tLookup').onclick = () => jplLookup($('tq').value.trim());
  $('tMsg').onclick = e => { const a = e.target.closest('a[data-des]'); if (a) { e.preventDefault(); jplLookup(a.dataset.des); } };
  $('trackChips').onclick = e => { const c = e.target.closest('.chip'); if (!c) return; const id = c.dataset.id;
    if (e.target.closest('.cx')) toggleTrack(id); else if (e.target.closest('.cf')) { focusId = focusId === id ? null : id; renderChips(); } };
  $('layersBtn').onclick = () => { const h = $('layers').hidden; $('layers').hidden = !h; $('layersBtn').textContent = 'Layers ' + (h ? '▴' : '▾'); };
  $('advBtn').onclick = () => { const h = $('advScroll').hidden; $('advScroll').hidden = !h; $('advArrow').textContent = h ? '▴' : '▾'; if (h) updateUI(); };
  $('zoom').oninput = () => { anim = null; ztGoal = +$('zoom').value; };
  $('zSphx').onclick = () => { focusId = null; renderChips(); animateTo(1, 2200); }; $('zSys').onclick = () => animateTo(tFit(), 1600); $('zFar').onclick = () => animateTo(0, 1600);
  cv.addEventListener('wheel', e => { e.preventDefault(); zoomBy(clamp(-e.deltaY*(e.deltaMode === 1 ? 33 : 1), -120, 120)*0.0006); }, {passive: false});   // one mouse-wheel notch = ~5% of the zoom range
  const pointers = new Map(); let prevPinch = null;
  cv.onpointerdown = e => { pointers.set(e.pointerId, [e.clientX, e.clientY]); cv.setPointerCapture(e.pointerId); dragging = true; cv.classList.add('drag'); prevPinch = null; };
  cv.onpointerup = cv.onpointercancel = e => { pointers.delete(e.pointerId); prevPinch = null; if (!pointers.size) { dragging = false; cv.classList.remove('drag'); } };
  cv.onpointermove = e => {
    cursor = [e.clientX, e.clientY]; const p = pointers.get(e.pointerId); if (!p) return;
    if (pointers.size === 1) { const dx = (e.clientX - p[0])*0.005, dy = (e.clientY - p[1])*0.005; if (e.shiftKey || e.buttons & 2) rotateBy(0, 0, dx); else rotateBy(dx, dy, 0); }
    else if (pointers.size === 2) { pointers.set(e.pointerId, [e.clientX, e.clientY]); const [a, b] = [...pointers.values()], d = Math.hypot(a[0]-b[0], a[1]-b[1]), ang = Math.atan2(b[1]-a[1], b[0]-a[0]);
      if (prevPinch) { zoomBy(Math.log(d/prevPinch.d)*0.5); rotateBy(0, 0, ang - prevPinch.ang); } prevPinch = {d, ang}; return; }
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
  };
  cv.ondblclick = e => openAnalysis(e.clientX, e.clientY);
  let introSeen = false; try { introSeen = localStorage.getItem('xarvis.introSeen') === '1'; } catch (err) {}
  $('intro').hidden = introSeen; if (!introSeen) requestAnimationFrame(() => $('intro').classList.add('show'));
  const closeIntro = () => { introSeen = $('introSkip').checked; try { if ($('introSkip').checked) localStorage.setItem('xarvis.introSeen', '1'); else localStorage.removeItem('xarvis.introSeen'); } catch (err) {} $('intro').classList.add('out'); setTimeout(() => { $('intro').hidden = true; $('intro').classList.remove('out'); }, 450); };
  $('introGo').onclick = closeIntro;
  $('intro').onclick = e => { if (e.target === $('intro')) closeIntro(); };
  $('helpLink').onclick = e => { e.preventDefault(); $('introSkip').checked = introSeen; $('intro').hidden = false; $('intro').classList.remove('show'); requestAnimationFrame(() => requestAnimationFrame(() => $('intro').classList.add('show'))); $('introGo').focus(); };
  $('anaX').onclick = $('anaClose').onclick = closeAnalysis; tlInit();
  $('anaBack').onclick = e => { if (e.target === $('anaBack')) closeAnalysis(); };
  $('anaCopy').onclick = async () => { try { await navigator.clipboard.writeText(JSON.stringify(anaPayloadNow, null, 2)); $('anaCopy').textContent = 'Copied'; } catch (err) { $('anaCopy').textContent = 'Copy failed'; } };
  cv.onpointerleave = () => { cursor = null; }; cv.oncontextmenu = e => e.preventDefault();
  addEventListener('keydown', e => { if (!$('intro').hidden) { if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); $('introGo').click(); } return; } if (!$('anaBack').hidden) { if (e.key === 'Escape') closeAnalysis(); else tlKeys(e); return; } if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return; const st = 0.06;
    if (e.key === 'q' || e.key === 'Q') rotateBy(0, 0, -st); else if (e.key === 'e' || e.key === 'E') rotateBy(0, 0, st);
    else if (e.key === 'ArrowLeft') rotateBy(-st, 0, 0); else if (e.key === 'ArrowRight') rotateBy(st, 0, 0); else if (e.key === 'ArrowUp') rotateBy(0, -st, 0); else if (e.key === 'ArrowDown') rotateBy(0, st, 0);
    else if (e.key === ' ') { e.preventDefault(); $('pause').click(); } else if (e.key === 'l' || e.key === 'L') $('live').click(); });
  addEventListener('resize', resize);
}

// ===================================================================================== boot ===
(async function boot() {
  setTimeout(() => { const f = $('fade'); if (f) f.classList.add('gone'); }, 8000);   // never leave the page black if data is slow
  resize(); initUI(); zt = ztGoal = tFit(); $('zoom').value = zt;
  setJD(DATA_FIRST); sim.dir = 0;                              // open on the first day of SPHEREx data (paused, like "Jump to… First data")
  try { meta = await api('/api/meta'); } catch (e) { console.warn('meta', e); meta = {launch_jd: 2460746.6325, science_start_jd: 2460796.5, science_end_jd: 2461557.5, ephemeris: 'unavailable', de_start_jd: null}; }
  loadSky(); loadSmallBodies(); loadTargets(); loadAnalysed();
  requestAnimationFrame(frame);
})();
// Debug hooks: inspect or drive the viewer from the browser console (e.g. XARVIS.setJD(2460796.5)); also used by tests
window.XARVIS = {visGeom: () => visGeom(), visCv, skyDirAt, skyXY, toView, skyScale, get W() { return W; }, get H() { return H; }, tracked: () => [...tracked.values()], catalog: () => catalog, get labelRects() { return labelRects; }, sim, setJD, updateScene, updateStars, kepler, trackEval, scratch, get state() { return state; }, get sky() { return sky; }, get track() { return track; }, get snap() { return snap; }, advanceTime, RATES, nowJD, fmtDate, parseDate, ecl2gal, ecl2eq, skyDirAt, get R() { return R; }, set R(v) { R = v; }};
