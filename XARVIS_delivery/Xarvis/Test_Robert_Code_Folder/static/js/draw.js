// XARVIS viewer · draw.js
// Drawing helpers: canvas resolution, the 3D projection and the stereographic star backdrop, viewport clipping,
// hover tooltips, stars and constellations with non-overlapping labels, and the real asteroids / Trojans / Kuiper belt.
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// ============================================================================== resolution ===
function applyRes() { W = innerWidth; H = innerHeight; K = Math.max(0.6, DPR*resScale); cv.width = Math.round(W*K); cv.height = Math.round(H*K); mainCtx.setTransform(K,0,0,K,0,0); }
function resize() { DPR = Math.min(window.devicePixelRatio || 1, 2); applyRes(); }

// ================================================================================ projection ===
const sFit = () => Math.min(W, H)/2/36, sFar = () => sFit()/40, sMax = () => 2.2e6*Math.min(W, H)/800;
const scaleOf = t => sFar()*Math.pow(sMax()/sFar(), t);
const tFit = () => Math.log(sFit()/sFar())/Math.log(sMax()/sFar());
let zt = 0, ztGoal = 0, anim = null;
function centerOf(s) { if (!state) return [0,0,0]; if (focusId && tracked.has(focusId)) { const f = tracked.get(focusId).pos(); if (f) return f; } const p = state.spherex.pos, k = Math.max(0, 1 - sFit()/s); return [p[0]*k, p[1]*k, p[2]*k]; }
// Constellations fade in only once you are well outside the planets, and gradually: the fade is spread evenly over the
// zoom slider (logarithmic in scale) from 1/3 of the planets view all the way out to the "Far out" view.
const FADE_START = 0.33, FADE_FULL = 0.025;               // 0.025 x sFit = the "Far out" view
const constellationFade = s => smooth(0, 1, Math.log(FADE_START*sFit()/s)/Math.log(FADE_START/FADE_FULL));
function project(p, c, s) { const v = toView([p[0]-c[0], p[1]-c[1], p[2]-c[2]]); return [W/2 + v[0]*s, H/2 - v[1]*s, v[2]]; }
const skyScale = () => 0.7*Math.max(W, H);
function skyXY(v, SK, clampIt) { let a = -v[2]; if (a < -0.9) { if (!clampIt) return null; a = -0.97; } const k = SK/(1 + a); return [W/2 + v[0]*k, H/2 - v[1]*k]; }
const distStr = au => au < 0.02 ? Math.round(au*AU_KM).toLocaleString() + ' km' : au.toFixed(3) + ' AU';

// ---- clipping: huge off-screen paths are what made zooming stutter, so every segment is clipped to the viewport
const MARGIN = 40;
function clipSeg(x0, y0, x1, y1) {
  const xmin = -MARGIN, xmax = W + MARGIN, ymin = -MARGIN, ymax = H + MARGIN, dx = x1 - x0, dy = y1 - y0; let t0 = 0, t1 = 1;
  const p = [-dx, dx, -dy, dy], q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0];
  for (let i = 0; i < 4; i++) { if (p[i] === 0) { if (q[i] < 0) return null; } else { const r = q[i]/p[i];
    if (p[i] < 0) { if (r > t1) return null; if (r > t0) t0 = r; } else { if (r < t0) return null; if (r < t1) t1 = r; } } }
  return [x0 + t0*dx, y0 + t0*dy, x0 + t1*dx, y0 + t1*dy];
}
const onScr = (x, y) => x >= -MARGIN && x <= W + MARGIN && y >= -MARGIN && y <= H + MARGIN;
function makePen() { let has = false, lx = 0, ly = 0;
  return { gap() { has = false; }, line(x0, y0, x1, y1) {
    const seg = onScr(x0, y0) && onScr(x1, y1) ? [x0, y0, x1, y1] : clipSeg(x0, y0, x1, y1);
    if (!seg) { has = false; return; }
    if (!has || seg[0] !== lx || seg[1] !== ly) ctx.moveTo(seg[0], seg[1]);
    ctx.lineTo(seg[2], seg[3]); lx = seg[2]; ly = seg[3]; has = true; } }; }
// closed path of heliocentric points (optionally offset by `base`), inline projection, clipped
function polyline(pts, c, s, stride = 1, base) {
  const R0 = R[0], R1 = R[1], hw = W/2, hh = H/2, n = pts.length, pen = makePen(), bx = base ? base[0] : 0, by = base ? base[1] : 0, bz = base ? base[2] : 0;
  ctx.beginPath(); let px = 0, py = 0, first = true;
  const at = i => { const p = pts[i], dx = p[0]+bx-c[0], dy = p[1]+by-c[1], dz = p[2]+bz-c[2]; return [hw + (R0[0]*dx + R0[1]*dy + R0[2]*dz)*s, hh - (R1[0]*dx + R1[1]*dy + R1[2]*dz)*s]; };
  for (let i = 0; i < n; i += stride) { const [x, y] = at(i); if (!first) pen.line(px, py, x, y); px = x; py = y; first = false; }
  if ((n - 1) % stride) { const [x, y] = at(n - 1); pen.line(px, py, x, y); }
  ctx.stroke();
}
const maxR = pts => { let m = 0; for (const p of pts) m = Math.max(m, Math.hypot(p[0], p[1], p[2])); return m; };
const radiiCache = new WeakMap();
function orbitRadii(set) { if (!set) return {}; if (!radiiCache.has(set)) { const r = {}; for (const k in set) r[k] = maxR(set[k]); radiiCache.set(set, r); } return radiiCache.get(set); }
function strokeOrbits(set, c, s, suffix) {
  const radii = orbitRadii(set);
  for (const name in set) { const perim = 2*Math.PI*(radii[name] || 1)*s; if (perim < 8) continue;
    ctx.strokeStyle = COLORS[name] + suffix; polyline(set[name], c, s, clamp(Math.floor(3*set[name].length/perim), 1, 16)); }
}
function drawGreatPath(pts, SK) { const pen = makePen(); let px = 0, py = 0, ok = false;
  for (const p of pts) { const q = skyXY(toView(p), SK); if (!q) { ok = false; pen.gap(); continue; } if (ok) pen.line(px, py, q[0], q[1]); px = q[0]; py = q[1]; ok = true; } }
function drawVecPath(vs, SK) { const R0 = R[0], R1 = R[1], R2 = R[2], hw = W/2, hh = H/2, pen = makePen(); let px = 0, py = 0, ok = false;
  for (let i = 0; i < vs.length; i += 3) { const x = vs[i], y = vs[i+1], z = vs[i+2], a = -(R2[0]*x + R2[1]*y + R2[2]*z);
    if (a < -0.9) { ok = false; pen.gap(); continue; }
    const k = SK/(1 + a), qx = hw + (R0[0]*x + R0[1]*y + R0[2]*z)*k, qy = hh - (R1[0]*x + R1[1]*y + R1[2]*z)*k;
    if (ok) pen.line(px, py, qx, qy); px = qx; py = qy; ok = true; } }
function smallCircle(center, radDeg, n = 120) {
  const c = norm(center), p1 = norm(cross(c, Math.abs(c[2]) < 0.9 ? [0,0,1] : [1,0,0])), p2 = cross(c, p1), a = radDeg*D2R, out = [];
  for (let i = 0; i <= n; i++) { const f = i/n*2*Math.PI; out.push([0,1,2].map(k => Math.cos(a)*c[k] + Math.sin(a)*(Math.cos(f)*p1[k] + Math.sin(f)*p2[k]))); }
  return out;
}

// =============================================================================== hover tips ===
let hits = [], cursor = null, dragging = false, tipShown = '';
function addHit(x, y, r, prio, html) { if (x < -20 || x > W+20 || y < -20 || y > H+20) return; hits.push({x, y, r: Math.max(r, 8), prio, html}); }
function tipHTML(title, kind, rows, note) {
  return `<b>${title}</b>${kind ? `<span class="sub">${kind}</span>` : ''}` + (rows.length ? '<table>' + rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</table>' : '') + (note ? `<div class="note">${note}</div>` : '');
}
function bodyRows(pos, viewer, viewerName) {
  const r = Math.hypot(...pos), rows = [['Heliocentric r', r.toFixed(3) + ' AU'], ['Ecl. lon / lat', ((Math.atan2(pos[1], pos[0])/D2R + 360) % 360).toFixed(2) + '° / ' + (Math.asin(pos[2]/r)/D2R).toFixed(2) + '°'], ['X, Y, Z (AU)', pos.map(v => v.toFixed(3)).join(', ')]];
  if (viewer) { const v = sub(pos, viewer), d = Math.hypot(...v); if (d > 1e-12) { rows.push(['From ' + viewerName, distStr(d)]); rows.push(['RA / Dec (J2000)', fmtRADec(norm(v))]); } }
  return rows;
}
function pickHit() {
  const tip = $('tip'); let best = null, bs = 1e9;
  if (cursor && !dragging) for (const h of hits) { const d = Math.hypot(h.x - cursor[0], h.y - cursor[1]); if (d <= h.r) { const sc = d/h.r - h.prio; if (sc < bs) { bs = sc; best = h; } } }
  cv.classList.toggle('hit', !!best);
  if (!best) { if (tipShown !== '') { tip.style.display = 'none'; tipShown = ''; } return; }
  const html = typeof best.html === 'function' ? best.html() : best.html;
  if (html !== tipShown) { tip.innerHTML = html; tipShown = html; tip.style.display = 'block'; }
  tip.style.left = Math.min(W - tip.offsetWidth - 8, cursor[0] + 16) + 'px'; tip.style.top = Math.min(H - tip.offsetHeight - 8, cursor[1] + 16) + 'px';
}
const glowCache = {};
function drawGlow(x, y, r, col) {
  let c = glowCache[col];
  if (!c) { c = glowCache[col] = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d'), gr = g.createRadialGradient(32,32,0,32,32,32);
    gr.addColorStop(0, col + 'ff'); gr.addColorStop(.25, col + '88'); gr.addColorStop(1, col + '00'); g.fillStyle = gr; g.fillRect(0,0,64,64); }
  ctx.drawImage(c, x - r, y - r, 2*r, 2*r);
}

// ==================================================================================== sky ===
function paintStars(SK, sf) {
  const v = sky.v, R0 = R[0], R1 = R[1], R2 = R[2], hw = W/2, hh = H/2;
  sky.bins.forEach((idx, b) => {
    ctx.fillStyle = BIN_COL[b]; ctx.beginPath();
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j], x = v[i*3], y = v[i*3+1], z = v[i*3+2], a = -(R2[0]*x + R2[1]*y + R2[2]*z);
      if (a < -0.9) continue;
      const k = SK/(1 + a), px = hw + (R0[0]*x + R0[1]*y + R0[2]*z)*k, py = hh - (R1[0]*x + R1[1]*y + R1[2]*z)*k;
      if (px < -6 || px > W+6 || py < -6 || py > H+6) continue;
      const r = sky.r[i]*sf; if (r < 1) ctx.rect(px - r, py - r, 2*r, 2*r); else { ctx.moveTo(px + r, py); ctx.arc(px, py, r, 0, 6.2832); }
    }
    ctx.fill();
  });
}
function paintLines(SK) {
  if (opt.cLines) { ctx.strokeStyle = '#4a6fb0'; ctx.lineWidth = 1.1; ctx.beginPath(); for (const l of sky.lines) drawVecPath(l, SK); ctx.stroke(); }
  if (opt.cNames) drawNames(SK);
}
// ---- label placement: a name is only drawn if it does not overlap one already placed; otherwise it tries nearby spots,
// and is hidden until there is room (zoom in or rotate and it appears).
const textW = {};
function measure(font, text) { const k = font + '|' + text; if (textW[k] === undefined) { ctx.font = font; textW[k] = ctx.measureText(text).width; } return textW[k]; }
let labelRects = [], baseRects = [];                          // labelRects: rectangles drawn this frame (read by the tests); baseRects: space reserved first
let uiRectsCache = [], uiRectsT = 0;
function uiReserved() {                                       // screen areas covered by the panels
  const t = performance.now();
  if (t - uiRectsT > 300) { uiRectsT = t; uiRectsCache = ['brand', 'tools', 'dock'].map($).filter(e => e.offsetParent !== null).map(e => { const r = e.getBoundingClientRect(); return [r.left - 4, r.top - 4, r.right + 4, r.bottom + 4]; }); }
  return uiRectsCache;
}
const hitsRect = (r, list) => { for (const q of list) if (r[0] < q[2] + 3 && r[2] + 3 > q[0] && r[1] < q[3] + 2 && r[3] + 2 > q[1]) return true; return false; };
function placeLabel(text, font, x, y, tries) {                // tries: [[dx, dy], ...] top-left offsets from (x, y)
  const w = measure(font, text), h = 13;
  for (const [dx, dy] of tries) {
    const lx = x + dx(w), ly = y + dy, r = [lx, ly - h + 2, lx + w, ly + 3];
    if (r[0] < 4 || r[2] > W - 4 || r[1] < 4 || r[3] > H - 4 || hitsRect(r, labelRects)) continue;
    labelRects.push(r); ctx.font = font; ctx.fillText(text, lx, ly); return true;
  }
  return false;
}
function drawNames(SK) {
  labelRects = baseRects.slice();
  const cf = 'italic 11px system-ui', sf = '11px system-ui', pos = [];
  for (const l of sky.labels) { const q = skyXY(toView(l.v), SK); if (q && q[0] > 0 && q[0] < W && q[1] > 0 && q[1] < H) pos.push([l, q, Math.hypot(q[0] - W/2, q[1] - H/2)]); }
  pos.sort((a, b) => a[2] - b[2]);                              // labels near the middle of the view get first pick of the space
  ctx.fillStyle = '#5f7fc0';
  for (const [l, q] of pos) placeLabel(l.name, cf, q[0], q[1], [[w => -w/2, 0], [w => -w/2, -15], [w => -w/2, 15], [w => -w/2, -30], [w => -w/2, 30], [w => 8, -8], [w => -w - 8, -8]]);
  ctx.fillStyle = '#c9d6ff'; const v = sky.v;
  for (const s of sky.names) {                                  // brightest stars first (sky.names is sorted by magnitude)
    const q = skyXY(toView([v[s.i*3], v[s.i*3+1], v[s.i*3+2]]), SK);
    if (q && q[0] > 0 && q[0] < W && q[1] > 0 && q[1] < H) placeLabel(s.name, sf, q[0], q[1], [[w => 6, -5], [w => -w - 6, -5], [w => 6, 12], [w => -w - 6, 12]]);
  }
}
function drawStarfield(alpha, fade = 1) {
  if (!sky) return;
  const SK = skyScale(), sf = 1;
  if (opt.cStars) { ctx.globalAlpha = alpha; paintStars(SK, sf); ctx.globalAlpha = 1; }
  if ((opt.cLines || opt.cNames) && fade > 0.01) { ctx.globalAlpha = alpha*fade; paintLines(SK); ctx.globalAlpha = 1; }
  if (opt.cStars && cursor) {
    const v = sky.v, R0 = R[0], R1 = R[1], R2 = R[2];
    for (const i of sky.bright) {
      const x = v[i*3], y = v[i*3+1], z = v[i*3+2], a = -(R2[0]*x + R2[1]*y + R2[2]*z); if (a < -0.9) continue;
      const k = SK/(1 + a), px = W/2 + (R0[0]*x + R0[1]*y + R0[2]*z)*k, py = H/2 - (R1[0]*x + R1[1]*y + R1[2]*z)*k;
      if (Math.abs(px - cursor[0]) > 10 || Math.abs(py - cursor[1]) > 10) continue;
      const nm = sky.nameOf.get(i), e = [x, y, z];
      addHit(px, py, 8, 0, () => tipHTML(nm || 'Star', 'star', [['RA / Dec (J2000)', fmtRADec(e)], ['Ecl. lon / lat', eclLonLat(e).map(q => q.toFixed(2) + '°').join(' / ')],
        ['Magnitude', sky.mag[i].toFixed(2)], ['Proper motion', Math.hypot(...sky.pm[i]).toFixed(0) + ' mas/yr'], ['Hipparcos', 'HIP ' + sky.hip[i]]], 'ESA Hipparcos position, moved to the date shown.'));
    }
  }
}

// ======================================== real asteroids, Trojans, Kuiper belt (NASA/JPL SBDB) ===
// static/small_bodies.json holds the real orbits of the brightest objects of each class. Each object is advanced from its
// own osculating elements (two-body motion), so the dots are real objects in their real places, and they move with time.
let small = null;
const SMALL_COL = {neo: '#d77a6a', belt: '#c2a173', trojan: '#e2a45f', tno: '#7a9ccc', lookup: '#e8e3d0', candidate: '#ff5ea8'};
// one class of objects from rows [a, e, i, node, peri, M, epoch, H, diameter, name, flags]
function makeClass(key, label, rows, extra = {}) {
  const n = rows.length;
  const o = Object.assign({key, label, n, col: SMALL_COL[key] || '#aaa', name: [], flags: [], H: new Float32Array(n), D: new Float32Array(n),
             a: new Float64Array(n), e: new Float64Array(n), nm: new Float64Array(n), m0: new Float64Array(n), ep: new Float64Array(n),
             P: new Float64Array(3*n), Q: new Float64Array(3*n), x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n), jd: -1e12, stride: 0}, extra);
  rows.forEach(([a, e, i, om, w, ma, epoch, H, D, name, flags], k) => {
    const I = i*D2R, O = om*D2R, W = w*D2R, cO = Math.cos(O), sO = Math.sin(O), cW = Math.cos(W), sW = Math.sin(W), cI = Math.cos(I), sI = Math.sin(I);
    o.a[k] = a; o.e[k] = e; o.m0[k] = ma*D2R; o.ep[k] = epoch; o.nm[k] = 0.01720209895/Math.pow(a, 1.5);          // Gauss's constant: rad/day
    o.P[3*k] = cW*cO - sW*sO*cI; o.P[3*k+1] = cW*sO + sW*cO*cI; o.P[3*k+2] = sW*sI;                                 // perifocal axes in the ecliptic frame
    o.Q[3*k] = -sW*cO - cW*sO*cI; o.Q[3*k+1] = -sW*sO + cW*cO*cI; o.Q[3*k+2] = cW*sI;
    o.H[k] = H == null ? NaN : H; o.D[k] = D == null ? NaN : D; o.name[k] = name || ('object ' + (k + 1)); o.flags[k] = flags || 'p';
  });
  return o;
}
async function loadSmallBodies() {
  try {
    const d = await api('/static/small_bodies.json'), out = {source: d.source, retrieved: d.retrieved, cls: []};
    for (const key in d.classes) { const c = d.classes[key]; out.cls.push(makeClass(key, c.label, c.rows, {total: c.total_known, cutoff: c.cutoff})); }
    small = out; buildCatalog();
  } catch (e) { console.warn('small bodies', e); }
}
function keplerXYZ(o, k, jd, out) {                                     // two-body position of object k of class o (AU, ecliptic J2000)
  const e = o.e[k], a = o.a[k], TWO_PI = 2*Math.PI;
  let M = (o.m0[k] + o.nm[k]*(jd - o.ep[k])) % TWO_PI; if (M > Math.PI) M -= TWO_PI; if (M < -Math.PI) M += TWO_PI;
  let E = e < 0.8 ? M + e*Math.sin(M) : Math.PI;
  for (let it = 0; it < 30; it++) { const d = (E - e*Math.sin(E) - M)/(1 - e*Math.cos(E)); E -= d; if (Math.abs(d) < 1e-10) break; }
  const xp = a*(Math.cos(E) - e), yp = a*Math.sqrt(1 - e*e)*Math.sin(E), j = 3*k;
  out[0] = xp*o.P[j] + yp*o.Q[j]; out[1] = xp*o.P[j+1] + yp*o.Q[j+1]; out[2] = xp*o.P[j+2] + yp*o.Q[j+2]; return out;
}
// Advance only the objects that will be drawn (every `stride`-th), and only when time has moved enough to show.
function updateClass(o, jd, stride) {
  const thr = o.key === 'tno' ? 10 : o.key === 'neo' ? 0.05 : 0.25;                                   // days; Kuiper-belt objects crawl
  if (o.stride === stride && Math.abs(jd - o.jd) < thr) return;
  o.jd = jd; o.stride = stride;
  const v = [0, 0, 0];
  for (let k = 0; k < o.n; k += stride) { keplerXYZ(o, k, jd, v); o.x[k] = v[0]; o.y[k] = v[1]; o.z[k] = v[2]; }
}
function drawRegions(c, s) {
  if (!opt.cBelts || !small || W/s < 0.5) return;
  const R0 = R[0], R1 = R[1], hw = W/2, hh = H/2, sz = 1.2 + Math.min(1.1, 0.5*Math.log2(Math.max(1, s/sFit()))), s2 = s*s;
  const areaAU2 = {neo: 6, belt: Math.PI*(3.3*3.3 - 2.1*2.1), trojan: 4, tno: Math.PI*(50*50 - 30*30)};
  for (const o of small.cls) {
    if (!(o.key in areaAU2)) continue;
    const stride = Math.max(1, Math.floor(o.n/clamp(areaAU2[o.key]*s2/5, 150, o.n)));      // never draw more dots than the screen area can show
    updateClass(o, sim.jd, stride);
    ctx.fillStyle = o.col; ctx.globalAlpha = o.key === 'belt' ? 0.6 : 0.7; ctx.beginPath();
    for (let i = 0; i < o.n; i += stride) {
      const x = o.x[i] - c[0], y = o.y[i] - c[1], z = o.z[i] - c[2];
      const px = hw + (R0[0]*x + R0[1]*y + R0[2]*z)*s, py = hh - (R1[0]*x + R1[1]*y + R1[2]*z)*s;
      if (px > -4 && px < W+4 && py > -4 && py < H+4) ctx.rect(px, py, sz, sz);
    }
    ctx.fill(); ctx.globalAlpha = 1;
  }
  const [sx0, sy0] = project([0,0,0], c, s), J = state.byName.Jupiter.pos, lj = Math.atan2(J[1], J[0]);
  const label = (o, r, th, col, text) => { const q = project([r*Math.cos(th), r*Math.sin(th), 0], c, s); if (q[0] < 0 || q[0] > W || q[1] < 0 || q[1] > H) return;
    if (Math.hypot(q[0] - sx0, q[1] - sy0) > 45) { ctx.fillStyle = col; ctx.font = 'italic 12px system-ui'; ctx.fillText(text || o.label, q[0] + 6, q[1]); }
    addHit(q[0], q[1], 16, -0.3, () => tipHTML(o.label, 'real objects', [['Shown', o.n.toLocaleString() + ' of ' + o.total.toLocaleString() + ' known'], ['Selection', 'brightest: ' + o.cutoff],
      ['Positions', "advanced from JPL's orbital elements for " + fmtDate(sim.jd, false)]],
      'Source: ' + small.source + ' (retrieved ' + small.retrieved + '). Each object moves on a two-body orbit from its own elements; accuracy drops for dates far from the element epoch. SPHEREx finds objects like these as sources that move between its repeated scans.')); };
  const by = Object.fromEntries(small.cls.map(o => [o.key, o]));
  if (by.belt) label(by.belt, 2.7, 0.5, '#d8b987', 'Main asteroid belt');
  if (by.tno) label(by.tno, 44, 2.2, '#8fb0dd', 'Kuiper belt');
  if (by.trojan) { label(by.trojan, 5.2, lj + Math.PI/3, '#e8b070', 'Jupiter Trojans (L4)'); label(by.trojan, 5.2, lj - Math.PI/3, '#e8b070', 'Jupiter Trojans (L5)'); }
}
