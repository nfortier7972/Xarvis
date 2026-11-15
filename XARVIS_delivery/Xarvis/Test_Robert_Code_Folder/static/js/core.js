// XARVIS viewer · core.js
// Shared constants, DOM helpers, layer toggles, the simulation clock, vector maths, frame conversions and JPL mean
// orbital elements (the planets outside the DE440s range). Everything else builds on this file.
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// =============================================================================== constants ===
const AU_KM = 149597870.7, D2R = Math.PI/180, EPS = (84381.406/3600)*D2R, EARTH_R_KM = 6371, J2000 = 2451545.0;
const COLORS = {Mercury:'#b5b5b5', Venus:'#e8cda2', Earth:'#4da3ff', Mars:'#e2724b', Jupiter:'#d9b38c', Saturn:'#e6d3a1',
                Uranus:'#8fe3e8', Neptune:'#5b7bff', Moon:'#c9c9c9'};
const PLANETS = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune'];
const RADIUS_KM = {Mercury:2440, Venus:6052, Earth:6371, Mars:3390, Jupiter:69911, Saturn:58232, Uranus:25362, Neptune:24622, Moon:1737};
const RATES = [['Real time',1/86400],['1 min / s',1/1440],['10 min / s',10/1440],['1 hour / s',1/24],['1 day / s',1],['1 week / s',7],
               ['1 month / s',30.4375],['1 year / s',365.25]];

// ================================================================================== helpers ===
const $ = id => document.getElementById(id);
const cv = $('c'), mainCtx = cv.getContext('2d');
let ctx = mainCtx;
let W = 0, H = 0, DPR = 1, K = 1, resScale = 1;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a)/(b - a), 0, 1); return t*t*(3 - 2*t); };
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = a => { const n = Math.hypot(a[0],a[1],a[2]) || 1; return [a[0]/n,a[1]/n,a[2]/n]; };
const sub = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const txt = {};
const setText = (id, v) => { if (txt[id] !== v) { txt[id] = v; $(id).textContent = v; } };
const setHTML = (id, v) => { if (txt[id] !== v) { txt[id] = v; $(id).innerHTML = v; } };
const opt = {};
for (const id of ['cStars','cLines','cOrbits','cBelts','cVisNow','cVisAll','cAnalysed']) {
  opt[id] = $(id).checked; $(id).addEventListener('change', () => { opt[id] = $(id).checked; opt.cNames = opt.cLines; });
}
opt.cNames = opt.cLines;
async function api(path) { const r = await fetch(path); if (!r.ok) throw new Error(path + ' ' + r.status); return r.json(); }

// ===================================================================================== time ===
const nowJD = () => Date.now()/86400000 + 2440587.5;
const sim = {jd: nowJD(), dir: 1, rateIdx: 0, live: true};
const simRate = () => sim.dir*RATES[sim.rateIdx][1];                       // days per real second
function jdToDate(jd) { return new Date((jd - 2440587.5)*86400000); }
function fmtDate(jd, withSeconds = true) {
  const d = jdToDate(jd); let y = d.getUTCFullYear();
  const ys = y < 0 ? '−' + String(-y).padStart(4,'0') : (y > 9999 ? '+' + y : String(y).padStart(4,'0'));
  const p = n => String(n).padStart(2,'0');
  return `${ys}-${p(d.getUTCMonth()+1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}` + (withSeconds ? `:${p(d.getUTCSeconds())}` : '');
}
function parseDate(s) {                                                    // "2030-06-01 12:00", "-500-03-01", "12000-1-1"
  const m = s.trim().replace('−','-').match(/^([+-]?\d{1,6})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*(?:UTC|Z)?$/i);
  if (!m) return null;
  const d = new Date(0); d.setUTCFullYear(+m[1], +m[2]-1, +m[3]); d.setUTCHours(+(m[4]||0), +(m[5]||0), +(m[6]||0), 0);
  const jd = d.getTime()/86400000 + 2440587.5;
  return isFinite(jd) ? jd : null;
}
const yearsSince2000 = jd => (jd - J2000)/365.25;
// Time is limited to the span SPHEREx has data for: from the first science data to the latest observation.
function advanceTime(dt) {
  if (sim.live) { sim.jd = Math.min(nowJD(), dataLatest()); return; }
  sim.jd += simRate()*dt;
  if (sim.jd <= DATA_FIRST) { sim.jd = DATA_FIRST; sim.dir = 0; }                       // reached the first data: stop
  else if (sim.jd >= dataLatest()) { sim.jd = dataLatest(); if (sim.dir > 0 && dataLatest() >= nowJD() - 1e-4) { sim.live = true; sim.rateIdx = 0; sim.dir = 1; $('speed').value = 0; } else sim.dir = 0; }   // reached now: go live
}
function setJD(jd) { sim.jd = clamp(jd, DATA_FIRST, dataLatest()); sim.live = false; }

// =================================================================================== maths ===
function matFromYawPitch(yaw, pitch) {
  const r0 = [Math.cos(yaw), -Math.sin(yaw), 0];
  const r1 = [Math.sin(pitch)*Math.sin(yaw), Math.sin(pitch)*Math.cos(yaw), Math.cos(pitch)];
  return [r0, r1, cross(r0, r1)];
}
function lookAt(f, upHint) {
  const z = [-f[0], -f[1], -f[2]];
  if (Math.abs(dot(f, upHint)) > 0.99) upHint = [1,0,0];
  const x = norm(cross(upHint, z));
  return [x, cross(z, x), z];
}
// R: rows = screen right, screen up, toward-camera, expressed in the heliocentric ecliptic J2000 frame
let R = matFromYawPitch(-0.5, 1.0);
function orthonormalize() { const r0 = norm(R[0]); let r1 = R[1]; const d = dot(r1, r0); r1 = norm([r1[0]-d*r0[0], r1[1]-d*r0[1], r1[2]-d*r0[2]]); R = [r0, r1, cross(r0, r1)]; }
function applyView(M) { R = [0,1,2].map(i => [0,1,2].map(j => M[i][0]*R[0][j] + M[i][1]*R[1][j] + M[i][2]*R[2][j])); orthonormalize(); }
const Ry = a => [[Math.cos(a),0,Math.sin(a)],[0,1,0],[-Math.sin(a),0,Math.cos(a)]];
const Rx = b => [[1,0,0],[0,Math.cos(b),-Math.sin(b)],[0,Math.sin(b),Math.cos(b)]];
const Rz = c => [[Math.cos(c),-Math.sin(c),0],[Math.sin(c),Math.cos(c),0],[0,0,1]];
function rotateBy(dx, dy, roll) { if (dx) applyView(Ry(dx)); if (dy) applyView(Rx(dy)); if (roll) applyView(Rz(roll)); }
const toView = v => [dot(R[0],v), dot(R[1],v), dot(R[2],v)];
const fromView = v => [R[0][0]*v[0]+R[1][0]*v[1]+R[2][0]*v[2], R[0][1]*v[0]+R[1][1]*v[1]+R[2][1]*v[2], R[0][2]*v[0]+R[1][2]*v[1]+R[2][2]*v[2]];
function eq2ecl(ra, dec) {
  const x = Math.cos(dec)*Math.cos(ra), y = Math.cos(dec)*Math.sin(ra), z = Math.sin(dec);
  return [x, y*Math.cos(EPS) + z*Math.sin(EPS), -y*Math.sin(EPS) + z*Math.cos(EPS)];
}
function eqVec2ecl(v) { return [v[0], v[1]*Math.cos(EPS) + v[2]*Math.sin(EPS), -v[1]*Math.sin(EPS) + v[2]*Math.cos(EPS)]; }
function ecl2eqVec(v) { return [v[0], v[1]*Math.cos(EPS) - v[2]*Math.sin(EPS), v[1]*Math.sin(EPS) + v[2]*Math.cos(EPS)]; }
function ecl2eq(v) { const e = ecl2eqVec(v); return [(Math.atan2(e[1], e[0])/D2R + 360) % 360, Math.asin(clamp(e[2], -1, 1))/D2R]; }
const GAL = [[-0.0548755604162154,-0.8734370902348850,-0.4838350155487132],[0.4941094278755837,-0.4448296299600112,0.7469822444972189],[-0.8676661490190047,-0.1980763734312015,0.4559837761750669]];
function ecl2gal(v) { const e = ecl2eqVec(v), g = [dot(GAL[0], e), dot(GAL[1], e), dot(GAL[2], e)];
  return [(Math.atan2(g[1], g[0])/D2R + 360) % 360, Math.asin(clamp(g[2], -1, 1))/D2R]; }
function eclLonLat(v) { return [(Math.atan2(v[1], v[0])/D2R + 360) % 360, Math.asin(clamp(v[2]/Math.hypot(v[0],v[1],v[2]), -1, 1))/D2R]; }
function fmtRA(deg, secs = true) { const t = deg/15, h = Math.floor(t), m = Math.floor((t-h)*60), s = Math.round(((t-h)*60-m)*60);
  const [hh, mm, ss] = s === 60 ? (m === 59 ? [(h+1)%24, 0, 0] : [h, m+1, 0]) : [h, m, s];
  return String(hh).padStart(2,'0') + 'h ' + String(mm).padStart(2,'0') + 'm' + (secs ? ' ' + String(ss).padStart(2,'0') + 's' : ''); }
function fmtDec(deg, secs = true) { const a = Math.abs(deg), d = Math.floor(a), m = Math.floor((a-d)*60), s = Math.round(((a-d)*60-m)*60);
  const [dd, mm, ss] = s === 60 ? (m === 59 ? [d+1, 0, 0] : [d, m+1, 0]) : [d, m, s];
  return (deg < 0 ? '−' : '+') + String(dd).padStart(2,'0') + '° ' + String(mm).padStart(2,'0') + '′' + (secs ? ' ' + String(ss).padStart(2,'0') + '″' : ''); }
const fmtRADec = v => { const [ra, dec] = ecl2eq(v); return fmtRA(ra, false) + ' / ' + fmtDec(dec, false); };

// ======================================================================= Keplerian planets ===
// Standish (JPL) mean elements, valid 1800-2050 (inner planets to ~0.01 deg, Jupiter-Neptune to ~0.2 deg); used only when time runs too fast for the
// ephemeris track (>1 day per second), where positions cannot be told apart at the speed anyway.
const KEP = {
  Mercury:[0.38709927,0.00000037,0.20563593,0.00001906,7.00497902,-0.00594749,252.25032350,149472.67411175,77.45779628,0.16047689,48.33076593,-0.12534081],
  Venus:[0.72333566,0.00000390,0.00677672,-0.00004107,3.39467605,-0.00078890,181.97909950,58517.81538729,131.60246718,0.00268329,76.67984255,-0.27769418],
  Earth:[1.00000261,0.00000562,0.01671123,-0.00004392,-0.00001531,-0.01294668,100.46457166,35999.37244981,102.93768193,0.32327364,0.0,0.0],
  Mars:[1.52371034,0.00001847,0.09339410,0.00007882,1.84969142,-0.00813131,-4.55343205,19140.30268499,-23.94362959,0.44441088,49.55953891,-0.29257343],
  Jupiter:[5.20288700,-0.00011607,0.04838624,-0.00013253,1.30439695,-0.00183714,34.39644051,3034.74612775,14.72847983,0.21252668,100.47390909,0.20469106],
  Saturn:[9.53667594,-0.00125060,0.05386179,-0.00050991,2.48599187,0.00193609,49.95424423,1222.49362201,92.59887831,-0.41897216,113.66242448,-0.28867794],
  Uranus:[19.18916464,-0.00196176,0.04725744,-0.00004397,0.77263783,-0.00242939,313.23810451,428.48202785,170.95427630,0.40805281,74.01692503,0.04240589],
  Neptune:[30.06992276,0.00026291,0.00859048,0.00005105,1.77004347,0.00035372,-55.12002969,218.45945325,44.96476227,-0.32241464,131.78422574,-0.00508664]};
function kepler(name, T, Eanom) {                          // heliocentric ecliptic J2000 position (AU); T = Julian centuries from J2000
  const q = KEP[name], a = q[0]+q[1]*T, e = q[2]+q[3]*T, inc = (q[4]+q[5]*T)*D2R, L = q[6]+q[7]*T, lp = q[8]+q[9]*T, om = q[10]+q[11]*T;
  const w = (lp - om)*D2R, node = om*D2R;
  let E = Eanom;
  if (E === undefined) { const M = (((L - lp) % 360) + 540) % 360 - 180; const Mr = M*D2R; E = Mr + e*Math.sin(Mr);
    for (let i = 0; i < 12; i++) E -= (E - e*Math.sin(E) - Mr)/(1 - e*Math.cos(E)); }
  const xp = a*(Math.cos(E) - e), yp = a*Math.sqrt(1 - e*e)*Math.sin(E);
  const cw = Math.cos(w), sw = Math.sin(w), co = Math.cos(node), so = Math.sin(node), ci = Math.cos(inc), si = Math.sin(inc);
  return [(cw*co - sw*so*ci)*xp + (-sw*co - cw*so*ci)*yp, (cw*so + sw*co*ci)*xp + (-sw*so + cw*co*ci)*yp, (sw*si)*xp + (cw*si)*yp];
}
function kepOrbit(name, T, n = 181) { const o = []; for (let i = 0; i <= n; i++) o.push(kepler(name, T, 2*Math.PI*i/n)); return o; }
