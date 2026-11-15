// XARVIS viewer · tracker.js
// Track objects panel: the catalogue of planets and JPL small bodies, search and filters, live JPL lookups, and the
// highlighted orbits of tracked objects. XARVIS_addCandidates() is the hook for SPHEREx candidates from the backend.
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// ========================================================================== object tracker ===
// Every object the viewer knows (planets, the Moon, SPHEREx, the JPL small bodies in small_bodies.json) can be searched,
// filtered and tracked; any other asteroid or comet can be looked up live in NASA/JPL's Small-Body Database.
// Tracked objects are highlighted on the map with their orbit, and say whether SPHEREx can see them right now.
const TYPE_LABEL = {planet: 'Planet', moon: 'Moon', spacecraft: 'Spacecraft', neo: 'Near-Earth object', belt: 'Main-belt asteroid',
                    trojan: 'Jupiter Trojan', tno: 'Kuiper-belt object', lookup: 'From JPL lookup', candidate: 'SPHEREx candidate'};
const TYPE_SHORT = {planet: 'Planet', moon: 'Moon', spacecraft: 'Spacecraft', neo: 'NEO', belt: 'Main belt', trojan: 'Trojan', tno: 'Kuiper belt', lookup: 'JPL lookup', candidate: 'Candidate'};
const ALBEDO = 0.14, estD = (H, p = ALBEDO) => 1329/Math.sqrt(p)*Math.pow(10, -H/5);        // diameter (km) from brightness H and an assumed albedo
const esc = t => String(t).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
let catalog = [], tracked = new Map(), focusId = null, lookupCls = null, candidateCls = null;
function smallEntry(o, k) {
  const D = o.D[k], H = o.H[k], f = o.flags[k];
  return {id: o.key + ':' + k, name: o.name[k], type: o.key, o, k, H, D: !isNaN(D) ? D : !isNaN(H) ? estD(H) : null, sizeKnown: !isNaN(D),
          numbered: f[0] === 'n', neo: f.includes('N'), pha: f.includes('P'), unknown: o.key === 'candidate', pos: () => keplerXYZ(o, k, sim.jd, [0, 0, 0])};
}
function buildCatalog() {
  catalog = PLANETS.map(n => ({id: 'pl:' + n, name: n, type: 'planet', D: 2*RADIUS_KM[n], sizeKnown: true, numbered: true, pos: () => state && state.byName[n].pos}));
  catalog.push({id: 'moon', name: 'Moon', type: 'moon', D: 2*RADIUS_KM.Moon, sizeKnown: true, numbered: true, pos: () => state && state.byName.Moon ? state.byName.Moon.pos : null});
  for (const o of [...(small ? small.cls : []), lookupCls, candidateCls]) if (o) for (let k = 0; k < o.n; k++) catalog.push(smallEntry(o, k));
  for (const e of catalog) e.key = e.name.toLowerCase();
  refreshTrackList();
}
// TODO(backend): SPHEREx moving-object candidates. Call XARVIS_addCandidates([[a, e, i, node, peri, M, epoch, H, D, name, flags], ...])
// with fitted orbits (same columns as small_bodies.json) and they appear under Known / unknown -> "Unknown: SPHEREx candidates".
window.XARVIS_addCandidates = rows => { candidateCls = makeClass('candidate', 'SPHEREx candidates', rows); buildCatalog(); };
function entryById(id) { return catalog.find(e => e.id === id) || tracked.get(id); }
function geoDir(p) { return state ? norm(sub(p, state.E)) : null; }
function sizeText(e) { if (e.D == null) return e.type === 'spacecraft' ? '2.6 m' : 'size unknown'; const d = e.D >= 100 ? Math.round(e.D).toLocaleString() : e.D >= 1 ? e.D.toFixed(1) : (e.D*1000).toFixed(0) + ' m'; return (e.D >= 1 ? d + ' km' : d) + (e.sizeKnown ? '' : ' (est.)'); }
function filtered() {
  const q = $('tq').value.trim().toLowerCase(), show = $('fShow').value, only = $('fOnly').value, out = [];
  for (const e of catalog) {
    if (q && !e.key.includes(q)) continue;
    if (show === 'candidate' ? !e.unknown : e.unknown) continue;                             // known vs unknown objects
    if (show !== 'all' && show !== 'candidate' && !(show === 'planets' ? (e.type === 'planet' || e.type === 'moon') : e.type === show)) continue;
    if (only === 'pha' && !e.pha) continue; if (only === 'big' && !(e.D >= 100)) continue; if (only === 'measured' && !(e.o && e.sizeKnown)) continue;
    if ((only === 'vis' || only === 'near') && state) {
      const p = e.pos(); if (!p || e.name === 'Earth') continue;
      if (only === 'near' && Math.hypot(...sub(p, state.E)) > 0.3) continue;
      if (only === 'vis' && !visibilityOf(geoDir(p)).visible_now) continue;
    }
    out.push(e);
  }
  const rank = {planet: 0, moon: 1, spacecraft: 2};
  // with a search: exact name or number first, then whole-word matches, then the rest; otherwise planets first, then by size
  const word = q ? new RegExp('(^|[^a-z0-9])' + q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '($|[^a-z0-9])') : null;
  const qs = e => !q ? 0 : (e.key === q || e.key.split(' (')[0].split(' ').includes(q) && e.key.split(' (')[0].endsWith(q)) ? 0 : word.test(e.key) ? 1 : 2;
  out.sort((a, b) => qs(a) - qs(b) || (rank[a.type] ?? 3) - (rank[b.type] ?? 3) || (b.D ?? -1) - (a.D ?? -1));
  return out;
}
let trackListT = 0;
function refreshTrackList() {
  if (!$('trackPanel') || $('trackPanel').hidden) return;
  trackListT = performance.now();
  const list = filtered(), q = $('tq').value.trim(), shown = list.slice(0, 80), st = $('fShow').value === 'candidate' ? 'unknown' : '';
  setText('tCount', list.length ? `${list.length.toLocaleString()} match${list.length === 1 ? '' : 'es'}` + (list.length > shown.length ? ` · showing the ${shown.length} largest` : '') : '');
  let html = shown.map(e => { const p = state && e.pos(), r = p ? Math.hypot(...p) : null;
    return `<div class="ti${tracked.has(e.id) ? ' on' : ''}" data-id="${esc(e.id)}"><span class="tn">${esc(e.name)}</span><span class="tm">${TYPE_SHORT[e.type]} · ${sizeText(e)}${r != null && e.type !== 'spacecraft' ? ' · ' + r.toFixed(2) + ' AU' : ''}${e.pha ? ' · PHA' : ''}</span></div>`; }).join('');
  if (!list.length) html = st === 'unknown' ? '<div class="tmsg">No SPHEREx candidates yet. Unknown objects will appear here once the SPHEREx analysis backend reports them.</div>'
                                            : '<div class="tmsg">Nothing in the built-in list matches. Try the JPL search below.</div>';
  $('tList').innerHTML = html;
  $('tLookup').hidden = !q; $('tLookup').textContent = `Search NASA/JPL for “${q.slice(0, 24)}”`;
}
function renderChips() {
  setText('trackN', tracked.size ? '· ' + tracked.size : '');
  $('trackChips').innerHTML = [...tracked.values()].map(e => `<span class="chip${focusId === e.id ? ' focus' : ''}" data-id="${esc(e.id)}"><b>${esc(e.name)}</b><button class="ghost cf" title="Centre the view on it">${focusId === e.id ? 'Following' : 'Centre'}</button><button class="ghost cx" title="Stop tracking">✕</button></span>`).join('');
}
function toggleTrack(id) {
  if (tracked.has(id)) { tracked.delete(id); if (focusId === id) focusId = null; }
  else { const e = entryById(id); if (!e) return; if (tracked.size >= 10) tracked.delete(tracked.keys().next().value); tracked.set(id, e); }
  renderChips(); refreshTrackList();
}
async function jplLookup(q) {
  $('tMsg').textContent = 'Asking NASA/JPL…';
  try {
    const r = await fetch('/api/sbdb/lookup?q=' + encodeURIComponent(q)), d = await r.json();
    if (d.matches) { $('tMsg').innerHTML = 'Several objects match. Pick one: ' + d.matches.map(m => `<a href="#" data-des="${esc(m.des)}">${esc(m.name)}</a>`).join(' · '); return; }
    if (!d.object) { $('tMsg').textContent = d.error || 'Not found.'; return; }
    const row = d.object.row, name = row[9], prev = lookupCls ? lookupCls.rawRows : [];
    if (prev.some(x => x[9] === name)) { $('tMsg').textContent = name + ' is already in the list.'; }
    else { const rows = [...prev, row]; lookupCls = makeClass('lookup', 'Looked up in NASA/JPL SBDB', rows, {rawRows: rows}); buildCatalog(); $('tMsg').textContent = `Added ${name} (${d.object.class || 'small body'}) from NASA/JPL.`; }
    const e = catalog.find(x => x.name === name); if (e && !tracked.has(e.id)) toggleTrack(e.id);
  } catch (err) { $('tMsg').textContent = 'Could not reach NASA/JPL from the server.'; }
}
function orbitPts(e) {
  if (e.o) { const o = e.o, k = e.k, a = o.a[k], ec = o.e[k], b = a*Math.sqrt(1 - ec*ec), j = 3*k, pts = [];
    for (let i = 0; i <= 240; i++) { const E = i/240*2*Math.PI, xp = a*(Math.cos(E) - ec), yp = b*Math.sin(E); pts.push([xp*o.P[j] + yp*o.Q[j], xp*o.P[j+1] + yp*o.Q[j+1], xp*o.P[j+2] + yp*o.Q[j+2]]); }
    return pts; }
  if (e.type === 'planet' && state.orbits) return state.orbits[e.name];
  return null;
}
function drawTracked(c, s) {
  if (!tracked.size || !state) return;
  for (const e of tracked.values()) {
    const p = e.pos(); if (!p) continue;
    const pts = orbitPts(e);
    if (pts) { ctx.strokeStyle = '#ffb81c'; ctx.globalAlpha = 0.75; ctx.lineWidth = 1.4; polyline(pts, c, s); ctx.globalAlpha = 1; }
    const [x, y] = project(p, c, s); if (x < -40 || x > W + 40 || y < -40 || y > H + 40) continue;
    const pulse = 10 + 2*Math.sin(performance.now()/300);
    ctx.strokeStyle = '#ffb81c'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, pulse, 0, 7); ctx.stroke();
    ctx.beginPath(); for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.moveTo(x + dx*(pulse + 3), y + dy*(pulse + 3)); ctx.lineTo(x + dx*(pulse + 9), y + dy*(pulse + 9)); } ctx.stroke();
    ctx.fillStyle = '#ffb81c'; ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 7); ctx.fill();
    ctx.font = 'bold 12px system-ui'; const w = measure('bold 12px system-ui', e.name); ctx.fillStyle = '#02040acc'; ctx.fillRect(x + 16, y - 22, w + 8, 17);
    ctx.fillStyle = '#ffd27a'; ctx.fillText(e.name, x + 20, y - 9);
    addHit(x, y, 14, 0.7, () => trackTip(e));
  }
}
function trackTip(e) {
  const p = e.pos(); if (!p) return tipHTML(esc(e.name), 'tracked', [], 'Not shown at this time or speed.');
  const rows = [['Type', TYPE_LABEL[e.type]], ['Size', sizeText(e)]];
  if (e.H != null && !isNaN(e.H)) rows.push(['Absolute magnitude H', e.H.toFixed(2)]);
  rows.push(['From the Sun', Math.hypot(...p).toFixed(3) + ' AU']);
  if (e.name !== 'Earth' && e.type !== 'spacecraft') { const g = sub(p, state.E); rows.push(['From Earth', distStr(Math.hypot(...g))], ['RA / Dec from Earth', fmtRADec(norm(g))]);
    const v = visibilityOf(norm(g)); rows.push(['SPHEREx can see it now', v.visible_now ? 'yes' : 'no · ' + v.reason]); }
  if (e.o) rows.push(['Orbit', `a ${e.o.a[e.k].toFixed(3)} AU · e ${e.o.e[e.k].toFixed(3)}`], ['Designation', e.numbered ? 'numbered (well-determined orbit)' : 'provisional']);
  const note = e.o ? (e.sizeKnown ? 'Measured diameter' : `Diameter estimated from H with albedo ${ALBEDO} (range ${sizeRange(e.H)})`) + '. Two-body orbit from NASA/JPL elements (epoch ' + fmtDate(e.o.ep[e.k], false).slice(0, 10) + '); accuracy drops far from that date.' : '';
  return tipHTML(esc(e.name), 'tracked', rows, note);
}
const sizeRange = H => { const lo = estD(H, 0.25), hi = estD(H, 0.05), f = d => d >= 1 ? d.toFixed(d >= 100 ? 0 : 1) + ' km' : (d*1000).toFixed(0) + ' m'; return f(lo) + '–' + f(hi); };
