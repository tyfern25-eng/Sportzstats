/* Sportzstats front end. Reads the JSON files in /data and config/leagues.json. */
const S = { meta: null, players: [], defs: [], teams: [], backtest: [], leagues: [], week: null, byId: {}, byName: {}, league: 0, sort: {} };

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "–" : `${Number(v).toFixed(1)}%`);
const signed = (v) => (v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${Number(v).toFixed(1)}%`);
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* storage unavailable */ } },
};

// ------------------------------------------------------------ name matching
function norm(n) {
  return String(n).toLowerCase().replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, "").replace(/[^a-z0-9/ ]/g, "").replace(/\s+/g, " ").trim();
}
const TEAM_ALIASES = {};
function buildIndex() {
  S.byId = {}; S.byName = {};
  for (const p of S.players) {
    S.byId[p.id] = p;
    const key = norm(p.name);
    // prefer the higher-usage player if two share a name
    if (!S.byName[key] || (p.ppg || 0) > (S.byName[key].ppg || 0)) S.byName[key] = p;
    if (p.pos === "DST") {
      const nick = p.name.replace(" D/ST", "");
      [nick, `${nick} dst`, `${nick} d/st`, `${nick} defense`, p.team, `${p.team} dst`, `${p.team} d/st`, `${p.team} defense`]
        .forEach((a) => { TEAM_ALIASES[norm(a)] = p; });
    }
  }
}
function findPlayer(name) {
  const k = norm(name);
  if (!k) return null;
  if (S.byName[k]) return S.byName[k];
  if (TEAM_ALIASES[k]) return TEAM_ALIASES[k];
  // fallback: unique last-name + first initial
  const parts = k.split(" ");
  if (parts.length >= 2) {
    const last = parts[parts.length - 1], fi = parts[0][0];
    const hits = S.players.filter((p) => { const n = norm(p.name).split(" "); return n[n.length - 1] === last && n[0][0] === fi; });
    if (hits.length) return hits.sort((a, b) => (b.ppg || 0) - (a.ppg || 0))[0];
  }
  const hits = S.players.filter((p) => norm(p.name).includes(k));
  return hits.length === 1 ? hits[0] : null;
}

// ------------------------------------------------------------ projections
function projFor(p, week = S.week) {
  return (p.proj || []).find((x) => x.week === week) || null;
}
const gradeTag = (g) => (g ? `<span class="grade g-${esc(g)}">${esc(g)}</span>` : "");
function oppText(pj) {
  if (!pj) return "–";
  if (pj.bye) return '<span class="tag bye">BYE</span>';
  return `${pj.home ? "vs" : "@"} ${esc(pj.opp)}`;
}
function flagsText(p, pj) {
  const out = [];
  if (pj && pj.flags) pj.flags.forEach((f) => out.push(`<span class="tag inj">${esc(f)}</span>`));
  else if (p.injury && p.injury.status && p.injury.week >= S.meta.through_week)
    out.push(`<span class="tag inj">${esc(p.injury.status)}</span>`);
  return out.join("");
}

// ------------------------------------------------------------ lineup optimizer
const ELIG = { QB: ["QB"], RB: ["RB"], WR: ["WR"], TE: ["TE"], FLEX: ["RB", "WR", "TE"], OP: ["QB", "RB", "WR", "TE"], "SUPERFLEX": ["QB", "RB", "WR", "TE"], "D/ST": ["DST"], DST: ["DST"], K: ["K"], "RB/WR": ["RB", "WR"], "WR/TE": ["WR", "TE"] };
function optimize(roster, slots) {
  const val = (p) => { const pj = projFor(p); return pj && !pj.bye ? pj.proj || 0 : -1; };
  const pool = [...roster].sort((a, b) => val(b) - val(a));
  const used = new Set();
  // fill strict slots first, flexible slots last
  const order = slots.map((s, i) => ({ s: s.toUpperCase(), i })).sort((a, b) => (ELIG[a.s]?.length || 9) - (ELIG[b.s]?.length || 9));
  const lineup = new Array(slots.length).fill(null);
  for (const { s, i } of order) {
    const elig = ELIG[s] || [s];
    const pick = pool.find((p) => !used.has(p.id) && elig.includes(p.pos));
    if (pick) { lineup[i] = pick; used.add(pick.id); }
  }
  const bench = pool.filter((p) => !used.has(p.id));
  return { lineup, bench, val };
}

// ------------------------------------------------------------ leagues
function leagueRoster(lg) {
  const saved = store.get(`ss_league_${lg.id}`);
  return saved || { name: lg.name, team: lg.team, slots: lg.slots, roster: lg.roster };
}

function renderTeams() {
  const el = $("#tab-teams");
  const lgs = S.leagues;
  const cur = lgs[S.league];
  const data = leagueRoster(cur);
  const matched = [], missing = [];
  for (const n of data.roster || []) { const p = findPlayer(n); p ? matched.push(p) : missing.push(n); }
  const uniq = [...new Map(matched.map((p) => [p.id, p])).values()];

  let html = `<div class="section-head"><div><h2>My Teams · Week ${S.week}</h2>
    <p>Best lineup for each league based on this week's projections. Click any player for the full breakdown.</p></div></div>
    <div class="pills">${lgs.map((l, i) => `<button class="pill ${i === S.league ? "active" : ""}" data-lg="${i}">${esc(leagueRoster(l).name)}</button>`).join("")}</div>`;

  html += `<div class="card"><div class="section-head" style="margin:0 0 12px"><div><h3>${esc(data.name)}</h3>
    <p>${esc(data.team || "")} · ${esc((data.slots || []).join(", "))}</p></div>
    <div><button class="btn" id="edit-roster">Edit roster</button></div></div>`;

  if (!uniq.length) {
    html += `<div class="empty"><h3>No players yet</h3><p>Click <b>Edit roster</b> and paste your players, one per line.</p></div></div>`;
    el.innerHTML = html;
    bindTeams();
    return;
  }
  if (missing.length) html += `<div class="warnbox">Couldn't find: ${missing.map(esc).join(", ")}. Check spelling (team defenses can be written like "Vikings D/ST").</div>`;

  const { lineup, bench, val } = optimize(uniq, data.slots);
  const starterVals = lineup.filter(Boolean).map(val);
  const row = (p, slot, isBench) => {
    if (!p) return `<tr><td class="slot">${esc(slot)}</td><td colspan="7" class="muted">Empty: no eligible player</td></tr>`;
    const pj = projFor(p);
    // close call: a bench player within 1.5 pts of the weakest starter at an eligible slot
    let close = "";
    if (isBench && pj && !pj.bye) {
      const rivals = lineup.filter((s, i) => s && (ELIG[data.slots[i].toUpperCase()] || []).includes(p.pos));
      const minR = Math.min(...rivals.map(val));
      if (rivals.length && (pj.proj || 0) >= minR - 1.5) close = `<span class="tag close">Close call</span>`;
    }
    const note = pj && pj.notes && pj.notes.length ? `<div class="why">${esc(pj.notes[0].text)}</div>` : "";
    return `<tr class="clickable ${isBench ? "bench" : ""}" data-pid="${esc(p.id)}">
      <td class="slot">${esc(slot)}</td>
      <td><div class="pname">${esc(p.name)} ${flagsText(p, pj)}${close}<small>${esc(p.pos)} · ${esc(p.team)}</small></div>${note}</td>
      <td>${oppText(pj)}</td>
      <td class="num proj">${pj && !pj.bye ? fmt(pj.proj) : "0.0"}</td>
      <td class="num muted">${pj && !pj.bye ? `${fmt(pj.floor)}–${fmt(pj.ceil)}` : "–"}</td>
      <td>${pj ? gradeTag(pj.grade) : ""}</td>
      <td class="num">${fmt(p.ppg)}</td>
      <td class="num">${fmt(p.ros?.ppg)}</td></tr>`;
  };
  const total = starterVals.filter((v) => v > 0).reduce((a, b) => a + b, 0);
  html += `<div class="big" style="margin-bottom:12px"><div><div class="val">${fmt(total)}</div><span class="muted">Projected starting lineup</span></div></div>
    <div class="table-wrap"><table><thead><tr><th>Slot</th><th>Player</th><th>Opp</th><th class="num">Proj</th><th class="num">Range</th><th>Matchup</th><th class="num">PPG</th><th class="num">ROS PPG</th></tr></thead><tbody>
    ${lineup.map((p, i) => row(p, data.slots[i], false)).join("")}
    <tr class="divider"><td colspan="8">Bench</td></tr>
    ${bench.map((p) => row(p, "BN", true)).join("")}
    </tbody></table></div></div>`;

  // rest of season outlook
  const ros = [...uniq].sort((a, b) => (b.ros?.ppg || 0) - (a.ros?.ppg || 0));
  html += `<div class="card"><h3 style="margin-bottom:10px">Rest-of-season outlook</h3><div class="table-wrap"><table><thead><tr>
    <th>Player</th><th class="num">ROS PPG</th><th class="num">ROS total</th><th>Schedule</th><th>Playoffs (15-17)</th><th class="num">Usage trend</th><th>Outlook</th></tr></thead><tbody>
    ${ros.map((p) => {
      const r = p.ros || {};
      const sg = r.sched ? gradeTag(gradeOf(r.sched)) : "–";
      const pg = r.playoff_sched ? gradeTag(gradeOf(r.playoff_sched)) : "–";
      const tr = r.trend == null ? "–" : `<span class="${r.trend > 0 ? "up" : r.trend < 0 ? "down" : ""}">${r.trend > 0 ? "+" : ""}${fmt(r.trend)}</span>`;
      return `<tr class="clickable" data-pid="${esc(p.id)}"><td class="pname">${esc(p.name)}<small>${esc(p.pos)} · ${esc(p.team)}</small></td>
        <td class="num">${fmt(r.ppg)}</td><td class="num">${fmt(r.total)}</td><td>${sg}</td><td>${pg}</td><td class="num">${tr}</td>
        <td><div class="why" style="max-width:520px;margin:0">${esc(r.outlook || "")}</div></td></tr>`;
    }).join("")}</tbody></table></div></div>`;

  el.innerHTML = html;
  bindTeams();
}
function gradeOf(m) {
  const cuts = [[1.15, "A+"], [1.08, "A"], [1.03, "B+"], [0.98, "B"], [0.94, "C+"], [0.9, "C"], [0.85, "D"]];
  for (const [c, g] of cuts) if (m >= c) return g;
  return "F";
}
function bindTeams() {
  const el = $("#tab-teams");
  el.querySelectorAll("[data-lg]").forEach((b) => b.onclick = () => { S.league = +b.dataset.lg; renderTeams(); });
  el.querySelectorAll("[data-pid]").forEach((r) => r.onclick = () => openPlayer(r.dataset.pid));
  const eb = $("#edit-roster");
  if (eb) eb.onclick = () => editRoster();
}
function editRoster() {
  const lg = S.leagues[S.league];
  const d = leagueRoster(lg);
  $("#modal-body").innerHTML = `<h2>Edit ${esc(d.name)}</h2>
    <p class="muted">Saved in this browser. To make it permanent everywhere, put the same names in <code>config/leagues.json</code> in the repo.</p>
    <div class="grid2"><div><label class="muted">League name</label><input id="e-name" style="width:100%" value="${esc(d.name)}"></div>
    <div><label class="muted">Your team name</label><input id="e-team" style="width:100%" value="${esc(d.team || "")}"></div></div>
    <p><label class="muted">Lineup slots (comma separated: QB, RB, WR, TE, FLEX, OP, D/ST, K)</label>
    <input id="e-slots" style="width:100%" value="${esc((d.slots || []).join(", "))}"></p>
    <p><label class="muted">Roster: one player per line (defenses like "Vikings D/ST")</label>
    <textarea id="e-roster" rows="16">${esc((d.roster || []).join("\n"))}</textarea></p>
    <div style="display:flex;gap:8px"><button class="btn primary" id="e-save">Save</button><button class="btn" id="e-reset">Reset to file</button></div>`;
  showModal();
  $("#e-save").onclick = () => {
    store.set(`ss_league_${lg.id}`, {
      name: $("#e-name").value.trim() || lg.name, team: $("#e-team").value.trim(),
      slots: $("#e-slots").value.split(",").map((s) => s.trim()).filter(Boolean),
      roster: $("#e-roster").value.split("\n").map((s) => s.trim()).filter(Boolean),
    });
    hideModal(); renderTeams();
  };
  $("#e-reset").onclick = () => { store.del(`ss_league_${lg.id}`); hideModal(); renderTeams(); };
}

// ------------------------------------------------------------ players table
const PCOLS = [
  ["name", "Player", (p) => `<span class="pname">${esc(p.name)} ${flagsText(p, projFor(p))}<small>${esc(p.pos)} · ${esc(p.team)}</small></span>`, (p) => p.name],
  ["proj", "Proj", (p) => { const j = projFor(p); return j && !j.bye ? `<b class="proj">${fmt(j.proj)}</b>` : "BYE"; }, (p) => projFor(p)?.proj ?? -1, true],
  ["opp", "Opp", (p) => oppText(projFor(p)), (p) => projFor(p)?.opp || ""],
  ["grade", "Matchup", (p) => gradeTag(projFor(p)?.grade), (p) => projFor(p)?.mult ?? 0],
  ["ppg", "PPG", (p) => fmt(p.ppg), (p) => p.ppg ?? 0, true],
  ["l3", "Last 3", (p) => fmt(p.ppg_l3), (p) => p.ppg_l3 ?? 0, true],
  ["ros", "ROS PPG", (p) => fmt(p.ros?.ppg), (p) => p.ros?.ppg ?? 0, true],
  ["snap", "Snap %", (p) => fmt(p.adv?.snap), (p) => p.adv?.snap ?? 0, true],
  ["tgt", "Tgt share", (p) => pct(p.adv?.tgt_share), (p) => p.adv?.tgt_share ?? 0, true],
  ["tgtb", "Tgt % vs blitz", (p) => pct(p.adv?.tgt_share_blitz), (p) => p.adv?.tgt_share_blitz ?? 0, true],
  ["tgtnb", "Tgt % no blitz", (p) => pct(p.adv?.tgt_share_noblitz), (p) => p.adv?.tgt_share_noblitz ?? 0, true],
  ["wopr", "WOPR", (p) => fmt(p.adv?.wopr, 2), (p) => p.adv?.wopr ?? 0, true],
  ["adot", "aDOT", (p) => fmt(p.adv?.adot), (p) => p.adv?.adot ?? 0, true],
  ["rz", "RZ tgt %", (p) => pct(p.adv?.rz_share), (p) => p.adv?.rz_share ?? 0, true],
  ["car", "Carry share", (p) => pct(p.adv?.carry_share), (p) => p.adv?.carry_share ?? 0, true],
  ["gl", "Goal-line %", (p) => pct(p.adv?.gl_share), (p) => p.adv?.gl_share ?? 0, true],
  ["ypc", "YPC", (p) => fmt(p.adv?.ypc), (p) => p.adv?.ypc ?? 0, true],
  ["ypt", "Yds/tgt", (p) => fmt(p.adv?.ypt), (p) => p.adv?.ypt ?? 0, true],
  ["yac", "YAC/rec", (p) => fmt(p.adv?.yac_rec), (p) => p.adv?.yac_rec ?? 0, true],
  ["catch", "Catch %", (p) => pct(p.adv?.catch), (p) => p.adv?.catch ?? 0, true],
];
function renderPlayers() {
  const el = $("#tab-players");
  if (!el.dataset.ready) {
    const teams = [...new Set(S.players.map((p) => p.team))].sort();
    el.innerHTML = `<div class="section-head"><div><h2>Player Database</h2>
      <p>Every offensive player with advanced usage stats. Shares are weighted toward recent games. Click a column to sort, a player to open him.</p></div></div>
      <div class="filters"><input id="f-q" placeholder="Search players…">
      <select id="f-pos"><option value="">All positions</option>${["QB", "RB", "WR", "TE", "K", "DST"].map((x) => `<option>${x}</option>`).join("")}</select>
      <select id="f-team"><option value="">All teams</option>${teams.map((t) => `<option>${t}</option>`).join("")}</select>
      <select id="f-min"><option value="5">PPG ≥ 5</option><option value="0">Everyone</option><option value="10">PPG ≥ 10</option></select></div>
      <div class="table-wrap" id="ptable"></div>`;
    el.dataset.ready = 1;
    ["f-q", "f-pos", "f-team", "f-min"].forEach((id) => $(`#${id}`).addEventListener("input", drawPlayers));
    S.sort.players = S.sort.players || { key: "proj", asc: false };
  }
  drawPlayers();
}
function drawPlayers() {
  const q = norm($("#f-q").value), pos = $("#f-pos").value, team = $("#f-team").value, min = +$("#f-min").value;
  let rows = S.players.filter((p) => (!q || norm(p.name).includes(q)) && (!pos || p.pos === pos) && (!team || p.team === team) && ((p.ppg || 0) >= min || q));
  const { key, asc } = S.sort.players;
  const col = PCOLS.find((c) => c[0] === key);
  rows.sort((a, b) => { const x = col[3](a), y = col[3](b); return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); });
  rows = rows.slice(0, 300);
  $("#ptable").innerHTML = `<table><thead><tr>${PCOLS.map((c) => `<th data-k="${c[0]}" class="${c[4] ? "num" : ""} ${key === c[0] ? "sorted" + (asc ? " asc" : "") : ""}">${c[1]}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((p) => `<tr class="clickable" data-pid="${esc(p.id)}">${PCOLS.map((c) => `<td class="${c[4] ? "num" : ""}">${c[2](p)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("#ptable").querySelectorAll("th").forEach((th) => th.onclick = () => {
    const k = th.dataset.k; const s = S.sort.players;
    S.sort.players = { key: k, asc: s.key === k ? !s.asc : k === "name" || k === "opp" };
    drawPlayers();
  });
  $("#ptable").querySelectorAll("[data-pid]").forEach((r) => r.onclick = () => openPlayer(r.dataset.pid));
}

// ------------------------------------------------------------ defenses
function heat(v, invert = false) {
  if (v === null || v === undefined) return "";
  const x = Math.max(-1, Math.min(1, (invert ? -v : v) / 15));
  const c = x > 0 ? `rgba(61,220,132,${0.08 + 0.3 * x})` : `rgba(255,93,93,${0.08 + 0.3 * -x})`;
  return `style="background:${c}"`;
}
function renderDefenses() {
  const el = $("#tab-defenses");
  const nextOpp = {};
  S.teams.forEach((t) => { const g = (t.sched || []).find((s) => s && s.week === S.week); if (g) nextOpp[t.team] = g; });
  const cols = [
    ["name", "Defense"], ["blitz", "Blitz %"], ["pressure", "Pressure %"], ["sack", "Sack %"],
    ["pass_adj", "Pass yds adj"], ["rush_adj", "Rush yds adj"], ["ypt_RB", "vs RB tgts"], ["ypt_WR", "vs WR tgts"], ["ypt_TE", "vs TE tgts"],
    ["fQB", "QB pts/g"], ["fRB", "RB pts/g"], ["fWR", "WR pts/g"], ["fTE", "TE pts/g"], ["pa_g", "Pts allowed"],
  ];
  const val = (d, k) => (k.startsWith("f") && k.length === 3 ? d.fpa[k.slice(1)] : d[k]);
  const s = S.sort.defs || (S.sort.defs = { key: "blitz", asc: false });
  const rows = [...S.defs].sort((a, b) => { const x = val(a, s.key), y = val(b, s.key); return (x > y ? 1 : x < y ? -1 : 0) * (s.asc ? 1 : -1); });
  el.innerHTML = `<div class="section-head"><div><h2>Defensive Tendencies</h2>
    <p>Blitz and pressure rates, plus how many more (green) or fewer (red) yards each defense gives up than its opponents normally gain. Adjusted for strength of opponent. "Pts/g" columns are PPR points allowed per game to each position.</p></div></div>
    <div class="table-wrap"><table><thead><tr>${cols.map((c) => `<th data-k="${c[0]}" class="${c[0] !== "name" ? "num" : ""} ${s.key === c[0] ? "sorted" + (s.asc ? " asc" : "") : ""}">${c[1]}</th>`).join("")}<th>Wk ${S.week} opp</th></tr></thead><tbody>
    ${rows.map((d) => {
      const g = nextOpp[d.team];
      return `<tr><td class="pname">${esc(d.name)}<small>${esc(d.team)}</small></td>
      <td class="num">${pct(d.blitz)} <span class="muted">#${d.blitz_rank}</span></td>
      <td class="num">${pct(d.pressure)} <span class="muted">#${d.pressure_rank}</span></td>
      <td class="num">${pct(d.sack)}</td>
      ${["pass_adj", "rush_adj", "ypt_RB", "ypt_WR", "ypt_TE"].map((k) => `<td class="num" ${heat(d[k])}>${signed(d[k])}</td>`).join("")}
      ${["QB", "RB", "WR", "TE"].map((k) => `<td class="num">${fmt(d.fpa[k])} <span class="muted">#${d.fpa_rank[k] ?? "–"}</span></td>`).join("")}
      <td class="num">${fmt(d.pa_g)}</td>
      <td>${g ? `${g.home ? "vs" : "@"} ${esc(g.opp)}` : '<span class="tag bye">BYE</span>'}</td></tr>`;
    }).join("")}</tbody></table></div>
    <p class="muted" style="margin-top:8px">Blitz % blends this season with a little of last season so small samples don't swing wildly. Rank #1 = blitzes/pressures most; for points allowed, #1 = most generous.</p>`;
  el.querySelectorAll("th[data-k]").forEach((th) => th.onclick = () => {
    const k = th.dataset.k; S.sort.defs = { key: k, asc: S.sort.defs.key === k ? !S.sort.defs.asc : k === "name" }; renderDefenses();
  });
}

// ------------------------------------------------------------ QB vs blitz
function renderBlitz() {
  const el = $("#tab-blitz");
  const qbs = S.players.filter((p) => p.pos === "QB" && p.qb_blitz && (p.qb_blitz.n_db || 0) >= 60)
    .sort((a, b) => (b.qb_blitz.ydb_b - b.qb_blitz.ydb_nb) - (a.qb_blitz.ydb_b - a.qb_blitz.ydb_nb));
  const defBy = Object.fromEntries(S.defs.map((d) => [d.team, d]));
  const lg = S.meta.league;
  el.innerHTML = `<div class="section-head"><div><h2>QB vs the Blitz</h2>
    <p>How each starting QB changes when the defense sends extra rushers. "Short %" is the share of throws 5 yards or less past the line (check-downs). The target-mix columns show where the ball goes when blitzed vs not, which feeds directly into the projections for his receivers. League average: blitz on ${fmt(lg.blitz)}% of dropbacks.</p></div></div>
    <div class="table-wrap"><table><thead><tr><th>QB</th><th class="num">Yds/dropback blitzed</th><th class="num">Not blitzed</th><th class="num">Sack % blitzed</th><th class="num">Not</th>
    <th class="num">Short % blitzed</th><th class="num">Not</th><th class="num">RB tgt % blitzed / not</th><th class="num">TE tgt % blitzed / not</th><th class="num">WR tgt % blitzed / not</th><th>Wk ${S.week} opp blitz</th></tr></thead><tbody>
    ${qbs.map((p) => {
      const q = p.qb_blitz, pj = projFor(p), od = pj && !pj.bye ? defBy[pj.opp] : null;
      const d = (a, b) => `${fmt(a)} / ${fmt(b)}`;
      return `<tr class="clickable" data-pid="${esc(p.id)}"><td class="pname">${esc(p.name)}<small>${esc(p.team)}</small></td>
        <td class="num ${q.ydb_b > q.ydb_nb ? "up" : "down"}">${fmt(q.ydb_b)}</td><td class="num">${fmt(q.ydb_nb)}</td>
        <td class="num">${fmt(q.sack_b)}</td><td class="num">${fmt(q.sack_nb)}</td>
        <td class="num">${fmt(q.short_b)}</td><td class="num">${fmt(q.short_nb)}</td>
        <td class="num">${d(q.mix_b_RB, q.mix_nb_RB)}</td><td class="num">${d(q.mix_b_TE, q.mix_nb_TE)}</td><td class="num">${d(q.mix_b_WR, q.mix_nb_WR)}</td>
        <td>${od ? `${pj.home ? "vs" : "@"} ${esc(pj.opp)} · ${pct(od.blitz)} <span class="muted">#${od.blitz_rank}</span>` : "–"}</td></tr>`;
    }).join("")}</tbody></table></div>
    <p class="muted" style="margin-top:8px">Splits blend this season with some of last season and are pulled toward league average when the sample is small.</p>`;
  el.querySelectorAll("[data-pid]").forEach((r) => r.onclick = () => openPlayer(r.dataset.pid));
}

// ------------------------------------------------------------ model page
function renderModel() {
  const bt = S.backtest || [];
  const avg = (k) => bt.length ? bt.reduce((a, b) => a + (b[k] || 0), 0) / bt.length : null;
  $("#tab-model").innerHTML = `<div class="section-head"><div><h2>How the projections work</h2></div></div>
    <div class="grid2"><div class="card"><h3>1 · Opportunity</h3><p>For every player: share of team targets, air yards, red-zone targets, carries and goal-line carries, weighted so recent games count more (a game 4 weeks ago counts half as much as last week).</p>
    <h3>2 · Team volume</h3><p>How many plays, targets and carries the team should get, adjusted for the opponent's pace and the betting spread (favorites run more, underdogs throw more). Touchdowns come from the Vegas implied team total when available.</p></div>
    <div class="card"><h3>3 · Matchup</h3><p>Opponent-adjusted defense strength: yards per target allowed to RBs, WRs and TEs separately, yards per carry, and pass yards per dropback, all compared to what those offenses normally do.</p>
    <h3>4 · Blitz tendencies</h3><p>For each QB, where the ball goes when blitzed vs not (check-downs to RBs and TEs, or forcing it to WRs) and how efficient he is. Against a heavy-blitz defense, targets shift toward the players his QB favors under pressure.</p></div></div>
    <div class="card"><h3>Track record</h3><p>Each past week was projected using only the data available before it, then compared with what happened. "Season average" is the simple alternative of using each player's points per game so far.</p>
    ${bt.length ? `<div class="table-wrap"><table><thead><tr><th>Week</th><th class="num">Players</th><th class="num">Avg miss: model</th><th class="num">Avg miss: season avg</th><th class="num">Right start/sit: model</th><th class="num">Season avg</th></tr></thead><tbody>
    ${bt.map((b) => `<tr><td>${b.week}</td><td class="num">${b.n}</td><td class="num">${fmt(b.mae_model, 2)}</td><td class="num">${fmt(b.mae_naive, 2)}</td><td class="num">${fmt(b.startsit_model)}%</td><td class="num">${fmt(b.startsit_naive)}%</td></tr>`).join("")}
    <tr><td><b>Average</b></td><td></td><td class="num"><b>${fmt(avg("mae_model"), 2)}</b></td><td class="num">${fmt(avg("mae_naive"), 2)}</td><td class="num"><b>${fmt(avg("startsit_model"))}%</b></td><td class="num">${fmt(avg("startsit_naive"))}%</td></tr>
    </tbody></table></div>` : `<p class="muted">Not run yet.</p>`}</div>
    <div class="card"><h3>Data</h3><p>Play-by-play, charting (blitzers per play), snap counts, injuries and schedules from <a href="https://github.com/nflverse" target="_blank" rel="noopener">nflverse</a>. Refreshed automatically several times a week. Last update: ${esc(S.meta.generated)} UTC, games through week ${S.meta.through_week}.</p></div>`;
}

// ------------------------------------------------------------ player modal
let chart = null;
function openPlayer(pid) {
  const p = S.byId[pid];
  if (!p) return;
  const pj = projFor(p);
  const a = p.adv || {};
  const inj = p.injury && p.injury.status ? `<span class="tag inj">${esc(p.injury.status)} · ${esc(p.injury.injury || "")} (wk ${p.injury.week})</span>` : "";
  const notes = pj && pj.notes ? pj.notes.map((n) => `<li class="${n.good === true ? "good" : n.good === false ? "bad" : "info"}">${esc(n.text)}</li>`).join("") : "";
  const c = pj && pj.comp ? pj.comp : null;
  const compLine = c ? (p.pos === "QB"
    ? `${fmt(c.pass_att)} att · ${fmt(c.pass_yds, 0)} pass yds · ${fmt(c.pass_td, 2)} pass TD · ${fmt(c.ints, 2)} INT · ${fmt(c.carries)} car · ${fmt(c.rush_yds, 0)} rush yds`
    : `${fmt(c.targets)} tgt · ${fmt(c.rec)} rec · ${fmt(c.rec_yds, 0)} rec yds · ${fmt(c.carries)} car · ${fmt(c.rush_yds, 0)} rush yds · ${fmt((c.rec_td || 0) + (c.rush_td || 0), 2)} TD`) : "";
  const stat = (v, l) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`;
  let adv = "";
  if (p.pos === "QB" && p.qb_blitz) {
    const q = p.qb_blitz;
    adv = [stat(fmt(a.ypa), "Yds/att"), stat(fmt(q.ydb_b), "Yds/db blitzed"), stat(fmt(q.ydb_nb), "Yds/db not"), stat(pct(q.sack_b), "Sack % blitzed"),
      stat(pct(q.short_b), "Short % blitzed"), stat(pct(q.short_nb), "Short % not"), stat(pct(q.blitz_faced), "Blitzed rate"), stat(pct(a.carry_share), "Team carries")].join("");
  } else if (["RB", "WR", "TE"].includes(p.pos)) {
    adv = [stat(fmt(a.snap), "Snap %"), stat(pct(a.tgt_share), "Target share"), stat(pct(a.tgt_share_l3), "Tgt share L3"), stat(pct(a.air_share), "Air yds share"),
      stat(fmt(a.wopr, 2), "WOPR"), stat(fmt(a.adot), "aDOT"), stat(pct(a.rz_share), "RZ tgt share"), stat(pct(a.carry_share), "Carry share"),
      stat(pct(a.gl_share), "Goal-line share"), stat(fmt(a.ypc), "Yds/carry"), stat(fmt(a.ypt), "Yds/target"), stat(fmt(a.yac_rec), "YAC/rec"),
      stat(pct(a.catch), "Catch rate"), stat(fmt(a.epa_tgt, 2), "EPA/target"),
      stat(pct(a.tgt_share_blitz), `Tgt % vs blitz (${a.tgt_blitz_n ?? 0})`), stat(pct(a.tgt_share_noblitz), `Tgt % no blitz (${a.tgt_noblitz_n ?? 0})`)].join("");
  }
  const sched = (p.proj || []).map((x) => `<tr><td>${x.week}</td><td>${oppText(x)}</td><td class="num">${x.bye ? "–" : fmt(x.proj)}</td><td class="num muted">${x.bye ? "" : `${fmt(x.floor)}–${fmt(x.ceil)}`}</td><td>${x.bye ? "" : gradeTag(x.grade)}</td></tr>`).join("");
  const logCols = p.pos === "QB" ? ["patt", "pyd", "ptd", "ints", "car", "ruyd"] : p.pos === "DST" ? ["sacks", "ints", "fum", "pa"] : p.pos === "K" ? [] : ["tgt", "rec", "ryd", "car", "ruyd", "td"];
  const logHead = { patt: "Att", pyd: "Pass yds", ptd: "Pass TD", ints: "INT", car: "Car", ruyd: "Rush yds", tgt: "Tgt", rec: "Rec", ryd: "Rec yds", td: "TD", sacks: "Sacks", fum: "FR", pa: "Pts allowed" };
  const log = [...(p.log || [])].reverse().map((g) => `<tr><td>${g.week}</td><td>${esc(g.opp || "")}</td><td class="num"><b>${fmt(g.ppr)}</b></td>${logCols.map((k) => `<td class="num">${g[k] ?? "–"}</td>`).join("")}</tr>`).join("");

  $("#modal-body").innerHTML = `<div class="phead">${p.headshot ? `<img src="${esc(p.headshot)}" alt="">` : ""}<div>
      <h2>${esc(p.name)}</h2><div class="muted">${esc(p.pos)} · ${esc(p.team)} · ${p.games} games · ${fmt(p.ppg)} PPG ${inj}</div></div></div>
    <div class="grid2">
      <div class="card"><div class="big"><div><div class="val">${pj && !pj.bye ? fmt(pj.proj) : "BYE"}</div><span class="muted">Week ${S.week} ${oppText(pj)}</span></div>
        ${pj && !pj.bye ? `<div><div class="mono" style="font-size:18px">${fmt(pj.floor)} – ${fmt(pj.ceil)}</div><span class="muted">Likely range</span></div><div>${gradeTag(pj.grade)}<br><span class="muted">Matchup</span></div>` : ""}</div>
        ${compLine ? `<p class="muted" style="margin:10px 0 0">${compLine}</p>` : ""}
        ${pj && pj.flags && pj.flags.length ? `<div class="warnbox">${pj.flags.map(esc).join(" · ")}</div>` : ""}
        ${notes ? `<ul class="notes">${notes}</ul>` : `<p class="muted">No major matchup factors this week; projection is mostly his normal usage.</p>`}</div>
      <div class="card"><h3 style="margin-bottom:8px">PPR points by week</h3><div class="chart-box"><canvas id="pchart"></canvas></div></div>
    </div>
    ${adv ? `<div class="card"><h3 style="margin-bottom:10px">Advanced metrics</h3><div class="stats-grid">${adv}</div></div>` : ""}
    <div class="card"><h3 style="margin-bottom:6px">Rest-of-season outlook</h3><p>${esc(p.ros?.outlook || "")}</p>
      <div class="grid2"><div class="table-wrap"><table><thead><tr><th>Wk</th><th>Opp</th><th class="num">Proj</th><th class="num">Range</th><th>Matchup</th></tr></thead><tbody>${sched}</tbody></table></div>
      <div class="table-wrap"><table><thead><tr><th>Wk</th><th>Opp</th><th class="num">PPR</th>${logCols.map((k) => `<th class="num">${logHead[k]}</th>`).join("")}</tr></thead><tbody>${log}</tbody></table></div></div></div>`;
  showModal();
  drawChart(p);
}
function drawChart(p) {
  if (chart) { chart.destroy(); chart = null; }
  const cv = $("#pchart");
  if (!cv || !window.Chart) return;
  const played = p.log || [];
  const fut = (p.proj || []).filter((x) => !x.bye && !x.played);
  const labels = [...played.map((g) => `W${g.week}`), ...fut.map((x) => `W${x.week}`)];
  chart = new Chart(cv, {
    type: "bar",
    data: { labels, datasets: [
      { label: "Actual", data: [...played.map((g) => g.ppr), ...fut.map(() => null)], backgroundColor: "#c6f432", borderRadius: 4 },
      { label: "Projected", data: [...played.map(() => null), ...fut.map((x) => x.proj)], backgroundColor: "rgba(90,169,255,.55)", borderRadius: 4 },
    ] },
    options: { maintainAspectRatio: false, plugins: { legend: { labels: { color: "#8a9bb0" } } },
      scales: { x: { stacked: true, ticks: { color: "#8a9bb0" }, grid: { display: false } }, y: { ticks: { color: "#8a9bb0" }, grid: { color: "#243041" } } } },
  });
}
function showModal() { $("#modal").classList.remove("hidden"); document.body.style.overflow = "hidden"; }
function hideModal() { $("#modal").classList.add("hidden"); document.body.style.overflow = ""; if (chart) { chart.destroy(); chart = null; } }

// ------------------------------------------------------------ boot
const RENDER = { teams: renderTeams, players: renderPlayers, defenses: renderDefenses, blitz: renderBlitz, model: renderModel };
let activeTab = "teams";
function renderActive() { RENDER[activeTab](); }

async function load() {
  const get = (u) => fetch(u, { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(u); return r.json(); });
  try {
    const [meta, players, defs, teams, leagues] = await Promise.all([
      get("data/meta.json"), get("data/players.json"), get("data/defenses.json"), get("data/teams.json"), get("config/leagues.json"),
    ]);
    Object.assign(S, { meta, players, defs, teams, leagues: leagues.leagues || [] });
    if (meta.has_backtest) S.backtest = await get("data/backtest.json").catch(() => []);
  } catch (e) {
    $("#updated").textContent = "Could not load data. Run the data pipeline first.";
    console.error(e);
    return;
  }
  buildIndex();
  S.week = store.get("ss_week") && store.get("ss_week") >= S.meta.target_week ? store.get("ss_week") : S.meta.target_week;
  const sel = $("#week");
  for (let w = S.meta.target_week; w <= S.meta.last_week; w++) sel.insertAdjacentHTML("beforeend", `<option value="${w}" ${w === S.week ? "selected" : ""}>Week ${w}</option>`);
  sel.onchange = () => { S.week = +sel.value; store.set("ss_week", S.week); renderActive(); };
  const d = new Date(S.meta.generated.endsWith("Z") || S.meta.generated.includes("+") ? S.meta.generated : S.meta.generated + "Z");
  $("#updated").textContent = `${S.meta.season} season · stats through week ${S.meta.through_week} · updated ${d.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}`;
  $("#tabs").querySelectorAll("button").forEach((b) => b.onclick = () => {
    $("#tabs").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
    activeTab = b.dataset.tab; renderActive();
  });
  $("#modal-close").onclick = hideModal;
  $("#modal").onclick = (e) => { if (e.target.id === "modal") hideModal(); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideModal(); });
  renderActive();
}
load();
