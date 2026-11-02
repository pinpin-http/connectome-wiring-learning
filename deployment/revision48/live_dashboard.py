#!/usr/bin/env python3
"""REVISION48 live page for the owner (French, Paris time): artifacts/monitor/revision48.html, rewritten every 30 s.

    python3 deployment/revision48/live_dashboard.py [--once]

Served by the smb-dashboard service (artifacts/monitor over Caddy, token required) at /revision48.html.
Reads only counts and times: campaign_status.json, activity.json, the running job's log.txt (decision counter),
the benchmarks, backup.json and nvidia-smi. Never a score (the owner's rule: no reading before the declared one)."""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/revision48"
OUT = ROOT / "artifacts/monitor/revision48.html"
PARIS = ZoneInfo("Europe/Paris")
ORDER = ("tuning", "reference", "tuned", "supervised", "lognorm", "autapse")
LABEL = {"tuning": "Réglage (9 essais / graphe)", "reference": "Référence", "tuned": "Réglé (sélection scellée)",
         "supervised": "Supervisé", "lognorm": "Poids logarithmiques", "autapse": "Autapses conservées"}
T0 = datetime(2026, 10, 24, 18, 22, 14, tzinfo=timezone.utc)
END = T0 + timedelta(hours=48)
LAST_START = END - timedelta(hours=4)
RATE = 0.2594
HORIZON = {"tuning": 1500}


def load(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def parse(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fr(d: datetime | None, day: bool = True) -> str:
    if d is None:
        return "—"
    d = d.astimezone(PARIS)
    days = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
    return (f"{days[d.weekday()]} {d:%d/%m} " if day else "") + f"{d:%H:%M}"


def dur(seconds: float) -> str:
    s = int(max(0, seconds))
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d}" if h else f"{m} min"


def gpu() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, mu, mt, t, p = [x.strip() for x in q.strip().split(",")]
        return {"util": float(u), "mem": float(mu) / 1024, "mem_total": float(mt) / 1024, "temp": float(t), "power": float(p)}
    except Exception:
        return {}


def runner_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "scripts/revision48/campaign.py"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def job_dir(key: str) -> Path:
    parts = key.split(":")
    phase = parts[0]
    if phase == "tuning":
        graph, gm = ":".join(parts[1:-1]), parts[-1]
        return ART / "runs/tuning" / gm / graph.replace(":", "")
    return ART / "runs/main" / phase / ":".join(parts[1:]).replace(":", "")


def decision(key: str) -> int | None:
    try:
        lines = (job_dir(key) / "log.txt").read_text().splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        for tok in line.split():
            if tok.startswith("k="):
                try:
                    return int(tok[2:])
                except ValueError:
                    pass
    return 0


def bench() -> dict:
    b = (load(ART / "checks/benchmark.json") or {}).get("5_benchmark", {})
    lb = load(ART / "checks/benchmark_lognorm.json") or {}
    main = b.get("projected_main_run_s", 1009)
    return {"tuning": b.get("projected_tuning_batch_s", 317), "reference": main, "tuned": main, "supervised": main,
            "lognorm": lb.get("projected_main_run_s", 1685), "autapse": main}


def projection(st: dict, act: dict, now: datetime) -> dict:
    """Remaining wall time from the observed mean minutes per phase (benchmark scaled by the observed/benchmark ratio
    where a phase has no finished job yet) plus the observed idle time between jobs."""
    b = bench()
    log = st.get("done_log", [])
    obs: dict[str, list[float]] = {}
    for x in log:
        obs.setdefault(x["job"].split(":")[0], []).append(x["minutes"] * 60)
    mean = {p: sum(v) / len(v) for p, v in obs.items() if v and p != "tuned"}
    ratios = [mean[p] / b[p] for p in mean if b.get(p)]
    ratio = max(ratios[-1:] + [0]) if ratios else 1.0
    main_ratio = (mean["reference"] / b["reference"]) if "reference" in mean else None
    per = {}
    for p in ORDER:
        if p in mean:
            per[p] = mean[p]
        elif p == "tuning":
            per[p] = b[p] * ratio
        else:
            per[p] = b[p] * (main_ratio if main_ratio else 1.0)
    gaps = []
    ends = [parse(x["utc"]) for x in log]
    starts = []
    by = act.get("by_phase", {})
    remaining = 0.0
    for p in ORDER:
        left = by.get(p, {}).get("total", 0) - by.get(p, {}).get("done", 0)
        remaining += left * per[p]
    run = st.get("running", {})
    for k, v in run.items():
        since = parse(v.get("since"))
        if since:
            remaining -= min((now - since).total_seconds(), per[k.split(":")[0]] * 0.95)
    n_left = sum(by.get(p, {}).get("total", 0) - by.get(p, {}).get("done", 0) for p in ORDER)
    # idle between jobs: median over the last 10 boundaries (next start = its end - its minutes), 35 s if unknown
    # only since the current runner start (the runner restarted 2026-10-25T05:47Z to remove the idle time)
    since = parse(st.get("started_utc"))
    cur = [x for x in log if since and parse(x["utc"]) and parse(x["utc"]) > since]
    for a_, b_ in zip(cur[-11:-1], cur[-10:]):
        e, s = parse(a_["utc"]), parse(b_["utc"])
        if e and s:
            gaps.append(max(0.0, (s - e).total_seconds() - b_["minutes"] * 60))
    gap = sorted(gaps)[len(gaps) // 2] if gaps else 35.0
    remaining += n_left * gap
    return {"per": per, "end": now + timedelta(seconds=max(0, remaining)), "remaining_s": remaining, "bench": b}


def svg_progress(st: dict, total: int, launch: datetime | None, proj_end: datetime, now: datetime) -> str:
    log = st.get("done_log", [])
    if not launch:
        return ""
    t1 = max(proj_end, now) + timedelta(minutes=30)
    t1 = max(t1, LAST_START + timedelta(hours=1))
    W, H, L, B = 680, 220, 44, 30
    span = (t1 - launch).total_seconds()

    def x(t):
        return L + (W - L - 10) * (t - launch).total_seconds() / span

    def y(n):
        return H - B - (H - B - 12) * n / total
    pts = [(x(launch), y(0))]
    for i, e in enumerate(log, 1):
        t = parse(e["utc"])
        pts.append((x(t), pts[-1][1]))
        pts.append((x(t), y(i)))
    pts.append((x(now), pts[-1][1]))
    path = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    proj = f"{x(now):.1f},{y(len(log)):.1f} {x(proj_end):.1f},{y(total):.1f}"
    ticks = []
    t = launch.astimezone(PARIS).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while t < t1:
        if t.hour % 6 == 0:
            xx = x(t)
            lab = fr(t) if t.hour == 0 else f"{t:%H}h"
            ticks.append(f'<line x1="{xx:.1f}" y1="12" x2="{xx:.1f}" y2="{H - B}" class="grid"/>'
                         f'<text x="{xx:.1f}" y="{H - B + 16}" class="ax" text-anchor="middle">{lab}</text>')
        t += timedelta(hours=1)
    yt = "".join(f'<line x1="{L}" y1="{y(n):.1f}" x2="{W - 10}" y2="{y(n):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{y(n) + 4:.1f}" class="ax" text-anchor="end">{n}</text>'
                 for n in (0, 40, 80, 120, 167) if n <= total)
    ls, le = x(LAST_START), x(END) if END < t1 else None
    marks = (f'<line x1="{ls:.1f}" y1="12" x2="{ls:.1f}" y2="{H - B}" class="limit"/>'
             f'<text x="{ls - 4:.1f}" y="24" class="ax lim" text-anchor="end">dernier lancement</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Travaux terminés au cours du temps">{yt}{"".join(ticks)}'
            f'{marks}<polyline points="{proj}" class="proj"/><polyline points="{path}" class="done"/>'
            f'<circle cx="{x(now):.1f}" cy="{y(len(log)):.1f}" r="4" class="dot"/></svg>')


def page() -> str:
    now = datetime.now(timezone.utc)
    st = load(ART / "campaign_status.json", {}) or {}
    act = load(ART / "activity.json", {}) or {}
    bk = load(ROOT / "artifacts/monitor/backup.json", {}) or {}
    g = gpu()
    alive = runner_alive()
    by = act.get("by_phase", {})
    total = act.get("total", 167)
    done = act.get("done", 0)
    launch = parse((st.get("restarts") or [{}])[0].get("previous_start_utc") or st.get("started_utc"))  # first launch
    pj = projection(st, act, now)
    proj_end = pj["end"]
    margin = (LAST_START - proj_end).total_seconds()
    finished = bool(st.get("finished_utc"))
    if finished:
        state, cls = "Campagne terminée", "ok"
    elif not alive:
        state, cls = "Runner arrêté", "bad"
    elif st.get("failed"):
        state, cls = "En cours — échec(s) à examiner", "warn"
    else:
        state, cls = "En cours", "ok"
    pct = 100 * done / total if total else 0

    rows = []
    for p in ORDER:
        d = by.get(p, {})
        t, dn, rn, fl = d.get("total", 0), d.get("done", 0), d.get("running", 0), d.get("failed", 0)
        w = 100 * dn / t if t else 0
        wr = 100 * rn / t if t else 0
        per = pj["per"][p]
        mark = "obs." if any(x["job"].startswith(p + ":") for x in st.get("done_log", [])) else "banc"
        rows.append(
            f'<tr><td>{LABEL[p]}</td><td class="bar"><div class="track"><div class="fill" style="width:{w:.1f}%"></div>'
            f'<div class="run" style="left:{w:.1f}%;width:{max(wr, 0):.1f}%"></div></div></td>'
            f'<td class="num">{dn}/{t}</td><td class="num">{"" if not fl else f"<span class=bad>{fl} échec</span>"}</td>'
            f'<td class="num muted">{dur(per)} / run ({mark})</td></tr>')

    run_html = []
    for k, v in (st.get("running") or {}).items():
        since = parse(v.get("since"))
        p = k.split(":")[0]
        hz = HORIZON.get(p, 6000)
        kk = decision(k)
        el = (now - since).total_seconds() if since else 0
        exp = pj["per"][p]
        frac = min(1.0, el / exp) if exp else 0
        name = ":".join(k.split(":")[1:])
        run_html.append(
            f'<div class="job"><div class="jt"><b>{html.escape(LABEL[p])}</b> · <code>{html.escape(name)}</code>'
            f'{" · essai " + str(v.get("try")) if v.get("try", 1) > 1 else ""}</div>'
            f'<div class="track big"><div class="fill" style="width:{100 * frac:.1f}%"></div></div>'
            f'<div class="muted">depuis {fr(since, False)} ({dur(el)}) · dernier examen à la décision '
            f'{kk if kk is not None else "—"} / {hz} · fin estimée {fr(since + timedelta(seconds=exp), False) if since else "—"}</div></div>')
    if not run_html:
        run_html = ['<div class="muted">aucun travail en cours (transition, sauvegarde ou fin)</div>']

    recent = list(reversed(st.get("done_log", [])[-10:]))
    rec = "".join(f'<tr><td>{fr(parse(x["utc"]), False)}</td><td>{html.escape(LABEL[x["job"].split(":")[0]])}</td>'
                  f'<td><code>{html.escape(":".join(x["job"].split(":")[1:]))}</code></td><td class="num">{x["minutes"]:.1f} min</td></tr>'
                  for x in recent) or '<tr><td colspan=4 class="muted">aucun</td></tr>'
    inc = []
    for f in st.get("failed", []):
        inc.append(f'<li class="bad">échec : <code>{html.escape(f["job"])}</code> (code {f.get("rc")}) à {fr(parse(f["utc"]), False)}</li>')
    for f in st.get("retried", []):
        inc.append(f'<li class="warn">relancé : <code>{html.escape(f["job"])}</code> (code {f.get("rc")}) à {fr(parse(f["utc"]), False)}</li>')
    for f in st.get("skipped_modules", []):
        inc.append(f'<li class="warn">module non démarré faute de temps : {html.escape(LABEL.get(f["phase"], f["phase"]))} ({f["runs"]} runs)</li>')
    if st.get("selection_sealed"):
        s = st["selection_sealed"]
        inc.append(f'<li>sélection du réglage scellée à {fr(parse(s["utc"]), False)} (code {s["exit"]})</li>')
    for p, b in (st.get("phase_backups") or {}).items():
        inc.append(f'<li>sauvegarde complète après « {html.escape(LABEL.get(p, p))} » : code {b["exit"]} à {fr(parse(b["utc"]), False)}</li>')
    inc_html = "".join(inc) or '<li class="muted">aucun</li>'

    cost_now = RATE * (now - T0).total_seconds() / 3600
    bk_ok = bk.get("ok")
    bk_last = parse(bk.get("last_success_utc"))
    gpu_html = (f'{g["util"]:.0f} % · {g["mem"]:.1f}/{g["mem_total"]:.0f} Go · {g["temp"]:.0f} °C · {g["power"]:.0f} W'
                if g else "indisponible")
    runner_end = parse(act.get("projected_end_utc"))
    jph = act.get("jobs_per_hour")
    return f"""<title>REVISION48 en direct</title>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30">
<style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--accent:#2f6fb5;--run:#9cc0e8;--ok:#2e7d4f;--warn:#a86400;--bad:#b3261e;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--ink:#ecebe6;--muted:#9b9a94;--line:#33322f;--accent:#6ea8e6;--run:#33557a;--ok:#5cc08a;--warn:#e0a24a;--bad:#ef6f66;color-scheme:dark}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:16px}}h1{{font-size:20px;margin:4px 0 2px}}h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px;font-weight:600}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}}
.muted{{color:var(--muted);font-size:13px}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}.bad{{color:var(--bad)}}
.pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:13px;font-weight:600;border:1px solid currentColor}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.kpi b{{display:block;font-size:22px;font-variant-numeric:tabular-nums}}.kpi span{{color:var(--muted);font-size:12.5px}}
.track{{position:relative;height:10px;background:var(--line);border-radius:6px;overflow:hidden}}.track.big{{height:14px;margin:8px 0 6px}}
.fill{{position:absolute;left:0;top:0;bottom:0;background:var(--accent)}}.run{{position:absolute;top:0;bottom:0;background:var(--run)}}
table{{width:100%;border-collapse:collapse}}td{{padding:6px 4px;border-top:1px solid var(--line);vertical-align:middle}}tr:first-child td{{border-top:0}}
td.bar{{width:38%}}.num{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}code{{font-size:12.5px}}
.job{{padding:4px 0}}ul{{margin:0;padding-left:18px}}li{{margin:3px 0}}
svg{{width:100%;height:auto;display:block}}svg .grid{{stroke:var(--line);stroke-width:1}}svg .ax{{fill:var(--muted);font-size:11px}}
svg .done{{fill:none;stroke:var(--accent);stroke-width:2.5}}svg .proj{{fill:none;stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 5;opacity:.6}}
svg .dot{{fill:var(--accent)}}svg .limit{{stroke:var(--warn);stroke-width:1.5;stroke-dasharray:3 3}}svg .lim{{fill:var(--warn)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:6px}}.sw{{display:inline-block;width:18px;height:3px;background:var(--accent);vertical-align:middle;margin-right:6px}}.sw.d{{background:none;border-top:2px dashed var(--accent);opacity:.7}}.sw.l{{background:none;border-top:2px dashed var(--warn)}}
</style>
<main>
<h1>REVISION48 — campagne en direct</h1>
<div class="muted">mis à jour {fr(now)} (heure de Paris) · rafraîchi toutes les 30 s · comptes et durées uniquement, aucun score</div>
<div class="card"><span class="pill {cls}">{state}</span>
<div class="kpis" style="margin-top:12px">
<div class="kpi"><b>{done} / {total}</b><span>travaux terminés ({pct:.1f} %)</span></div>
<div class="kpi"><b>{fr(proj_end)}</b><span>fin projetée (débit réel)</span></div>
<div class="kpi"><b class="{'ok' if margin >= 0 else 'bad'}">{'+' if margin >= 0 else '−'}{dur(abs(margin))}</b><span>marge avant le dernier lancement ({fr(LAST_START, True)})</span></div>
<div class="kpi"><b>{cost_now:.2f} $</b><span>coût depuis T0 · 48 h ≈ {48 * RATE:.2f} $</span></div>
</div>
<div class="track big" style="margin-top:14px"><div class="fill" style="width:{pct:.1f}%"></div></div>
<div class="muted">lancée {fr(launch)} · fin de location {fr(END)} · projection du runner (banc) {fr(runner_end)}{f" · {jph:.1f} travaux/h" if jph else ""}</div>
</div>
<div class="card"><h2>En cours</h2>{"".join(run_html)}</div>
<div class="card"><h2>Phases (ordre d'exécution)</h2><table>{"".join(rows)}</table>
<div class="muted" style="margin-top:6px">durée par run : moyenne observée (obs.) ou banc d'essai ajusté au débit observé (banc)</div></div>
<div class="card"><h2>Avancement</h2>{svg_progress(st, total, launch, proj_end, now)}
<div class="legend muted"><span><i class="sw"></i>terminés</span><span><i class="sw d"></i>projection</span><span><i class="sw l"></i>dernier lancement possible</span></div></div>
<div class="card"><h2>Machine et sauvegardes</h2><table>
<tr><td>GPU (RTX 3090)</td><td class="num">{gpu_html}</td></tr>
<tr><td>Runner</td><td class="num {'ok' if alive else 'bad'}">{'actif' if alive else 'arrêté'}</td></tr>
<tr><td>Dernière sauvegarde Drive complète</td><td class="num {'ok' if bk_ok else 'bad'}">{fr(bk_last)} {'' if bk_ok else '(dernier essai en échec)'}</td></tr>
</table></div>
<div class="card"><h2>Événements</h2><ul>{inc_html}</ul></div>
<div class="card"><h2>Derniers terminés</h2><table>{rec}</table></div>
</main>
"""


def main() -> int:
    once = "--once" in sys.argv
    while True:
        try:
            tmp = OUT.with_name(OUT.name + ".part")
            tmp.write_text(page())
            os.replace(tmp, OUT)
        except Exception as e:                               # never die: the page is a convenience
            print(f"{datetime.now(timezone.utc):%FT%TZ} live_dashboard error: {e!r}", flush=True)
        if once:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
