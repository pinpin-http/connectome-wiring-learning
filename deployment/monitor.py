#!/usr/bin/env python3
"""Persistent monitoring of the runs on the GPU server (deployment/README.md); stdlib only.

Writes under artifacts/monitor/: state.json (per-run state, progress, metrics, resources, cost),
resources.jsonl (resource samples), events.jsonl (run state changes) and dashboard.html (static page).
Runs are read from artifacts/deploy_runs/<run_id>; cost comes from `vastai show instance`.

    python3 deployment/monitor.py --once          # one update
    python3 deployment/monitor.py --loop 60       # the service
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment"))
import flydeploy as fd

OUT = ROOT / "artifacts" / "monitor"
RUNS = fd.RUNS_DIR
KEEP_SAMPLES = 7 * 24 * 60          # a week at one sample per minute
_cost_cache = {"t": 0.0, "data": None}
_cpu_prev = {"t": None, "usage": None}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- resources
def cgroup_read(name: str) -> str | None:
    p = Path("/sys/fs/cgroup") / name
    try:
        return p.read_text().strip()
    except OSError:
        return None


def resources() -> dict:
    r = {"utc": now()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=False)
        if out.returncode == 0 and out.stdout.strip():
            u, mu, mt, temp, pw = (x.strip() for x in out.stdout.strip().splitlines()[0].split(","))
            r.update(gpu_util=float(u), vram_used_gb=round(float(mu) / 1024, 2), vram_total_gb=round(float(mt) / 1024, 1),
                     gpu_temp_c=float(temp), gpu_power_w=float(pw) if pw not in ("[N/A]", "") else None)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    cpu_max = cgroup_read("cpu.max")                       # cgroup v2; v1 fallbacks below
    quota = None
    if cpu_max and cpu_max.split()[0] != "max":
        q, period = cpu_max.split()
        quota = int(q) / int(period)
    elif cgroup_read("cpu/cpu.cfs_quota_us") not in (None, "-1"):
        quota = int(cgroup_read("cpu/cpu.cfs_quota_us")) / int(cgroup_read("cpu/cpu.cfs_period_us") or 100000)
    stat = cgroup_read("cpu.stat")
    usage = None
    if stat and "usage_usec" in stat:
        usage = int(dict(line.split() for line in stat.splitlines())["usage_usec"])
    elif cgroup_read("cpuacct/cpuacct.usage"):
        usage = int(cgroup_read("cpuacct/cpuacct.usage")) // 1000          # ns -> us
    if usage is not None:
        t = time.time()
        if _cpu_prev["t"] is not None and t > _cpu_prev["t"]:
            cores = (usage - _cpu_prev["usage"]) / 1e6 / (t - _cpu_prev["t"])
            r["cpu_cores_used"] = round(cores, 2)
            if quota:
                r["cpu_pct_of_quota"] = round(100 * cores / quota, 1)
        _cpu_prev.update(t=t, usage=usage)
    r["cpu_quota_cores"] = round(quota, 2) if quota else os.cpu_count()
    mem_cur = cgroup_read("memory.current") or cgroup_read("memory/memory.usage_in_bytes")
    mem_max = cgroup_read("memory.max") or cgroup_read("memory/memory.limit_in_bytes")
    if mem_max and mem_max.isdigit() and int(mem_max) > 1 << 60:          # v1 "unlimited"
        mem_max = "max"
    if mem_cur:
        r["ram_used_gb"] = round(int(mem_cur) / 1e9, 2)
    if mem_max and mem_max != "max":
        r["ram_limit_gb"] = round(int(mem_max) / 1e9, 1)
    du = shutil.disk_usage("/workspace" if Path("/workspace").exists() else ROOT)
    r.update(disk_used_gb=round(du.used / 1e9, 1), disk_total_gb=round(du.total / 1e9, 1))
    r["load1"] = os.getloadavg()[0]
    return r


# --------------------------------------------------------------------------- cost
def cost() -> dict:
    """The instance's billing record (Vast API, instance-scoped key set by Vast in the
    container); refreshed every 30 min. Never stores or prints the key."""
    if time.time() - _cost_cache["t"] > 1800 or _cost_cache["data"] is None:
        data = None
        key, cid = os.environ.get("CONTAINER_API_KEY"), os.environ.get("CONTAINER_ID")
        if key and cid and shutil.which("vastai"):
            r = subprocess.run(["vastai", "show", "instance", cid, "--api-key", key, "--raw"],
                               capture_output=True, text=True, timeout=60, check=False)
            try:
                j = json.loads(r.stdout)
                data = {k: j.get(k) for k in ("dph_total", "dph_base", "storage_total_cost", "inet_up_cost",
                                              "inet_down_cost", "start_date", "cur_state", "is_bid", "geolocation")}
            except (json.JSONDecodeError, TypeError):
                data = None
        _cost_cache.update(t=time.time(), data=data)
    d = _cost_cache["data"]
    if not d or not d.get("dph_total") or not d.get("start_date"):
        return {"available": False, "note": "billing record unavailable (vastai CLI or CONTAINER_API_KEY missing)"}
    hours = (time.time() - float(d["start_date"])) / 3600
    return {"available": True, "rate_per_hour": round(d["dph_total"], 4),
            "compute_per_hour": round(d["dph_base"], 4), "storage_per_hour": round(d["storage_total_cost"], 4),
            "bandwidth_per_gb": {"up": round(d["inet_up_cost"], 4), "down": round(d["inet_down_cost"], 4)},
            "rented_since_utc": datetime.fromtimestamp(float(d["start_date"]), UTC).isoformat(timespec="minutes"),
            "hours_so_far": round(hours, 2), "cost_so_far": round(hours * d["dph_total"], 2),
            "per_day": round(24 * d["dph_total"], 2), "per_week": round(168 * d["dph_total"], 2),
            "stopped_storage_per_day": round(24 * d["storage_total_cost"], 2),
            "includes": "GPU/CPU/RAM rental and disk storage at the current rate, as if constant since the start",
            "unknown": "bandwidth actually used (billed per GB, not measured here), any rate change, Vast fees "
                       "outside the instance record; the account balance is in the Vast console"}


# --------------------------------------------------------------------------- runs
AUC_LINE = re.compile(r"'frames': (\d+), 'auc': \[([^\]]*)\]")
START_LINE = re.compile(r"^(Level\S+) delay\s+(\d+) \((\d+)s\): (.*)$")
BRAIN = re.compile(r"(\S+?)=(\d+)/(\d+)")


def parse_log(lines: list[str], exp: str) -> dict:
    m = {}
    if exp.startswith("cs-"):
        pts = []
        for ln in lines:
            a = AUC_LINE.search(ln)
            if a:
                pts.append({"frames": int(a.group(1)), "auc": [float(x) for x in a.group(2).split(",")]})
        m["auc_curve"] = pts
        sc = next((ln for ln in lines if ln.startswith("expert stream:")), None)
        if sc:
            m["stream_check"] = sc
    elif exp.startswith("probe-play"):
        per = {}
        secs = []
        for ln in lines:
            s = START_LINE.match(ln)
            if not s:
                continue
            secs.append(int(s.group(3)))
            for name, x, rnd in BRAIN.findall(s.group(4)):
                per.setdefault(name, []).append(int(x) - int(rnd))
        m["starts_done"] = len(secs)
        m["seconds_per_start"] = round(sum(secs[-8:]) / max(1, len(secs[-8:])), 1) if secs else None
        m["mean_x_minus_random_so_far"] = {k: round(sum(v) / len(v), 1) for k, v in per.items()}
    return m


def run_entry(rd: Path) -> dict:
    man = json.loads((rd / "manifest.json").read_text())
    st = fd.run_status(rd)
    exp, params = man["experiment"], man.get("params", {})
    state = st.get("state")
    if state == "running" and not fd.alive(st.get("supervisor_pid")):
        state = "DEAD"
    lines = (rd / "log.txt").read_text(errors="replace").splitlines() if (rd / "log.txt").exists() else []
    metrics = parse_log(lines, exp)
    prog = fd.run_progress(rd)
    elapsed = st.get("elapsed_s") or 0
    eta, speed, frac = None, None, None
    if exp.startswith("cs-") and metrics.get("auc_curve"):
        done = metrics["auc_curve"][-1]["frames"]
        frac = done / 4000
        if done and elapsed:
            speed = f"{done / (elapsed / 3600):.0f} frames/h"
            if state == "running":
                eta = round((4000 - done) / (done / elapsed) / 3600, 2)
    elif exp.startswith("probe-play"):
        done = metrics.get("starts_done", 0)
        frac = done / 64
        sps = metrics.get("seconds_per_start")
        if sps:
            speed = f"{3600 / sps:.1f} starts/h"
            if state == "running":
                eta = round((64 - done) * sps / 3600, 2)
    elif exp in ("stageD", "stageD-eval") and prog.get("decisions_k"):
        c, t = prog["decisions_k"].split("/")
        if c != "None" and t != "None":
            frac = int(c) / max(1, int(t))
        eta = prog.get("eta_h") if state == "running" else None
    errors = [ln for ln in lines if re.search(r"Traceback|Error|error:|Killed|out of memory", ln)][-5:]
    ckpts = []
    for o in man.get("outputs", []) + [f"artifacts/runs/checkpoints/common_stream_state_{params.get('gtag', '')}.pt"]:
        p = fd.ROOT / o
        if p.exists() and p.is_file():
            ckpts.append({"path": o, "size_mb": round(p.stat().st_size / 1e6, 1),
                          "age_min": round((time.time() - p.stat().st_mtime) / 60, 1)})
    resume = ("deployment/flydeploy run " + exp + "".join(f" --param {k}={v}" for k, v in params.items()
                                                         if k in ("graph", "suite", "rule", "stream", "ckpt") and v)
              + (" --resume" if fd.experiments().get(exp, {}).get("resumable") else ""))
    return {"run_id": rd.name, "experiment": exp, "objective": man.get("declared_in"),
            "graph": params.get("graph"), "params": {k: v for k, v in params.items()
                                                    if k not in ("gtag", "runname", "evalname", "x", "rule_suffix")},
            "seeds": man.get("seeds"), "commit": (man.get("code") or {}).get("commit", "")[:12],
            "state": state, "exit_code": st.get("exit_code"), "started_utc": st.get("started_utc"),
            "ended_utc": st.get("ended_utc"), "elapsed_h": round(elapsed / 3600, 2), "progress_fraction": frac,
            "speed": speed, "eta_h": eta, "progress": prog, "metrics": metrics, "log_tail": lines[-6:],
            "errors": errors, "checkpoints": ckpts, "results": [str(rd.relative_to(fd.ROOT))] + man.get("outputs", []),
            "resume_command": resume, "stop_command": f"deployment/flydeploy stop {rd.name}"}


def reports() -> list[dict]:
    """Declared readings already written by the probe reports (classes per brain)."""
    out = []
    for p in sorted((fd.ROOT / "artifacts/runs/frozen_probe").glob("report*.json")):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        brains = {k: b.get("class") for k, b in r.get("brains", {}).items() if b.get("class")}
        out.append({"file": str(p.relative_to(fd.ROOT)), "utc": r.get("utc"),
                    "scenario_c_available": r.get("scenario_c_available"), "classes": brains})
    return out


# --------------------------------------------------------------------------- page
def spark(values: list[float], w: int = 220, h: int = 40, lo: float | None = None, hi: float | None = None,
          color: str = "var(--s1)") -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - 2 - (v - lo) / rng * (h - 4):.1f}" for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" role="img">'
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/></svg>')


def auc_chart(pts: list[dict], w: int = 300, h: int = 90) -> str:
    """The readouts' AUC curves on one 0-1 scale, x = frames."""
    fmax = max(p["frames"] for p in pts) or 1
    y = lambda v: h - 4 - v * (h - 8)
    lines = [f'<line x1="0" x2="{w}" y1="{y(0.5):.1f}" y2="{y(0.5):.1f}" stroke="var(--line)" stroke-dasharray="4 3"/>']
    for i in range(len(pts[0]["auc"])):
        xy = " ".join(f"{p['frames'] / fmax * w:.1f},{y(p['auc'][i]):.1f}" for p in pts)
        lines.append(f'<polyline fill="none" stroke="var(--s{i + 1})" stroke-width="2" points="{xy}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" role="img">'
            + "".join(lines) + "</svg>")


def revision48_status() -> dict | None:
    """REVISION48 runner status: counts and times only (the runner never writes a score)."""
    p = fd.ROOT / "artifacts/revision48/campaign_status.json"
    try:
        s = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    keep = ("n_done", "n_total", "running", "retried", "failed", "skipped_deadline", "deadline_utc", "updated_utc",
            "started_utc", "finished_utc", "stopped_utc", "selection_sealed", "phase_backups")
    return {k: s.get(k) for k in keep if k in s}


def page(state: dict, history: list[dict]) -> str:
    e = html.escape
    res, c = state["resources"], state["cost"]
    cards = []
    for r in state["runs"]:
        badge = {"running": "run", "finished": "ok", "stopped": "warn", "failed": "bad", "DEAD": "bad"}.get(r["state"], "warn")
        bar = (f'<div class="bar"><span style="width:{100 * r["progress_fraction"]:.0f}%"></span></div>'
               if r["progress_fraction"] is not None else "")
        curve = ""
        if r["metrics"].get("auc_curve"):
            pts = r["metrics"]["auc_curve"]
            curve = (f"<p class=k>test AUC per scoring point, 3 readouts (0 to 1; 0.5 dashed), frames 0 → "
                     f"{pts[-1]['frames']}: last {', '.join(f'{x:.3f}' for x in pts[-1]['auc'])}</p>"
                     + auc_chart(pts))
        play = ""
        if r["metrics"].get("mean_x_minus_random_so_far"):
            play = "<p class=k>furthest x minus rate-matched random, mean so far (px)</p><table>" + "".join(
                f"<tr><td>{e(k)}</td><td>{v:+.1f}</td></tr>" for k, v in r["metrics"]["mean_x_minus_random_so_far"].items()
            ) + "</table>"
        errs = "".join(f"<pre class=err>{e(x)}</pre>" for x in r["errors"])
        ck = "".join(f"<li><code>{e(x['path'])}</code> {x['size_mb']} MB, {x['age_min']} min ago</li>" for x in r["checkpoints"])
        cards.append(f"""<section class=card><h3>{e(r['experiment'])} <small>{e(str(r['graph'] or ''))}</small>
<b class="badge {badge}">{e(str(r['state']))}</b></h3>
<p class=k>{e(str(r['objective'] or ''))}</p>{bar}
<p>{r['elapsed_h']} h elapsed{(' · ' + e(r['speed'])) if r['speed'] else ''}{(' · ETA ' + str(r['eta_h']) + ' h') if r['eta_h'] is not None else ''}
{(' · exit ' + str(r['exit_code'])) if r['exit_code'] is not None else ''}</p>
{curve}{play}
<details><summary>details</summary><p class=k>run {e(r['run_id'])} · commit {e(r['commit'])} · seeds: {e(str(r['seeds']))}</p>
<p class=k>params {e(json.dumps(r['params']))}</p>
<pre>{e(chr(10).join(r['log_tail']))}</pre>{errs}<ul>{ck}</ul>
<p class=k>resume: <code>{e(r['resume_command'])}</code><br>stop: <code>{e(r['stop_command'])}</code></p>
<p class=k>results: {e(', '.join(r['results']))}</p></details></section>""")
    h = history[-360:]
    rep = "".join(f"<li><code>{e(x['file'])}</code> C available: {x['scenario_c_available']} — "
                  + e(", ".join(f"{k}: {v}" for k, v in x["classes"].items())) + "</li>" for x in state["reports"])
    cost_html = (f"<p><b>${c['cost_so_far']}</b> so far ({c['hours_so_far']} h since {e(c['rented_since_utc'])} UTC) · "
                 f"${c['rate_per_hour']}/h = ${c['per_day']}/day, ${c['per_week']}/week · stopped (storage only): "
                 f"${c['stopped_storage_per_day']}/day · bandwidth ${c['bandwidth_per_gb']['down']}/GB</p>"
                 f"<p class=k>includes: {e(c['includes'])}. Unknown: {e(c['unknown'])}.</p>"
                 if c.get("available") else f"<p class=k>{e(c.get('note', ''))}</p>")
    rv = state.get("revision48") or {}
    revision_html = (e(f"{rv.get('n_done')}/{rv.get('n_total')} jobs done; running: {rv.get('running')}; "
                       f"retried {len(rv.get('retried', []))}, failed {len(rv.get('failed', []))}, "
                       f"skipped at the deadline {len(rv.get('skipped_deadline', []))}; deadline {rv.get('deadline_utc')}; "
                       f"updated {rv.get('updated_utc')}") if rv else "not started")
    b = state.get("backup", {})
    backup_html = (f"last attempt {e(str(b.get('utc', '-')))} UTC: {e(str(b.get('note', '')))}; last success "
                   f"{e(str(b.get('last_success_utc') or 'never'))} → {e(str(b.get('destination', '-')))}")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><meta http-equiv=refresh content=60>
<title>SMB GPU runs</title><style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--card:#fff;--line:#e4e3df;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ok:#1baf7a;--bad:#d03b3b;--warn:#c98500}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--card:#232322;--line:#3a3a38;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}}}
body{{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:10px 0}}
h1{{font-size:18px;margin:4px 0}} h3{{font-size:16px;margin:0 0 4px}} small,.k{{color:var(--mut);font-size:13px;margin:2px 0}}
.badge{{float:right;font-size:12px;padding:2px 8px;border-radius:9px;color:#fff;background:var(--warn)}}
.badge.ok{{background:var(--ok)}} .badge.bad{{background:var(--bad)}} .badge.run{{background:var(--s1)}}
.bar{{height:8px;background:var(--line);border-radius:4px;overflow:hidden}} .bar span{{display:block;height:100%;background:var(--s1)}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--bg);padding:6px;border-radius:6px}} .err{{color:var(--bad)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}} code{{font-size:12px;word-break:break-all}}
table{{font-size:13px}} td{{padding:1px 8px 1px 0}}
</style></head><body>
<h1>SMB_Flywire_Speedrun — GPU server</h1><p class=k>updated {e(state['utc'])} UTC · refreshes every 60 s · automatic record (not Claude's messages)</p>
<section class=card><h3>Resources</h3><div class=grid>
<div>GPU {res.get('gpu_util', '?')}%<br>{spark([x.get('gpu_util') for x in h], lo=0, hi=100)}</div>
<div>VRAM {res.get('vram_used_gb', '?')}/{res.get('vram_total_gb', '?')} GB<br>{spark([x.get('vram_used_gb') for x in h], lo=0, hi=res.get('vram_total_gb'))}</div>
<div>CPU {res.get('cpu_cores_used', '?')}/{res.get('cpu_quota_cores', '?')} cores<br>{spark([x.get('cpu_cores_used') for x in h], lo=0, hi=res.get('cpu_quota_cores'))}</div>
<div>RAM {res.get('ram_used_gb', '?')}/{res.get('ram_limit_gb', '?')} GB<br>{spark([x.get('ram_used_gb') for x in h], lo=0)}</div>
<div>Disk {res.get('disk_used_gb', '?')}/{res.get('disk_total_gb', '?')} GB</div></div><p class=k>last 6 h</p></section>
<section class=card><h3>Cost (estimate)</h3>{cost_html}</section>
<section class=card><h3>Off-instance backup</h3><p>{backup_html}</p></section>
<section class=card><h3>REVISION48 campaign</h3><p>{revision_html}</p></section>
<h2 style="font-size:16px">Runs</h2>{''.join(cards) or '<p>no runs yet</p>'}
<section class=card><h3>Declared readings (reports)</h3><ul>{rep or '<li>none yet</li>'}</ul></section>
</body></html>"""


# --------------------------------------------------------------------------- loop
def update() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    res = resources()
    with open(OUT / "resources.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    hist_lines = (OUT / "resources.jsonl").read_text().splitlines()
    if len(hist_lines) > KEEP_SAMPLES * 1.2:
        atomic_write(OUT / "resources.jsonl", "\n".join(hist_lines[-KEEP_SAMPLES:]) + "\n")
        hist_lines = hist_lines[-KEEP_SAMPLES:]
    history = [json.loads(x) for x in hist_lines[-400:] if x.strip()]
    runs = []
    for rd in sorted(RUNS.glob("*"), key=lambda d: d.stat().st_mtime, reverse=True)[:30]:
        if (rd / "manifest.json").exists() and (rd / "status.json").exists():
            try:
                runs.append(run_entry(rd))
            except (OSError, json.JSONDecodeError, KeyError) as ex:
                runs.append({"run_id": rd.name, "state": "unreadable", "error": repr(ex)})
    prev = {}
    if (OUT / "state.json").exists():
        try:
            prev = {r["run_id"]: r.get("state") for r in json.loads((OUT / "state.json").read_text()).get("runs", [])}
        except (OSError, json.JSONDecodeError):
            prev = {}
    with open(OUT / "events.jsonl", "a") as f:
        for r in runs:
            if prev.get(r["run_id"]) != r.get("state"):
                f.write(json.dumps({"utc": now(), "run_id": r["run_id"], "from": prev.get(r["run_id"]),
                                    "to": r.get("state")}) + "\n")
    try:
        backup = json.loads((OUT / "backup.json").read_text())
    except (OSError, json.JSONDecodeError):
        backup = {"ok": False, "note": "no backup attempt recorded yet (smb-backup service)"}
    state = {"utc": now(), "resources": res, "cost": cost(), "backup": backup, "runs": runs, "reports": reports(),
             "revision48": revision48_status(),
             "paths": {"runs": "artifacts/deploy_runs/", "results": "artifacts/runs/", "monitor": "artifacts/monitor/",
                       "revision48": "artifacts/revision48/"}}
    atomic_write(OUT / "state.json", json.dumps(state, indent=1, default=str))
    atomic_write(OUT / "dashboard.html", page(state, history))
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=float, default=0)
    a = ap.parse_args()
    if a.loop:
        while True:
            try:
                update()
            except Exception as ex:  # noqa: BLE001  (the service must keep running)
                print(f"{now()} monitor error: {ex!r}", flush=True)
            time.sleep(a.loop)
    st = update()
    print(f"{len(st['runs'])} runs; GPU {st['resources'].get('gpu_util')}%; -> {OUT}")


if __name__ == "__main__":
    main()
