#!/usr/bin/env python
"""From frame discrimination to frozen play (docs/RL_STUDY.md §12).

Why does a discrimination learnt in the weights (frozen AUC 0.88-0.96 on held-out frames) give
a poor frozen policy in play? Every brain is frozen. Subcommands: merge (build
`checkpoints/frozen_probe_<graph>.pt` from the common-stream brains plus scripted references),
fixed (AUC of P(jump) on fresh scripted sequences), play (4 levels x 8 starts, resumable),
report (declared classification C / B / A).

    .venv/bin/python scripts/gates/frozen_play_probe.py merge --graph connectome
    .venv/bin/python scripts/gates/frozen_play_probe.py fixed --graph connectome
    .venv/bin/python scripts/gates/frozen_play_probe.py play --graph connectome [--stop 17:00]
    .venv/bin/python scripts/gates/frozen_play_probe.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gates"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from mario_generalisation_eval import JUMP as JUMPS
from mario_generalisation_eval import Evaluator
from oracle_criterion_check import JUMP, RUN, criterion, play
from threshold_play_check import DELAYS, LEVELS, PRESS, ThresholdEvaluator, pjump

from flymario.paths import RUNS

CK = RUNS / "checkpoints"
OUT = RUNS / "frozen_probe"
DATE = "20261009"
REFS = ("ref_expert", "ref_alwaysrun")
# output-name letter of each training stream (the scripted stream has none); dagger /
# daggerctl brains get their own files
SX = {None: "", "expert": "x", "dagger": "d", "daggerctl": "c", "all": "a", "expertb1": "y"}


def hold_k(rule):
    """"hold6" -> 6; "hold2f05" -> 2 (§22: the f part is the calibration's false-positive target)."""
    return int(rule[4:].split("f")[0])


def rule_fpr(rule):
    return float("0." + rule.split("f")[1]) if rule and "f" in rule[4:] else None


def tag(graph):
    return graph.replace(":", "")


# --------------------------------------------------------------------------- merge
def merge(graph, stream=None):
    """stream "expert" (§14): the expert-stream brains (csx_ro1-3), no permuted brains."""
    import torch
    parts = [CK / f"common_stream{'_' + stream if stream else ''}_{tag(graph)}.pt"]
    if stream == "all":                       # §22: scripted, expert, DAgger and control brains together
        parts = [CK / f"common_stream{s_}_{tag(graph)}.pt" for s_ in ("", "_expert", "_dagger", "_daggerctl")]
    if graph == "connectome" and stream in (None, "all"):
        parts.append(CK / "common_stream_perm_connectome.pt")
    learners, m = [], []
    for p in parts:
        st = torch.load(p, map_location="cpu", weights_only=False)
        learners += st["learners"]
        m.append(st["brain"]["m"])
    m = torch.cat(m, 0)
    if graph == "connectome":                 # scripted: the weights of their slots are never used
        for name in REFS:
            learners.append({"name": name, "readout_seed": learners[0]["readout_seed"], "noise_seed": 8300,
                             "view": "pixels"})
        m = torch.cat([m, m[:1].repeat(len(REFS), 1)], 0)
    out = CK / f"frozen_probe{'_' + stream if stream else ''}_{tag(graph)}.pt"
    torch.save({"learners": learners, "brain": {"m": m}, "graph": graph, "parts": [str(p) for p in parts]}, out)
    print(out, [l["name"] for l in learners])


# --------------------------------------------------------------------------- step 1
def sequences(seed=20261090, max_dec=400, every=2):
    """Fresh scripted sequences on the play levels: one per (level, delay), from the level
    start after `delay` NOOP decisions; frames every 2nd decision while Mario is in play."""
    from flymario.env.retro_env import ACTION_NAMES, EnvConfig, MarioEnv
    from flymario.env.schematic import takeoff_label_v2
    noop = ACTION_NAMES.index("NOOP")
    seqs = []
    for li, level in enumerate(LEVELS):
        for d in DELAYS:
            rng = np.random.default_rng(seed + 100 * li + d)
            env = MarioEnv(EnvConfig(state=level))
            prev, _ = env.reset()
            for _ in range(d):
                prev, *_ = env.step(noop)
            fr, lab = [], []
            for n in range(max_dec):
                obs, _r, term, trunc, _i = env.step(PRESS if rng.random() < 0.25 else RUN)
                ram = env.env.unwrapped.get_ram()
                if n % every == 0 and ram[0xB5] == 1:
                    fr.append((prev, obs))
                    lab.append(bool(takeoff_label_v2(ram)))
                prev = obs
                if term or trunc:
                    break
            env.close()
            seqs.append({"level": level, "delay": d, "frames": fr, "label": np.array(lab)})
    return seqs


def sequences_v2(suite, max_dec=400, every=2):
    """§13.1: one fresh scripted sequence per harness-v2 start of the suite (the start's
    prefix, then run / jump with p .25), a frame every 2nd decision while Mario is in play."""
    from play_harness_v2 import STARTS, SUITES

    from flymario.env.retro_env import EnvConfig, MarioEnv
    from flymario.env.schematic import takeoff_label_v2
    seqs = []
    for si, st in enumerate(json.loads(STARTS.read_text())["starts"]):
        if st["level"] not in SUITES[suite]:
            continue
        rng = np.random.default_rng(20261090 + si)
        env = MarioEnv(EnvConfig(state=st["level"]))
        prev, _ = env.reset()
        for a in st["prefix"]:
            prev, *_ = env.step(a)
        fr, lab = [], []
        for n in range(max_dec):
            obs, _r, term, trunc, _i = env.step(PRESS if rng.random() < 0.25 else RUN)
            ram = env.env.unwrapped.get_ram()
            if n % every == 0 and ram[0xB5] == 1:
                fr.append((prev, obs))
                lab.append(bool(takeoff_label_v2(ram)))
            prev = obs
            if term or trunc:
                break
        env.close()
        seqs.append({"level": st["level"], "delay": st["start"], "frames": fr, "label": np.array(lab)})
    return seqs


def fixed(graph, suite=None, stream=None):
    from sklearn.metrics import roc_auc_score

    from flymario.env.vision import RetinotopicVision
    from flymario.training.sweep import sweep_once
    ev = Evaluator(str(CK / f"frozen_probe{'_' + stream if stream else ''}_{tag(graph)}.pt"), 1, 600, graph=graph)
    torch = ev.torch
    seqs = sequences_v2(suite) if suite else sequences()
    levels = list(dict.fromkeys(q["level"] for q in seqs))
    pre = f"fixedv2{SX[stream]}-{suite}" if suite else "fixed"
    vis = RetinotopicVision(ev.g)
    P, L, S = [], [], []
    for si, s in enumerate(seqs):
        for (prev, obs), lab in zip(s["frames"], s["label"]):
            vis.reset()
            vis.encode(prev)
            cur = torch.as_tensor(np.repeat(vis.encode(obs).astype(np.float32)[:, None], ev.B, 1), device=ev.brain.dev)
            c, _ = sweep_once(ev.brain, ev.params, cur, ev.K, 0.5, False)
            P.append(pjump(torch.einsum("bam,mb->ba", ev.R, c).cpu().numpy()))
            L.append(lab)
            S.append(si)
    P, L, S = np.array(P), np.array(L), np.array(S)
    lvl = np.array([seqs[k]["level"] for k in S])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{pre}-{tag(graph)}-{DATE}.npz", p=P, label=L, seq=S, level=lvl,
                        names=np.array(ev.names))
    rng = np.random.default_rng(0)
    res = {}
    for b, name in enumerate(ev.names):
        if name in REFS:
            continue
        auc = float(roc_auc_score(L, P[:, b]))
        boots = []
        for _ in range(2000):
            pick = rng.integers(0, len(seqs), len(seqs))
            idx = np.concatenate([np.flatnonzero(S == k) for k in pick])
            if L[idx].any() and not L[idx].all():
                boots.append(roc_auc_score(L[idx], P[idx, b]))
        per = {}                       # post hoc (labelled): per-level ranking and score level
        for lv in levels:
            k = lvl == lv
            per[lv] = {"n": int(k.sum()), "label_rate": round(float(L[k].mean()), 3),
                       "auc": round(float(roc_auc_score(L[k], P[k, b])), 4) if L[k].any() and not L[k].all() else None,
                       "p_on": round(float(P[k & L, b].mean()), 4) if (k & L).any() else None,
                       "p_off": round(float(P[k & ~L, b].mean()), 4) if (k & ~L).any() else None}
        res[name] = {"auc": round(auc, 4), "ci95": [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)],
                     "boots": [round(float(x), 4) for x in boots], "per_level": per}
        print(name, res[name]["auc"], res[name]["ci95"], flush=True)
    rep = {"what": "RL_STUDY §12 step 1: S_fixed on fresh scripted development sequences", "graph": graph,
           "n_frames": len(L), "n_sequences": len(seqs), "label_rate": round(float(L.mean()), 3),
           "brains": res, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (OUT / f"{pre}-{tag(graph)}-{DATE}.json").write_text(json.dumps(rep))
    ev.pool.close()


# --------------------------------------------------------------------------- steps 2-3
class ProbeEvaluator(ThresholdEvaluator):
    """The §20 threshold policy with a per-decision log; two scripted reference slots.
    rule "hold6" (§13.2): a trigger (want to jump while eligible) presses RIGHT+A+B for 6
    decisions, then the non-jump action for one; triggers and eligible decisions logged."""

    rule = "threshold"

    def view_frame(self, b, obs, info):
        from flymario.env.schematic import takeoff_label_v2
        if not hasattr(self, "_lab"):
            self._lab, self._x = np.zeros(self.B, bool), np.zeros(self.B)
        self._lab[b] = bool(takeoff_label_v2(info["ram"]))      # label of the frame the brain sees
        self._x[b] = info.get("mario_x", 0.0)
        return super().view_frame(b, obs, info)

    def scores(self, frames):
        sc = Evaluator.scores(self, frames)
        pj = pjump(sc)
        jump = pj >= self.tau
        logged = pj.copy()
        out = sc.copy()
        out[:, JUMPS] = -np.inf
        for b, name in enumerate(self.names):
            if name == "ref_expert":
                jump[b], logged[b] = self._lab[b], float(self._lab[b])
            elif name == "ref_alwaysrun":                   # no score of its own (its slot's weights are a copy)
                jump[b], logged[b] = False, 0.0
        trig = np.zeros(self.B, bool)
        elig = np.zeros(self.B, bool)
        if self.rule.startswith("hold"):
            want = jump.copy()
            for b in range(self.B):
                if self._hold[b] > 0:
                    self._hold[b] -= 1
                    self._release[b] = self._hold[b] == 0
                    jump[b] = True
                elif self._release[b]:
                    self._release[b] = False
                    jump[b] = False
                else:
                    elig[b] = True
                    trig[b] = want[b]
                    jump[b] = want[b]
                    if want[b]:
                        self._hold[b] = hold_k(self.rule) - 1     # hold6 -> 5 (§13.2); holdk (§20)
        out[jump] = -np.inf
        out[jump, PRESS] = 0.0
        for b, name in enumerate(self.names):
            if name in REFS and not jump[b]:
                out[b] = -np.inf
                out[b, RUN] = 0.0
        self._trace.append(jump.copy())
        self._log.append((logged, jump.copy(), out.argmax(1).tolist(), self._lab.copy(), self._x.copy(),
                          trig, elig))
        return out

    def episode(self, suite, level, delay):
        self._log = []
        self._hold, self._release = np.zeros(self.B, int), np.zeros(self.B, bool)
        rec = super().episode(suite, level, delay)
        for b, e in enumerate(rec):
            n = e["decisions"]
            e["log"] = {"p": [round(float(x[0][b]), 4) for x in self._log[:n]],
                        "jump": [bool(x[1][b]) for x in self._log[:n]],
                        "act": [x[2][b] for x in self._log[:n]],
                        "label": [bool(x[3][b]) for x in self._log[:n]],
                        "x": [float(x[4][b]) for x in self._log[:n]]}
            if self.rule.startswith("hold"):
                e["log"]["trig"] = [bool(x[5][b]) for x in self._log[:n]]
                e["log"]["elig"] = [bool(x[6][b]) for x in self._log[:n]]
                e["trigger_rate"] = sum(e["log"]["trig"]) / max(1, sum(e["log"]["elig"]))
        return rec


def _random_jumper(args):
    level, d, q, seed = args
    g = np.random.default_rng(seed)
    return play(level, d, lambda ram: JUMP if g.random() < q else RUN)["best_x"]


def _random_jumper_v2(args):
    level, prefix, q, seed = args
    from play_harness_v2 import _random
    return _random((level, prefix, q, seed))


def _random_hold6_v2(args):
    from play_harness_v2 import _random_hold6
    return _random_hold6(args)


def play_cmd(graph, k, stop, smoke=None, suite=None, rule=None, stream=None):
    """smoke: a checkpoint path; plays one start into a separate file (code check only).
    suite (§13.1): the harness-v2 starts of that suite instead of the 4 x 8 delays."""
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    if suite:
        from play_harness_v2 import STARTS, SUITES
        allst = json.loads(STARTS.read_text())["starts"]
        starts = [(st["level"], st["start"], st["prefix"], si) for si, st in enumerate(allst)
                  if st["level"] in SUITES[suite]]
        path = OUT / f"playv2{SX[stream]}-{suite}{'-' + rule if rule else ''}-{tag(graph)}{'-smoke' if smoke else ''}-{DATE}.json"
    else:
        starts = [(level, d, d, None) for level in LEVELS for d in DELAYS]
        path = OUT / f"play-{tag(graph)}{'-smoke' if smoke else ''}-{DATE}.json"
    done = json.loads(path.read_text()) if path.exists() else None
    ev = ProbeEvaluator(smoke or str(CK / f"frozen_probe{'_' + stream if stream else ''}_{tag(graph)}.pt"), 1, 600, graph=graph)
    if rule:
        ev.rule = rule
    if done:                                   # resume: same thresholds, skip finished starts
        ev.tau = np.array([done["calibration"]["brains"][n]["tau"] for n in ev.names])
        cal, rows = done["calibration"], done["rows"]
        print("resumed:", len(rows) // ev.B, "starts done", flush=True)
    else:
        cal, rows = ev.calibrate(rule_fpr(rule)), []
        print("calibration", json.dumps(cal), flush=True)
    finished = {(r["level"], r["delay"]) for r in rows}
    stop_at = None
    if stop:
        h, mi = (int(x) for x in stop.split(":"))
        stop_at = datetime.now(UTC).astimezone().replace(hour=h, minute=mi, second=0, microsecond=0).timestamp()
    pool = ProcessPoolExecutor(max_workers=6)
    complete = True
    for level, d, reset_arg, si in starts:
        if (level, d) in finished:
            continue
        if stop_at and time.time() > stop_at:
            complete = False
            break
        t0 = time.time()
        rec = ev.episode("probe", level, reset_arg)
        jobs, slots = [], []
        for b, e in enumerate(rec):
            q = e["trigger_rate"] if rule and rule.startswith("hold") else e["jumps"] / max(1, e["decisions"])
            for j in range(k):
                if suite:                  # §13 seeds, draw set 1, same draws for every brain
                    jobs.append((level, reset_arg, q, 20261200 + 50 * si + j)
                                + ((hold_k(rule),) if rule and rule.startswith("hold") else ()))
                else:
                    jobs.append((level, d, q, 20261091 + 1000 * LEVELS.index(level) + 10 * d + 100 * b + 100000 * j))
                slots.append(b)
        xs = list(pool.map(_random_hold6_v2 if rule and rule.startswith("hold") else _random_jumper_v2 if suite else _random_jumper, jobs))
        for b, e in enumerate(rec):
            rx = [x for x, s in zip(xs, slots) if s == b]
            rows.append({"learner": e["learner"], "level": level, "delay": d, "best_x": e["best_x"],
                         "end": e["end"], "decisions": e["decisions"],
                         "jump_rate": round(e["jumps"] / max(1, e["decisions"]), 3),
                         "jump_after_jump": e.get("jump_after_jump"), "random_best_x": float(np.mean(rx)),
                         "trigger_rate": e.get("trigger_rate"),
                         "log": e["log"]})
        print(f"{level} delay {d:2d} ({time.time() - t0:.0f}s): " + " ".join(
            f"{r['learner'][:6]}{r['learner'][-1]}={r['best_x']:.0f}/{r['random_best_x']:.0f}"
            for r in rows[-len(rec):]), flush=True)
        rep = {"what": "RL_STUDY §12 steps 2-3: frozen threshold play, per-decision log", "graph": graph,
               "checkpoint": str(CK / f"frozen_probe_{tag(graph)}.pt"), "calibration": cal, "k": k,
               "levels": LEVELS, "delays": DELAYS, "harness": "v2" if suite else "v1", "suite": suite, "rule": rule or "threshold",
               "rows": rows, "complete": False,
               "utc": datetime.now(UTC).isoformat(timespec="seconds")}
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(rep))
        tmp.replace(path)
        if smoke:
            complete = False
            break
    rep = json.loads(path.read_text())
    rep["complete"] = complete and len(rep["rows"]) == ev.B * len(starts)
    path.write_text(json.dumps(rep))
    print("complete" if rep["complete"] else "stopped (partial; relaunch to resume)", flush=True)
    pool.shutdown()
    ev.pool.close()
    del torch


# --------------------------------------------------------------------------- report
def _auc(lab, p):
    from sklearn.metrics import roc_auc_score
    lab, p = np.asarray(lab, bool), np.asarray(p, float)
    return float(roc_auc_score(lab, p)) if lab.any() and not lab.all() else None


def own_score(rows, rng):
    """S_own: AUC of logged P(jump) for the logged label over the agent's own decisions,
    bootstrap over episodes."""
    lab = [np.asarray(r["log"]["label"], bool) for r in rows]
    p = [np.asarray(r["log"]["p"], float) for r in rows]
    a = _auc(np.concatenate(lab), np.concatenate(p))
    boots = []
    for _ in range(2000):
        pick = rng.integers(0, len(rows), len(rows))
        v = _auc(np.concatenate([lab[i] for i in pick]), np.concatenate([p[i] for i in pick]))
        if v is not None:
            boots.append(v)
    return a, boots


def timing(rows):
    """Per label onset: lag of the first jump decision inside the label-on window (missed if
    none); press lengths; jump decisions while the label is off; failure context."""
    lags, missed, onsets, presses, false_jumps, off_dec = [], 0, 0, [], 0, 0
    fail = {"missed": 0, "pressed": 0, "unlabelled": 0}
    for r in rows:
        lab, jmp = np.asarray(r["log"]["label"], bool), np.asarray(r["log"]["jump"], bool)
        t = 0
        while t < len(lab):
            if lab[t] and (t == 0 or not lab[t - 1]):
                onsets += 1
                u = t
                while u < len(lab) and lab[u]:
                    u += 1
                hit = np.flatnonzero(jmp[t:u])
                if len(hit):
                    lags.append(int(hit[0]))
                else:
                    missed += 1
                t = u
            else:
                t += 1
        run = 0
        for j in list(jmp) + [False]:
            if j:
                run += 1
            elif run:
                presses.append(run)
                run = 0
        false_jumps += int((jmp & ~lab).sum())
        off_dec += int((~lab).sum())
        if r["end"] in ("death", "stall"):
            w = slice(max(0, len(lab) - 20), len(lab))
            if not lab[w].any():
                fail["unlabelled"] += 1
            elif (jmp[w] & lab[w]).any():
                fail["pressed"] += 1
            else:
                fail["missed"] += 1
    return {"onsets": onsets, "missed": missed, "miss_rate": round(missed / onsets, 3) if onsets else None,
            "median_lag": float(np.median(lags)) if lags else None,
            "press_length_median": float(np.median(presses)) if presses else None,
            "press_length_share_1": round(float(np.mean(np.array(presses) == 1)), 3) if presses else None,
            "jump_while_label_off": round(false_jumps / max(1, off_dec), 3), "failures": fail}


def play_test(rows):
    """The harness test, per level, and whether a pass is carried by one level: the mean
    difference over the other three levels is <= 0 when that level is removed."""
    c = criterion([r["best_x"] for r in rows], [r["random_best_x"] for r in rows])
    per = {}
    for lv in dict.fromkeys(r["level"] for r in rows):
        rr = [r for r in rows if r["level"] == lv]
        if rr:
            per[lv] = {"wins": int(sum(r["best_x"] > r["random_best_x"] for r in rr)),
                       "losses": int(sum(r["best_x"] < r["random_best_x"] for r in rr)),
                       "mean_diff": round(float(np.mean([r["best_x"] - r["random_best_x"] for r in rr])), 1)}
    carried = any(np.mean([r["best_x"] - r["random_best_x"] for r in rows if r["level"] != lv] or [0]) <= 0
                  for lv in per)
    return {**c, "per_level": per, "carried_by_one_level": bool(carried), "counts": bool(c["pass"] and not carried)}


def report(suite=None, rule=None, stream=None):
    rng = np.random.default_rng(1)
    x = SX[stream]
    pp, fp_ = (f"playv2{x}-{suite}{'-' + rule if rule else ''}", f"fixedv2{x}-{suite}") if suite else ("play", "fixed")
    n_parts = (5 if rule else 4) if suite else 3
    graphs = [p.stem.split("-")[n_parts - 2] for p in sorted(OUT.glob(f"{pp}-*-{DATE}.json"))
              if len(p.stem.split("-")) == n_parts]                 # not the -smoke files
    brains, notes = {}, []
    for gt in graphs:
        pl = json.loads((OUT / f"{pp}-{gt}-{DATE}.json").read_text())
        fx_path = OUT / f"{fp_}-{gt}-{DATE}.json"
        fx = json.loads(fx_path.read_text())["brains"] if fx_path.exists() else {}
        if not pl["complete"]:
            notes.append(f"{gt}: play PARTIAL ({len(pl['rows'])} rows)")
        for name in dict.fromkeys(r["learner"] for r in pl["rows"]):
            rr = [r for r in pl["rows"] if r["learner"] == name]
            a, boots = own_score(rr, rng)
            brains[f"{gt}/{name}"] = {
                "graph": gt, "name": name, "starts": len(rr), "tau": pl["calibration"]["brains"][name]["tau"],
                "mean_jump_rate": round(float(np.mean([r["jump_rate"] for r in rr])), 3),
                "s_own": round(a, 4) if a is not None else None,
                "s_own_ci95": [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)] if boots else None,
                "_s_own_boots": boots,
                "s_fixed": fx.get(name, {}).get("auc"), "s_fixed_ci95": fx.get(name, {}).get("ci95"),
                "_s_fixed_boots": fx.get(name, {}).get("boots", []),
                "play": play_test(rr), "timing": timing(rr)}
    neg = [b for b in brains.values() if b["name"].startswith(("untrained", "csperm"))]
    max_neg_own = max((b["s_own"] for b in neg if b["s_own"] is not None), default=None)
    max_neg_fixed = max((b["s_fixed"] for b in neg if b["s_fixed"] is not None), default=None)

    def informative(v, ci, ref):
        return v is not None and ci is not None and ref is not None and v >= 0.70 and ci[0] > ref

    refs = {}
    exp = brains.get("connectome/ref_expert")
    if exp:
        al = [r for r in json.loads((OUT / f"{pp}-connectome-{DATE}.json").read_text())["rows"]
              if r["learner"] == "ref_expert"]
        lab = np.concatenate([r["log"]["label"] for r in al])
        jmp = np.concatenate([r["log"]["jump"] for r in al])
        act = np.concatenate([r["log"]["act"] for r in al])
        refs["alignment"] = {"s_own_from_actions": _auc(lab, jmp.astype(float)),
                             "press_iff_label": bool(np.all((act == PRESS) == lab)),
                             "pass": _auc(lab, jmp.astype(float)) == 1.0 and bool(np.all((act == PRESS) == lab))}
        if suite:                              # §13.1: the harness-v2 validation run of the same policy
            val = json.loads((RUNS / f"harness_v2/validation{'-' + rule if rule else ''}-20261009.json").read_text())["expert"]
            from play_harness_v2 import STARTS
            allst = json.loads(STARTS.read_text())["starts"]
            vx = {(st["level"], st["start"]): v["best_x"] for st, v in zip(allst, val)}
            same = [abs(r["best_x"] - vx[(r["level"], r["delay"])]) < 1 for r in al]
            refs["positive_expert_passes"] = exp["play"]["counts"]
        else:
            ora = json.loads((RUNS / "oracle_criterion_check-takeoff2-k5-20261004.json").read_text())["episodes"]
            same = [abs(r["best_x"] - o["ram_jumper"]["best_x"]) < 1 for r in al for o in ora
                    if (o["level"], o["delay"]) == (r["level"], r["delay"])]
            refs["positive_expert_passes"] = exp["play"]["pass"]
        refs["expert_reproduces_oracle_path"] = {"starts_equal": int(sum(same)), "of": len(same)}
    for b in neg + [x for x in brains.values() if x["name"] == "ref_alwaysrun"]:
        refs.setdefault("negatives", {})[f"{b['graph']}/{b['name']}"] = {
            "play_pass": b["play"]["pass"], "s_own": b["s_own"], "s_fixed": b["s_fixed"]}
    neg_ok = all(not v["play_pass"] for v in refs.get("negatives", {}).values())
    # declared (§12): if the scripted expert fails the play test, play is descriptive and
    # scenario C cannot be declared
    c_available = bool(refs.get("positive_expert_passes")) and neg_ok
    for key, b in brains.items():
        if b["name"] in REFS or b["name"].startswith(("untrained", "csperm")):
            continue                                   # trained brains: cs_ro<i> (scripted stream), csx_ro<i> (expert)
        inf_own = informative(b["s_own"], b["s_own_ci95"], max_neg_own)
        inf_fix = informative(b["s_fixed"], b["s_fixed_ci95"], max_neg_fixed)
        drop, drop_ci = None, None
        if b["s_own"] is not None and b["s_fixed"] is not None and b["_s_own_boots"] and b["_s_fixed_boots"]:
            n = min(len(b["_s_own_boots"]), len(b["_s_fixed_boots"]))
            d = np.array(b["_s_fixed_boots"][:n]) - np.array(b["_s_own_boots"][:n])
            drop, drop_ci = round(b["s_fixed"] - b["s_own"], 4), [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]
        if c_available and b["play"]["counts"]:
            cls = "C useful frozen controller"
        elif inf_fix and drop is not None and drop >= 0.10 and drop_ci[0] > 0:
            cls = "B observation shift"
        elif inf_own:
            cls = "A decision / timing / interface"
        else:
            cls = "unclassified"
        b.update({"informative_own": inf_own, "informative_fixed": inf_fix, "drop_fixed_minus_own": drop,
                  "drop_ci95": drop_ci, "class": cls})
    for b in brains.values():
        b.pop("_s_own_boots", None)
        b.pop("_s_fixed_boots", None)
    rep = {"what": "RL_STUDY §12 report", "graphs": graphs, "notes": notes, "references": refs,
           "scenario_c_available": c_available,
           "negatives_all_fail_play": neg_ok, "max_negative_s_own": max_neg_own,
           "max_negative_s_fixed": max_neg_fixed, "brains": brains,
           "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (OUT / f"report{'v2' + x + '-' + suite if suite else ''}{'-' + rule if rule else ''}-{DATE}.json").write_text(
        json.dumps(rep, indent=1, default=float))
    print(json.dumps({"notes": notes, "references": refs, "negatives_all_fail_play": neg_ok,
                      "max_negative_s_own": max_neg_own, "max_negative_s_fixed": max_neg_fixed}, indent=1))
    for key, b in brains.items():
        print(f"{key:28s} S_fixed {b['s_fixed']} S_own {b['s_own']} {b['s_own_ci95']} jump {b['mean_jump_rate']} "
              f"play p {b['play']['p_greater']} diff {b['play']['mean_diff']} counts {b['play']['counts']} "
              f"miss {b['timing']['miss_rate']} lag {b['timing']['median_lag']} press1 {b['timing']['press_length_share_1']} "
              f"fail {b['timing']['failures']} -> {b.get('class', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("merge", "fixed", "play", "report"))
    ap.add_argument("--graph", default="connectome")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stop", default=None)
    ap.add_argument("--smoke", default=None, help="checkpoint for a one-start code check (separate output)")
    ap.add_argument("--suite", default=None, choices=("training", "unseen"), help="§13.1: harness-v2 starts")
    ap.add_argument("--rule", default=None, choices=("hold6", "hold2", "hold3", "hold4", "hold2f05"), help="§13.2: hold once triggered; §20: hold length k")
    ap.add_argument("--stream", default=None, choices=("expert", "dagger", "daggerctl", "all", "expertb1"), help="§14: the expert-stream brains; G-DAgger-v1 brains")
    a = ap.parse_args()
    {"merge": lambda: merge(a.graph, a.stream), "fixed": lambda: fixed(a.graph, a.suite, a.stream),
     "play": lambda: play_cmd(a.graph, a.k, a.stop, a.smoke, a.suite, a.rule, a.stream),
     "report": lambda: report(a.suite, a.rule, a.stream)}[a.cmd]()


if __name__ == "__main__":
    main()
