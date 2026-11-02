#!/usr/bin/env python
"""Gate F: does the interface leak task information, or bypass the network? (docs/PROTOCOL.md §10.4, §11.2)

Probes: A the HUD crop is applied on the current's path; B can a decoder recover score / timer /
lives from `vision.encode(frame)` at matched position (with positive controls); C the sensory
interface has no fitted parameters; D every graph condition injects into the same neurons (§10.2);
E the §11.2 readout category is declared and, for `fixed`, enforced.

Needs a ROM. Read-only; writes one JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from flymario import config
from flymario.agent import AgentConfig, FlyAgent
from flymario.connectome import datasets, randomize
from flymario.env import vision as vision_mod
from flymario.env.retro_env import EnvConfig, MarioEnv
from flymario.paths import ROOT, RUNS

#: A decoder above this on a task variable is leakage, not coincidence.
LEAK_FLOOR = 0.40
#: The four categories §11.2 requires the study to choose between.
READOUT_CATEGORIES = ("fixed", "minimally_trainable", "fully_trainable",
                      "part_of_the_learning_intervention")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def collect(env: MarioEnv, agent: FlyAgent, steps: int, episodes: int = 6,
            seed: int = 0) -> dict:
    """Injected current and the RAM state that produced it, per decision.

    Several episodes with varied actions, so score and timer differ at equal position (in a
    single forward run every RAM variable is a function of the decision index).
    """
    rng = np.random.default_rng(seed)
    cur, ram, frames = [], [], []
    per_ep = max(1, steps // episodes)
    for ep in range(episodes):
        obs, info = env.reset()
        agent.vision.reset()
        # A different action mixture per episode: some runs collect coins and
        # break blocks, others stall, so the score diverges from the position.
        w = rng.dirichlet(np.ones(env.n_actions))
        for _ in range(per_ep):
            cur.append(agent.vision.encode(obs).copy())
            ram.append({k: info.get(k) for k in
                        ("score", "time", "lives", "coins", "mario_x",
                         "player_y", "world", "area")})
            frames.append(obs.copy())
            a = 2 if rng.random() < 0.5 else int(rng.choice(env.n_actions, p=w))
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return {"current": np.stack(cur), "ram": ram, "frames": np.stack(frames)}


# --- probe A: the crop removes the rows that carry the HUD ----------------
def probe_hud_crop(agent: FlyAgent, frames: np.ndarray) -> dict:
    """Randomise the cropped band and check the injected current is unchanged: the crop
    is applied on the current's path, not merely declared."""
    rng = np.random.default_rng(0)
    agent.vision.reset()
    base = np.stack([agent.vision.encode(f) for f in frames])

    tampered = frames.copy()
    tampered[:, : vision_mod.HUD_ROWS] = rng.integers(
        0, 256, tampered[:, : vision_mod.HUD_ROWS].shape, dtype=np.uint8)
    agent.vision.reset()
    after = np.stack([agent.vision.encode(f) for f in tampered])

    # And the negative control: tampering below the crop MUST change it.
    below = frames.copy()
    below[:, vision_mod.HUD_ROWS:] = rng.integers(
        0, 256, below[:, vision_mod.HUD_ROWS:].shape, dtype=np.uint8)
    agent.vision.reset()
    after_below = np.stack([agent.vision.encode(f) for f in below])

    d_hud = float(np.abs(base - after).max())
    d_below = float(np.abs(base - after_below).max())
    return {
        "hud_rows": int(vision_mod.HUD_ROWS),
        "max_change_when_hud_randomised": d_hud,
        "max_change_when_below_randomised": d_below,
        "pass": bool(d_hud == 0.0 and d_below > 0.0),
        "note": "the second number is the control: if it is also 0 the probe is broken",
    }


# --- probe B: can a task variable be decoded, at matched position? --------
def probe_leakage(data: dict, agent: FlyAgent, n_bins: int = 8) -> dict:
    """Decode RAM variables from the injected current, within position bins.

    Position legitimately decodes (the level scrolls); within a narrow x-bin the scene is
    nearly fixed, so remaining decodability of score or timer is leakage.
    """
    injected = np.union1d(agent.vision.on_idx, agent.vision.off_idx)
    X = data["current"][:, injected]
    x = np.array([r["mario_x"] for r in data["ram"]], dtype=np.float64)

    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1.0
    binned = np.digitize(x, edges[1:-1])

    def score_within_bins(vals: np.ndarray) -> dict:
        accs, chances, used = [], [], 0
        for b in np.unique(binned):
            m = binned == b
            if m.sum() < 20:
                continue
            v = vals[m]
            if len(np.unique(v)) < 2:
                continue
            y = (v > np.median(v)).astype(int)
            counts = np.bincount(y, minlength=2)
            if counts.min() < 5:
                continue
            # Stratify. Plain KFold inside a narrow position bin can hand a
            # fold a single class -- the minority label is rare there by
            # construction -- and the fit then raises rather than scoring.
            n_splits = int(min(4, counts.min()))
            if n_splits < 2:
                continue
            Xs = StandardScaler().fit_transform(X[m])
            a = cross_val_score(LogisticRegression(C=1.0, max_iter=2000), Xs, y,
                                cv=StratifiedKFold(n_splits=n_splits, shuffle=True,
                                                   random_state=0),
                                scoring="accuracy")
            accs.append(float(a.mean()))
            chances.append(float(counts.max() / counts.sum()))
            used += 1
        if not accs:
            return {"skipped": "no position bin had enough variation"}
        excess = np.array(accs) - np.array(chances)
        return {"accuracy": float(np.mean(accs)),
                "majority_class_baseline": float(np.mean(chances)),
                "bins_used": used,
                "excess_over_baseline": float(excess.mean()),
                # One bin gives a point estimate with no error bar; say so
                # rather than letting the mean look like a measurement.
                "excess_sd_across_bins": (float(excess.std(ddof=1))
                                          if used > 1 else None),
                "excess_per_bin": [round(float(v), 3) for v in excess],
                "per_bin_accuracy": [round(float(v), 3) for v in accs]}

    out = {}
    for key in ("score", "time", "lives", "coins"):
        vals = np.array([r[key] for r in data["ram"]], dtype=np.float64)
        r = score_within_bins(vals)
        if "skipped" not in r:
            r["leaks"] = bool(r["excess_over_baseline"] > 0.15)
        out[key] = r

    # Positive control #1, within bins: `player_y` is visible and varies inside an x-band
    # (jumps). If it does not decode, the within-bin analysis has no power and "no excess"
    # on score or timer means nothing.
    y_vals = np.array([r["player_y"] for r in data["ram"]], dtype=np.float64)
    within_control = score_within_bins(y_vals)

    # Positive control #2, uncontrolled on purpose: position must decode over
    # the whole run. Weaker -- it is the confound the binning exists to remove.
    y = (x > np.median(x)).astype(int)
    Xs = StandardScaler().fit_transform(X)
    # Stratified and shuffled. The rows arrive in episode order and the label
    # is "past the median x", so a plain KFold hands whole folds a single class
    # and every fit raises.
    ctrl = cross_val_score(LogisticRegression(C=1.0, max_iter=2000), Xs, y,
                           cv=StratifiedKFold(4, shuffle=True, random_state=0),
                           scoring="accuracy")
    control = {"accuracy": float(ctrl.mean()), "chance": 0.5}

    # Power is claimed from the within-bin control when it ran, and only from
    # the global one as a fallback -- flagged, because it does not establish
    # within-bin power.
    if "skipped" in within_control:
        has_power = bool(control["accuracy"] > 0.65)
        power_basis = "global_only__within_bin_power_UNVERIFIED"
    else:
        has_power = bool(within_control["excess_over_baseline"] > 0.10)
        power_basis = "within_bin_player_y"

    leaks = [k for k, v in out.items() if v.get("leaks")]
    return {
        "per_variable": out,
        "positive_control_player_y_within_bins": within_control,
        "positive_control_mario_x_uncontrolled": control,
        "probe_has_power": has_power,
        "power_basis": power_basis,
        "variables_not_testable": sorted(k for k, v in out.items()
                                         if "skipped" in v),
        "position_bins": int(n_bins),
        "leaking_variables": leaks,
        "note": ("Scored within narrow mario_x bins. An uncontrolled version of "
                 "this probe returns 1.000 on every variable, including lives, "
                 "because the scrolling scene encodes the decision index and "
                 "every RAM variable is monotone in it. That is confounding by "
                 "time, not leakage."),
    }


# --- probe C: the sensory interface has no fitted parameters --------------
def probe_no_learned_parameters(agent: FlyAgent) -> dict:
    """`vision.py` states the rule (no fitted sensory front end); this asserts it."""
    v = agent.vision
    arrays = {k: a for k, a in vars(v).items() if isinstance(a, np.ndarray)}
    # Everything the interface holds must be either an index set or a
    # coordinate read off the graph -- never a value fitted to anything.
    derived = {"on_idx", "off_idx", "on_uv", "off_uv"}
    unexpected = sorted(set(arrays) - derived - {"_prev"})
    params = {f.name: getattr(v.p, f.name)
              for f in v.p.__dataclass_fields__.values()}
    return {
        "arrays_held": sorted(arrays),
        "unexpected_arrays": unexpected,
        "declared_parameters": params,
        "retinotopy_source": "graph.node_retino (measured ommatidial columns)",
        "pass": not unexpected,
        "note": ("gains are declared constants, not fitted values; the "
                 "retinotopic map is read from the connectome, not learned"),
    }


# --- probe D: every condition injects into the same neurons (§10.2) -------
def probe_shared_injection_sites(g, dataset: str, topologies, seed: int) -> dict:
    """Vision must be built from the anatomical graph, not the rewired one."""
    ref = None
    rows = {}
    for topo in topologies:
        cfg = AgentConfig(topology=topo, topology_seed=seed)
        a = FlyAgent(g, cfg)
        sites = np.union1d(a.vision.on_idx, a.vision.off_idx)
        rows[topo] = {"n_injected": int(len(sites)),
                      "eye": a.vision.coverage()["eye"],
                      "sha": hashlib.sha256(sites.tobytes()).hexdigest()[:16]}
        if ref is None:
            ref = rows[topo]["sha"]
    identical = all(r["sha"] == ref for r in rows.values())
    return {"per_topology": rows, "identical": bool(identical),
            "pass": bool(identical),
            "note": ("agent.py builds RetinotopicVision from `graph`, not "
                     "`self.wired`, so nulls inherit the same injection sites")}


# --- probe E: is the readout category declared? ---------------------------
def probe_readout_declaration() -> dict:
    """§11.2 requires the readout category frozen before confirmatory comparisons.

    Also finds the class implementing the declared category and, for `fixed`, proves its
    weights cannot be written.
    """
    declared = config.get("readout.category", None)
    from flymario import agent as agent_mod
    from flymario.policy import readout as readout_mod

    # Both modules: the readout categories are spread across them.
    by_category: dict[str, list[str]] = {}
    seen: set[int] = set()
    for mod in (agent_mod, readout_mod):
        for name, obj in vars(mod).items():
            if not inspect.isclass(obj) or id(obj) in seen:
                continue  # agent.py re-exports readout classes; count each once
            cat = getattr(obj, "CATEGORY", None)
            if isinstance(cat, str):
                seen.add(id(obj))
                by_category.setdefault(cat, []).append(
                    f"{obj.__module__}.{obj.__qualname__}")
    for v in by_category.values():
        v.sort()

    implementing = by_category.get(declared, [])

    # For `fixed`, enforce it: build one and try to write to its weights.
    enforced = None
    if declared == "fixed" and implementing:
        try:
            r = readout_mod.FixedRandomProjection(n_features=64, n_actions=8,
                                                  seed=0)
            try:
                r._W[0, 0] = 1.0
                enforced = False  # the write succeeded: not frozen at all
            except ValueError:
                enforced = True   # read-only array, as declared
        except Exception as exc:  # noqa: BLE001
            enforced = f"could not instantiate: {exc}"

    ok = (declared in READOUT_CATEGORIES and bool(implementing)
          and (enforced is not False))
    return {
        "declared_in_config": declared,
        "valid_categories": list(READOUT_CATEGORIES),
        "classes_by_category": by_category,
        "classes_implementing_declared": implementing,
        "weights_are_read_only": enforced,
        "pass": bool(ok),
        "note": ("§11.2: 'This choice is scientifically consequential and must "
                 "be frozen before confirmatory comparisons.' A declaration "
                 "with no class implementing it, or a `fixed` readout whose "
                 "weights can be written, is not a frozen choice."),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=datasets.DEFAULT)
    p.add_argument("--steps", type=int, default=480)
    p.add_argument("--episodes", type=int, default=8,
                   help="varied episodes, so score differs at equal position")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topologies", nargs="+",
                   default=["connectome", "shuffled_weights"])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    g = datasets.load(args.dataset)
    agent = FlyAgent(g, AgentConfig(topology="connectome", topology_seed=args.seed))
    env = MarioEnv(EnvConfig())

    print(f"{g.source} · injected {agent.vision.n_injected:,} cells · "
          f"eye {agent.vision.coverage()['eye']!r} · {args.steps} decisions\n")
    data = collect(env, agent, args.steps, episodes=args.episodes,
                   seed=args.seed)
    env.close()

    rep: dict = {"dataset": args.dataset, "steps": args.steps,
                 "coverage": agent.vision.coverage()}

    a = probe_hud_crop(agent, data["frames"][:24])
    rep["hud_crop"] = a
    print("A  HUD crop")
    print(f"     rows cropped                     {a['hud_rows']}")
    print(f"     current changes if HUD randomised {a['max_change_when_hud_randomised']:.6f}")
    print(f"     control, below the crop           {a['max_change_when_below_randomised']:.6f}")
    print(f"     -> {'PASS' if a['pass'] else 'FAIL'}\n")

    b = probe_leakage(data, agent)
    rep["leakage"] = b
    print("B  task-variable leakage, at matched position")
    for k, v in b["per_variable"].items():
        if "skipped" in v:
            print(f"     {k:<10} skipped: {v['skipped']}")
            continue
        flag = "LEAKS" if v["leaks"] else "no excess"
        sd = v["excess_sd_across_bins"]
        spread = f"+-{sd:.3f}" if sd is not None else "  (no SD, 1 bin)"
        print(f"     {k:<10} {v['accuracy']:.3f} vs majority {v['majority_class_baseline']:.3f}"
              f"  excess {v['excess_over_baseline']:+.3f} {spread}  {flag}  "
              f"({v['bins_used']} bins)")
    wc = b["positive_control_player_y_within_bins"]
    if "skipped" in wc:
        print(f"     control IN-BIN  player_y  skipped: {wc['skipped']}")
    else:
        wsd = wc["excess_sd_across_bins"]
        wspread = f"+-{wsd:.3f}" if wsd is not None else ""
        print(f"     control IN-BIN  player_y {wc['accuracy']:.3f} vs majority "
              f"{wc['majority_class_baseline']:.3f}  excess "
              f"{wc['excess_over_baseline']:+.3f} {wspread} ({wc['bins_used']} bins)")
    ctrl = b["positive_control_mario_x_uncontrolled"]
    print(f"     control GLOBAL  mario_x  {ctrl['accuracy']:.3f} "
          f"(chance {ctrl['chance']:.2f})  -- confounded by time, weak evidence")
    print(f"     -> power: {b['probe_has_power']} (basis: {b['power_basis']})")
    if b["variables_not_testable"]:
        print(f"     NOT TESTED: {', '.join(b['variables_not_testable'])}"
              " -- constant in this sample, no claim made")
    b["pass"] = bool(b["probe_has_power"] and not b["leaking_variables"])
    print(f"     -> {'PASS' if b['pass'] else 'FAIL'}\n")

    c = probe_no_learned_parameters(agent)
    rep["no_learned_parameters"] = c
    print("C  sensory interface has no fitted parameters")
    print(f"     arrays held        {', '.join(c['arrays_held'])}")
    print(f"     unexpected         {c['unexpected_arrays'] or 'none'}")
    print(f"     -> {'PASS' if c['pass'] else 'FAIL'}\n")

    d = probe_shared_injection_sites(g, args.dataset, args.topologies, args.seed)
    rep["shared_injection_sites"] = d
    print("D  injection sites identical across graph conditions (§10.2)")
    for topo, r in d["per_topology"].items():
        print(f"     {topo:<20} {r['n_injected']:,} cells · eye {r['eye']!r} · {r['sha']}")
    print(f"     -> {'PASS' if d['pass'] else 'FAIL'}\n")

    e = probe_readout_declaration()
    rep["readout_declaration"] = e
    print("E  readout category declared (§11.2)")
    print(f"     declared in config  {e['declared_in_config']!r}")
    for cat, names in sorted(e["classes_by_category"].items()):
        mark = "<- declared" if cat == e["declared_in_config"] else ""
        print(f"       {cat:<32} {', '.join(names)} {mark}")
    print(f"     weights read-only   {e['weights_are_read_only']}")
    print(f"     -> {'PASS' if e['pass'] else 'FAIL'}\n")

    verdict = all(x["pass"] for x in (a, b, c, d, e))
    rep["verdict"] = bool(verdict)
    print("=" * 66)
    print(f"GATE F {'PASS' if verdict else 'FAIL'}")
    print("=" * 66)
    if not e["pass"]:
        print("§11.2 needs an owner decision: which of the four categories is the")
        print("readout? It cannot be inferred from the code -- several exist.")

    graph_path = datasets.cache_path(args.dataset)
    rep["provenance"] = {
        "utc": datetime.now(timezone.utc).isoformat(), "machine": platform.node(),
        "commit": git_sha(), "python": platform.python_version(),
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "wall_seconds": round(time.perf_counter() - t0, 1), "argv": vars(args),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RUNS / f"gate_f_{args.dataset}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
