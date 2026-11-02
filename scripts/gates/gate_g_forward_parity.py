#!/usr/bin/env python
"""No-update BPTT/PlasticLIF forward validation on explicit shared noise.

    OMP_NUM_THREADS=12 .venv/bin/python scripts/gates/gate_g_forward_parity.py \
        --device cuda --run-tag malecns
    OMP_NUM_THREADS=1 .venv/bin/python scripts/gates/gate_g_forward_parity.py \
        --small --device cpu --run-tag cpu

Only training streams are used. Every tick is compared, including burn-in;
initial weights and a prescribed bounded perturbation are separate runs. This
is simulator validation, with no backward pass, optimizer or learning claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario import config
from flymario.agent import AgentConfig
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.run_manifest import RunManifest, dataset_files, sha256_file
from flymario.runctl import RunControl, StopRequested
from flymario.training.bptt import SurrogateBPTT
from flymario.training.plastic import PlasticLIF
from flymario.training.two_cue import MASTER, streams


def small_setup() -> dict:
    """An active signed recurrent network, for cheap CPU instrument checks."""
    rng = np.random.default_rng(112358)
    n = 64
    edges = rng.random((n, n)) < 0.12
    np.fill_diagonal(edges, False)
    signs = np.where(rng.random(n) < 0.75, 1.0, -1.0)
    values = rng.integers(1, 5, size=(n, n)).astype(np.float32) * 0.25
    W = sp.csr_matrix((edges * values * signs[None, :]).astype(np.float32))
    motor = np.arange(n - 12, n, dtype=np.int64)
    currents = rng.uniform(0.1, 0.35, (2, n)).astype(np.float32)
    return {"W": W, "motor": motor, "cues": currents}


def tensor_sha256(value) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare_forward(S: dict, *, params: LIFParams, seed_set: int, readout_seed: int,
                    weight_case: str, n_trials: int, warmup_ticks: int,
                    ticks_per_decision: int, device: str, ctl=None,
                    gpu_duty: float = 0.6) -> dict:
    """Compare independent simulator states against identical explicit inputs.

    There is deliberately no tolerance on spike agreement: small membrane
    rounding can cause a real threshold disagreement. Continuous errors and
    downstream decisions are reported separately, never used to hide it.
    """
    import torch

    if weight_case not in ("initial", "bounded_modified"):
        raise ValueError("weight_case must be initial or bounded_modified")
    if n_trials <= 0 or warmup_ticks < 0 or ticks_per_decision <= 0:
        raise ValueError("positive trial/decision counts and nonnegative warm-up required")
    if not 0.0 < gpu_duty <= 1.0:
        raise ValueError("gpu_duty must be in (0, 1]")
    stream = streams(seed_set, "train", n_trials)
    readout = np.asarray(FixedRandomProjection(len(S["motor"]), len(ACTION_NAMES),
                                              readout_seed).W)
    readout = readout[[ACTION_NAMES.index("RIGHT"), ACTION_NAMES.index("LEFT")]]
    model = SurrogateBPTT(S["W"], S["motor"], S["cues"], readout, params,
                          seed=stream["noise_seed"], device=device)
    reference = PlasticLIF(S["W"], S["motor"], params,
                           seeds=[stream["noise_seed"]], device=device)
    if not (torch.equal(model.edge_index[0], reference.post)
            and torch.equal(model.edge_index[1], reference.pre)
            and torch.equal(model.sign, reference.sign)
            and torch.equal(model.m0, reference.m0)):
        raise AssertionError("plastic edge order, signs or initial magnitudes differ")

    # Independent of cue labels, reward, readout and all held-out streams.
    # Cycling [0, .5, 1.5, 2] also exercises both allowed endpoints.
    factors = torch.tensor([0.0, 0.5, 1.5, 2.0], device=model.dev)
    magnitudes = model.m0.clone()
    if weight_case == "bounded_modified":
        magnitudes *= factors[torch.arange(magnitudes.numel(), device=model.dev) % 4]
    if not bool(((magnitudes >= 0) & (magnitudes <= model.hi)).all()):
        raise AssertionError("diagnostic weights violate the declared bounds")
    initial_weights_hash = tensor_sha256(magnitudes)
    with torch.no_grad():
        model.m.copy_(magnitudes)
        reference.set_weights(magnitudes[None, :])
        Wm = model._motor_matrix(model.m)

    beta = float(np.exp(-params.dt_ms / AgentConfig().trace_tau_ms))
    trace_ref = torch.zeros((len(S["motor"]), 1), device=model.dev)
    state = model.state
    generator = torch.Generator(device=model.dev).manual_seed(stream["noise_seed"])
    noise_digest = hashlib.sha256()
    ticks, decisions = [], []
    first_spike_divergence = None
    pace_clock = [time.monotonic()]
    total_ticks = warmup_ticks + n_trials * ticks_per_decision
    start = time.monotonic()
    with torch.no_grad():
        for tick in range(total_ticks):
            trial = -1 if tick < warmup_ticks else (tick - warmup_ticks) // ticks_per_decision
            cue = int(stream["cues"][max(trial, 0)])
            current = model.cues[:, cue]
            # Draw once per tick, in the reference simulator's original shape.
            # Both engines receive this tensor, not merely a matching seed.
            xi = torch.randn(model.n, generator=generator, device=model.dev)[:, None]
            xi = xi * params.noise_std
            noise_digest.update(xi.cpu().contiguous().numpy().tobytes())
            state = model._tick(*state, current, xi, Wm)
            spikes_ref = reference.step(current, noise=xi)
            trace_ref.mul_(beta).add_(spikes_ref.index_select(0, reference.motor).float())
            v, refrac, spikes, trace = state
            mismatch = spikes.bool() != spikes_ref
            spike_diff = int(mismatch.sum().item())
            motor_diff = int(mismatch.index_select(0, reference.motor).sum().item())
            if spike_diff and first_spike_divergence is None:
                neurons = torch.nonzero(mismatch[:, 0], as_tuple=False)[:16, 0]
                first_spike_divergence = {
                    "tick": tick, "phase": "warmup" if trial < 0 else "decision",
                    "trial": trial, "neuron_indices": neurons.cpu().tolist(),
                    "bptt_spikes": spikes[neurons, 0].cpu().tolist(),
                    "plastic_spikes": spikes_ref[neurons, 0].cpu().tolist(),
                    "previous_tick": ticks[-1] if ticks else None,
                }
            row = {
                "tick": tick, "trial": trial, "cue": cue,
                "bptt_spikes": int(spikes.sum().item()),
                "plastic_spikes": int(spikes_ref.sum().item()),
                "spike_mismatches": spike_diff, "motor_spike_mismatches": motor_diff,
                "refractory_mismatches": int((refrac != reference.refrac).sum().item()),
                "membrane_max_abs_error": float((v - reference.v).abs().max().item()),
                "trace_max_abs_error": float((trace - trace_ref).abs().max().item()),
                "finite": bool(torch.isfinite(v).all() & torch.isfinite(reference.v).all()
                               & torch.isfinite(trace).all() & torch.isfinite(trace_ref).all()),
            }
            ticks.append(row)
            if trial >= 0 and (tick - warmup_ticks + 1) % ticks_per_decision == 0:
                scores = model.readout @ trace
                ref_scores = model.readout @ trace_ref
                decisions.append({
                    "trial": trial, "cue": cue,
                    "bptt_logits": scores[:, 0].cpu().tolist(),
                    "plastic_logits": ref_scores[:, 0].cpu().tolist(),
                    "bptt_action": int(scores[:, 0].argmax().item()),
                    "plastic_action": int(ref_scores[:, 0].argmax().item()),
                    "logit_max_abs_error": float((scores - ref_scores).abs().max().item()),
                })
            if ctl is not None:
                ctl.sub(tick + 1, total_ticks, "shared-noise forward tick")
            model._pace_gpu(pace_clock, gpu_duty)

    unchanged = (tensor_sha256(model.m) == initial_weights_hash
                 and tensor_sha256(reference.m[0]) == initial_weights_hash)
    exact_spikes = all(row["spike_mismatches"] == 0 for row in ticks)
    exact_trace = all(row["trace_max_abs_error"] == 0 for row in ticks)
    finite = all(row["finite"] for row in ticks)
    return {
        "seed_set": seed_set, "purpose": "train", "noise_seed": stream["noise_seed"],
        "readout_seed": readout_seed, "weight_case": weight_case,
        "neurons": model.n, "motor_neurons": model.M, "plastic_edges": model.m.numel(),
        "edge_order_identical": True, "weight_sha256": initial_weights_hash,
        "noise_sha256": noise_digest.hexdigest(), "ticks_compared": total_ticks,
        "neuron_tick_comparisons": total_ticks * model.n,
        "motor_tick_comparisons": total_ticks * model.M,
        "spike_mismatches": sum(row["spike_mismatches"] for row in ticks),
        "motor_spike_mismatches": sum(row["motor_spike_mismatches"] for row in ticks),
        "ticks_with_spike_mismatch": sum(row["spike_mismatches"] > 0 for row in ticks),
        "max_membrane_abs_error": max(row["membrane_max_abs_error"] for row in ticks),
        "max_trace_abs_error": max(row["trace_max_abs_error"] for row in ticks),
        "decision_action_mismatches": sum(d["bptt_action"] != d["plastic_action"]
                                           for d in decisions),
        "decisions_compared": n_trials, "all_states_finite": finite,
        "weights_unchanged": unchanged, "exact_spikes": exact_spikes,
        "exact_motor_trace": exact_trace,
        "exact_forward_parity": bool(exact_spikes and exact_trace and finite and unchanged
            and all(row["refractory_mismatches"] == 0 for row in ticks)),
        "first_spike_divergence": first_spike_divergence,
        "seconds": time.monotonic() - start, "ticks": ticks, "decisions": decisions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true", help="64-neuron instrument check")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed-sets", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.50)
    ap.add_argument("--gpu-duty", type=float, default=0.6)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--resume", action="store_true", help="retain completed cases; rerun interrupted one")
    args = ap.parse_args()
    if args.trials <= 0 or not 0 < args.gpu_memory_fraction <= 0.50:
        ap.error("positive --trials and --gpu-memory-fraction in (0, .50] required")
    if not 0 < args.gpu_duty <= 1:
        ap.error("--gpu-duty must be in (0, 1]")
    if set(args.seed_sets) - {1, 2} or len(set(args.seed_sets)) != len(args.seed_sets):
        ap.error("--seed-sets must be distinct members of the declared training sets 1, 2")
    if args.run_tag and not all(c.isalnum() or c in "-_" for c in args.run_tag):
        ap.error("--run-tag may contain letters, digits, hyphens and underscores only")
    import torch

    if torch.device(args.device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("the requested GPU is not available; no silent CPU fallback")
        memory_device = torch.device(args.device)
        if memory_device.index is None:
            memory_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, memory_device)
        torch.cuda.reset_peak_memory_stats(memory_device)
    run_id = "gate_g_forward_parity" + ("_small" if args.small else "")
    if args.run_tag:
        run_id += "_" + args.run_tag
    output = RUNS / f"{run_id}.json"
    warmup = int(config.get("dynamics.warmup_ticks"))
    tpd = int(config.get("timing.ticks_per_decision"))
    params = (LIFParams(gain=0.15, noise_std=0.08, bias=0.02,
                        refractory_ms=6.66, v_reset=-0.1) if args.small else LIFParams())
    readout_seed = block_seeds(3)[0]
    design = {
        "small": args.small, "device": args.device, "n_trials": args.trials,
        "seed_sets": args.seed_sets, "purpose": "train", "readout_seed": readout_seed,
        "warmup_ticks": warmup, "ticks_per_decision": tpd, "params": asdict(params),
        "weight_cases": ["initial", "bounded_modified"],
        "modified_factors": [0, 0.5, 1.5, 2], "backward_or_optimizer": False,
        "noise": "one explicit per-tick torch sample shared by both engines",
        "gpu_memory_fraction": args.gpu_memory_fraction, "gpu_duty": args.gpu_duty,
        "normalization": config.get("dynamics.weight_transform"),
        "dataset": "synthetic-64" if args.small else config.get("connectome.dataset"),
        "implementation_sha256": {
            name: sha256_file(path) for name, path in {
                "diagnostic": Path(__file__),
                "bptt": ROOT / "src/flymario/training/bptt.py",
                "plastic": ROOT / "src/flymario/training/plastic.py",
                "config": ROOT / "configs/base.yaml",
            }.items()
        },
    }
    report = {"what": "Gate G no-update forward parity; simulator validation only",
              "complete": False, "design": design, "cases": {},
              "utc": datetime.now(UTC).isoformat(timespec="seconds")}
    if args.resume and output.exists():
        report = json.loads(output.read_text())
        if report["design"] != design:
            raise RuntimeError("saved design differs; use a new run tag")
    elif output.exists():
        raise FileExistsError(f"{output} exists; use --resume or a new --run-tag")
    report["complete"] = False

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        part = output.with_suffix(".json.part")
        part.write_text(json.dumps(report, indent=2, allow_nan=False))
        part.replace(output)

    inputs = {} if args.small else dataset_files()
    inputs.update({"diagnostic": Path(__file__),
                   "bptt": ROOT / "src/flymario/training/bptt.py",
                   "plastic": ROOT / "src/flymario/training/plastic.py",
                   "two_cue": ROOT / "src/flymario/training/two_cue.py",
                   "base_config": ROOT / "configs/base.yaml",
                   "gate_plan": ROOT / "docs/GATE_G_PLAN.md"})
    if not args.small:
        inputs["chosen_cue_pair"] = RUNS / "gate_g_pair.json"
    manifest = RunManifest(run_id, "Gate G", "no-update forward parity", design,
                           seeds={"master": MASTER, "purpose": "train",
                                  "seed_sets": args.seed_sets, "readout_seed": readout_seed},
                           dataset=design["dataset"], input_files=inputs,
                           output_files={"report": output})
    report["manifest"] = str(manifest.path)
    save()
    ctl = RunControl(run_id, total=2 * len(args.seed_sets), checkpoint=save,
                     already_done=len(report["cases"]), fresh=not args.resume,
                     unit="comparison", meta={"question": "Do training and evaluation forwards agree?",
                                              "method": "fixed weights and identical explicit noise"})
    try:
        if args.small:
            S = small_setup()
        else:
            from gate_g_task import setup
            S = setup()
        for seed_set in args.seed_sets:
            for weight_case in design["weight_cases"]:
                key = f"set{seed_set}_{weight_case}"
                if key in report["cases"]:
                    continue
                ctl.begin(key)
                result = compare_forward(
                    S, params=params, seed_set=seed_set, readout_seed=readout_seed,
                    weight_case=weight_case, n_trials=args.trials, warmup_ticks=warmup,
                    ticks_per_decision=tpd, device=args.device, ctl=ctl,
                    gpu_duty=args.gpu_duty,
                )
                report["cases"][key] = result
                save()
                summary = {k: result[k] for k in ("spike_mismatches", "max_trace_abs_error",
                                                "decision_action_mismatches", "exact_forward_parity")}
                ctl.done(summary)
                print(json.dumps({"case": key, **summary}), flush=True)
        report["complete"] = True
        report["exact_forward_parity"] = all(c["exact_forward_parity"]
                                             for c in report["cases"].values())
        if torch.device(args.device).type == "cuda":
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(args.device) / 2**20
            report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(args.device) / 2**20
        save()
    except StopRequested:
        save()
        ctl.close("stopped")
        manifest.finish("stopped")
        return
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        ctl.close("failed")
        manifest.finish("failed", error=report["error"])
        raise
    ctl.close("complete")
    manifest.finish("complete")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
