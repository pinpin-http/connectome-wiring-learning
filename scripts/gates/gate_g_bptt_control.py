#!/usr/bin/env python
"""Gate G controllability control: surrogate BPTT on the allowed motor edges.

    python scripts/gates/gate_g_bptt_control.py pilot
    python scripts/gates/gate_g_bptt_control.py evaluate

The staged pilot chooses Adam's learning rate on training data, adds one model
for the reverse mapping, and evaluates both on untouched streams as a narrow
controllability witness. The full evaluation still trains one BPTT model per
seed/readout/mapping and applies Gate G's existing controlled criterion. The
staged witness is not the full positive control.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario.models.spiking_reservoir import LIFParams  # noqa: E402
from flymario import config  # noqa: E402
from flymario.paths import RUNS  # noqa: E402
from flymario.run_manifest import RunManifest, dataset_files  # noqa: E402
from flymario.runctl import RunControl, StopRequested  # noqa: E402
from flymario.training.bptt import SurrogateBPTT  # noqa: E402
from flymario.training.two_cue import Learner, TwoCue, streams  # noqa: E402
from gate_g_task import N_TEST, N_TRAIN, READOUTS, SETS, criterion, setup  # noqa: E402

LR_GRID = (1e-4, 1e-3, 1e-2)
PILOT_TRIALS = 500
SMOKE_TRIALS = 1
SMOKE_LR = 1e-4
BLOCK_LEN = 1                    # staged control: ten-decision blocks do not fit
SMOKE_BLOCK_LEN = 1
WARMUP_TICKS = 150
TICKS_PER_DECISION = 16
DEVICE = "cuda"
CHECKPOINTS = RUNS / "checkpoints" / "gate_g_bptt"


def fit_one(S, seed_set: int, readout_seed: int, targets: np.ndarray,
            n_trials: int, lr: float, device: str = DEVICE, ctl=None,
            block_len: int = BLOCK_LEN, gpu_duty: float = 0.6,
            apply_update: bool = True, gradient_audit: bool = False,
            detect_anomaly: bool = False,
            state_grad_clip_norm: float | None = None):
    """Train one control model and return CPU magnitudes plus the training log."""
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.policy.readout import FixedRandomProjection
    import torch

    stream = streams(seed_set, "train", n_trials)
    params = LIFParams(gain=float(config.get("dynamics.gain")),
                       noise_std=float(config.get("dynamics.noise_std")))
    rd = np.asarray(FixedRandomProjection(len(S["motor"]), len(ACTION_NAMES), readout_seed).W)
    rd = rd[[ACTION_NAMES.index("RIGHT"), ACTION_NAMES.index("LEFT")]]
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    model = SurrogateBPTT(S["W"], S["motor"], S["cues"], rd, params,
                          seed=stream["noise_seed"], device=device, block_len=block_len)
    try:
        result = model.fit(stream["cues"], targets, n_trials, lr,
                           TICKS_PER_DECISION, WARMUP_TICKS, ctl=ctl,
                           gpu_duty=gpu_duty, apply_update=apply_update,
                           gradient_audit=gradient_audit,
                           detect_anomaly=detect_anomaly,
                           state_grad_clip_norm=state_grad_clip_norm)
    except BaseException as exc:
        exc.bptt_model_audit = model.audit
        raise
    if model.dev.type == "cuda":
        torch.cuda.synchronize(model.dev)
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(model.dev) / 2**20
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved(model.dev) / 2**20
    return model, result, stream


def cue_targets(cues: np.ndarray, mapping: int) -> np.ndarray:
    return np.asarray(cues, dtype=np.int64) ^ int(mapping)


def random_targets(stream: dict) -> np.ndarray:
    # The existing shuffled ±1 stream is independent of cue and action.
    return (np.asarray(stream["shuffled"]) > 0).astype(np.int64)


def evaluate_weights(S, seed_set: int, readout_seed: int, mapping: int,
                     weights, device: str = DEVICE, ctl=None,
                     gpu_duty: float = 0.6) -> float:
    L = [Learner("frozen", seed_set, None, 0.0, "true", mapping, readout_seed)]
    run = TwoCue(S["W"], S["motor"], S["cues"], L, "test", N_TEST,
                 record=True, device=device, cache_key=("gate_g", 0))
    run.brain.set_weights(run.torch.as_tensor(np.asarray(weights)[None, :],
                                               dtype=run.torch.float32, device=run.dev))
    run.run(N_TEST, ctl=ctl,
            gpu_duty=gpu_duty if run.dev.type == "cuda" else None)
    return float(run.log["correct"][:, 0].mean())


def save_weights(name: str, weights) -> str:
    import torch
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / f"{name}.pt"
    torch.save({"synapse_counts": weights.detach().cpu()}, path)
    return str(path.relative_to(ROOT))


def mode_pilot(S, args, ctl) -> None:
    rows = {}
    frozen_weights = None
    seed_set, readout_seed, mapping = 1, READOUTS[0], 0
    stream = streams(seed_set, "train", PILOT_TRIALS)
    targets = cue_targets(stream["cues"], mapping)
    for lr in LR_GRID:
        ctl.begin(f"pilot lr={lr:g}")
        t0 = time.time()
        model, result, _ = fit_one(S, seed_set, readout_seed, targets,
                                   PILOT_TRIALS, lr, args.device, ctl,
                                   block_len=args.block_len,
                                   gpu_duty=args.gpu_duty,
                                   state_grad_clip_norm=args.state_grad_clip_norm)
        if frozen_weights is None:
            frozen_weights = model.m0.detach().cpu().numpy().copy()
        rows[f"{lr:g}"] = {
            "lr": lr,
            "mean_accuracy_last_100": float(result["correct"][-100:].mean()),
            "mean_loss_last_100": float(result["loss"][-100:].mean()),
            "mean_accuracy_first_100": float(result["correct"][:100].mean()),
            "seconds": time.time() - t0,
            "peak_allocated_mib": result.get("peak_allocated_mib"),
            "peak_reserved_mib": result.get("peak_reserved_mib"),
            "weights": save_weights(f"pilot_lr_{lr:g}", result["weights"]),
        }
        del model, result
        ctl.done({"lr": lr, "accuracy_last_100": rows[f"{lr:g}"]["mean_accuracy_last_100"]})
        try:
            import torch
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    chosen = max(rows, key=lambda k: (rows[k]["mean_accuracy_last_100"],
                                     -rows[k]["mean_loss_last_100"], -rows[k]["lr"]))
    chosen_lr = float(rows[chosen]["lr"])
    import torch
    chosen_weights = torch.load(ROOT / rows[chosen]["weights"], map_location="cpu",
                                weights_only=True)["synapse_counts"].numpy()
    witness_map1_stream = streams(seed_set, "train", PILOT_TRIALS)
    witness_map1_targets = cue_targets(witness_map1_stream["cues"], 1)
    ctl.begin("staged witness mapping=1")
    model, result, _ = fit_one(
        S, seed_set, readout_seed, witness_map1_targets, PILOT_TRIALS,
        chosen_lr, args.device, ctl, block_len=args.block_len,
        gpu_duty=args.gpu_duty,
        state_grad_clip_norm=args.state_grad_clip_norm,
    )
    map1_weights = result["weights"].detach().cpu().numpy().copy()
    map1_file = save_weights("staged_witness_map1", result["weights"])
    map1_train_accuracy = float(result["correct"][-100:].mean())
    map1_peak_allocated_mib = result.get("peak_allocated_mib")
    map1_peak_reserved_mib = result.get("peak_reserved_mib")
    del model, result
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    ctl.done({"mapping": 1, "train_accuracy_last100": map1_train_accuracy})

    heldout, frozen = {}, {}
    for mapping, weights in ((0, chosen_weights), (1, map1_weights)):
        ctl.begin(f"staged heldout candidate mapping={mapping}")
        heldout[str(mapping)] = evaluate_weights(
            S, seed_set, readout_seed, mapping, weights, args.device, ctl,
            gpu_duty=args.gpu_duty,
        )
        ctl.done({"mapping": mapping, "heldout_candidate": heldout[str(mapping)]})
        ctl.begin(f"staged heldout frozen mapping={mapping}")
        frozen[str(mapping)] = evaluate_weights(
            S, seed_set, readout_seed, mapping, frozen_weights, args.device, ctl,
            gpu_duty=args.gpu_duty,
        )
        ctl.done({"mapping": mapping, "heldout_frozen": frozen[str(mapping)]})
    u = max(float(np.sqrt(max(p * (1.0 - p), 1e-12) / N_TEST))
            for p in [*heldout.values(), *frozen.values()])
    witness_checks = {
        "each_mapping_at_least_0.75": all(x >= 0.75 for x in heldout.values()),
        "beats_frozen_by_more_than_2u": all(
            heldout[m] - frozen[m] > 2.0 * u for m in heldout
        ),
    }
    staged_witness = {
        "protocol_version": 1,
        "scope": "one seed set, one readout, two mappings; no shuffled-target control; not full Gate G control",
        "block_len": args.block_len,
        "state_grad_clip_norm": args.state_grad_clip_norm,
        "training_trials_per_mapping": PILOT_TRIALS,
        "heldout_trials_per_mapping": N_TEST,
        "candidate_accuracy": heldout,
        "frozen_accuracy": frozen,
        "largest_binomial_se": u,
        "pass_rule": "both mappings >=0.75 and candidate - frozen > 2 * largest binomial SE",
        "checks": witness_checks,
        "pass": all(witness_checks.values()),
        "map1_weights": map1_file,
        "map1_train_accuracy_last100": map1_train_accuracy,
        "map1_peak_allocated_mib": map1_peak_allocated_mib,
        "map1_peak_reserved_mib": map1_peak_reserved_mib,
    }
    rep = {"what": "Gate G staged BPTT controllability witness",
           "training_rate_selection_uses_training_only": True,
           "seed_set": seed_set, "readout_seed": readout_seed,
           "mappings": [0, 1], "trials": PILOT_TRIALS, "block_len": args.block_len,
           "gpu_duty_target": args.gpu_duty,
           "gpu_memory_fraction": args.gpu_memory_fraction,
           "optimizer": "projected Adam", "learning_rates": rows,
           "state_grad_clip_norm": args.state_grad_clip_norm,
           "chosen_lr": chosen_lr,
           "staged_witness": staged_witness,
           "scope": "preliminary controllability witness only; full 18-model control not run",
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rep["manifest"] = args.manifest_path
    args.report_path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def mode_smoke(S, args, ctl) -> None:
    """Short memory/gradient check; numerical finiteness only, not learning."""
    seed_set, readout_seed, mapping = 1, READOUTS[0], 0
    n_trials = int(args.smoke_trials)
    block_len = int(args.smoke_block_len)
    if n_trials <= 0 or block_len <= 0 or block_len > n_trials:
        raise ValueError("smoke trials and block length must be positive; block <= trials")
    stream = streams(seed_set, "train", n_trials)
    targets = cue_targets(stream["cues"], mapping)
    ctl.begin("short BPTT memory/gradient check; no learning claim")
    t0 = time.time()
    if str(args.device).startswith("cuda"):
        import torch
        torch.cuda.reset_peak_memory_stats(args.device)
    try:
        model, result, _ = fit_one(
            S, seed_set, readout_seed, targets, n_trials, SMOKE_LR,
            args.device, ctl, block_len=block_len, gpu_duty=args.gpu_duty,
            apply_update=not args.no_optimizer_step,
            gradient_audit=args.diagnose_gradients,
            detect_anomaly=args.detect_anomaly,
            state_grad_clip_norm=args.state_grad_clip_norm,
        )
    except BaseException as exc:
        report = {
            "what": "BPTT gradient-source audit; no learning claim",
            "status": "failed before optimizer step",
            "training_only": True, "seed_set": seed_set,
            "readout_seed": readout_seed, "mapping": mapping,
            "trials": n_trials, "block_len": block_len,
            "optimizer_step_applied": False,
            "state_grad_clip_norm": args.state_grad_clip_norm,
            "detect_anomaly": bool(args.detect_anomaly),
            "failure": f"{type(exc).__name__}: {exc}",
            "gradient_audit": getattr(exc, "bptt_model_audit", None),
            "manifest": args.manifest_path,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(args.device) / 2**20
            report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(args.device) / 2**20
        args.report_path.write_text(json.dumps(report, indent=2))
        raise
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
        peak_allocated_mb = torch.cuda.max_memory_allocated(args.device) / 2**20
        peak_reserved_mb = torch.cuda.max_memory_reserved(args.device) / 2**20
    else:
        peak_allocated_mb = peak_reserved_mb = None
    rep = {
        "what": "BPTT finite-gradient/memory smoke; not a learning result",
        "training_only": True, "seed_set": seed_set,
        "readout_seed": readout_seed, "mapping": mapping,
        "trials": n_trials, "block_len": block_len, "lr": SMOKE_LR,
        "optimizer_step_applied": not args.no_optimizer_step,
        "state_grad_clip_norm": args.state_grad_clip_norm,
        "detect_anomaly": bool(args.detect_anomaly),
        "weights_finite": bool(model.torch.isfinite(result["weights"]).all()),
        "gradient_checks_completed": True,
        "peak_allocated_mib": peak_allocated_mb,
        "peak_reserved_mib": peak_reserved_mb,
        "gpu_duty_target": args.gpu_duty,
        "checkpoint_granularity": result["checkpoint_granularity"],
        "gradient_audit": result["gradient_audit"],
        "seconds": time.time() - t0,
        "manifest": args.manifest_path,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.report_path.write_text(json.dumps(rep, indent=2))
    ctl.done({"weights_finite": rep["weights_finite"]})
    print(json.dumps(rep, indent=2))


def mode_evaluate(S, args, ctl) -> None:
    pilot_path = RUNS / "gate_g_bptt_pilot.json"
    if not pilot_path.exists():
        raise SystemExit("run the training-only BPTT learning-rate pilot first")
    pilot = json.loads(pilot_path.read_text())
    lr = float(pilot["chosen_lr"])
    candidate, eta0, shuffled = {}, {}, {}
    candidate_by_mapping, no_learning_by_mapping, shuffled_by_mapping = {}, {}, {}
    model_files = {}
    n = {}
    for seed_set in SETS:
        for readout_seed in READOUTS:
            k = (seed_set, readout_seed)
            by_map, nolearn_by_map = [], []
            for mapping in (0, 1):
                name = f"set{seed_set}_ro{readout_seed}_map{mapping}"
                ctl.begin(name)
                t0 = time.time()
                model, result, _ = fit_one(
                    S, seed_set, readout_seed,
                    cue_targets(streams(seed_set, "train", N_TRAIN)["cues"], mapping),
                    N_TRAIN, lr, args.device, ctl, block_len=args.block_len,
                    gpu_duty=args.gpu_duty,
                    state_grad_clip_norm=args.state_grad_clip_norm,
                )
                model_files[name] = save_weights(name, result["weights"])
                by_map.append(evaluate_weights(S, seed_set, readout_seed, mapping,
                                               result["weights"].cpu().numpy(), args.device, ctl,
                                               args.gpu_duty))
                nolearn_by_map.append(evaluate_weights(S, seed_set, readout_seed, mapping,
                                                       model.m0.cpu().numpy(), args.device, ctl,
                                                       args.gpu_duty))
                print(f"BPTT {name}: held-out={by_map[-1]:.3f}, "
                      f"train-last100={result['correct'][-100:].mean():.3f}, "
                      f"{time.time() - t0:.1f}s", flush=True)
                del model, result
                if str(args.device).startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()
                ctl.done({"mapping": mapping, "held_out": by_map[-1]})
            candidate[k] = float(np.mean(by_map))
            eta0[k] = float(np.mean(nolearn_by_map))
            candidate_by_mapping[k] = {f"mapping{i}": float(v)
                                       for i, v in enumerate(by_map)}
            no_learning_by_mapping[k] = {f"mapping{i}": float(v)
                                         for i, v in enumerate(nolearn_by_map)}

            # One shuffled-target model per seed/readout; average its test score
            # over both mappings so its control is balanced by construction.
            tr = streams(seed_set, "train", N_TRAIN)
            shuffled_labels = random_targets(tr)
            name = f"set{seed_set}_ro{readout_seed}_shuffled"
            ctl.begin(name)
            model, result, _ = fit_one(S, seed_set, readout_seed, shuffled_labels,
                                       N_TRAIN, lr, args.device, ctl,
                                       block_len=args.block_len,
                                       gpu_duty=args.gpu_duty,
                                       state_grad_clip_norm=args.state_grad_clip_norm)
            model_files[name] = save_weights(name, result["weights"])
            shuf_by_map = [evaluate_weights(S, seed_set, readout_seed, mapping,
                                            result["weights"].cpu().numpy(), args.device, ctl,
                                            args.gpu_duty)
                           for mapping in (0, 1)]
            shuffled[k] = float(np.mean(shuf_by_map))
            shuffled_by_mapping[k] = {f"mapping{i}": float(v)
                                      for i, v in enumerate(shuf_by_map)}
            del model, result
            if str(args.device).startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
            ctl.done({"shuffled_target_mean": shuffled[k]})
            # Both mapping evaluations reuse the same held-out cue/noise
            # stream. Count the paired observations conservatively.
            n[k] = N_TEST

    judged = criterion(candidate, eta0, shuffled, n)
    mapping_ok = all(score >= 0.75 for values in candidate_by_mapping.values()
                     for score in values.values())
    judged["checks"]["each_mapping_at_least_0.75"] = bool(mapping_ok)
    judged["pass"] = bool(judged["pass"] and mapping_ok)
    rep = {"what": "Gate G BPTT controllability positive control",
           "optimizer": "projected Adam", "lr": lr,
           "n_train": N_TRAIN, "n_test_per_mapping": N_TEST,
           "candidate": {f"set{s}|ro{r}": v for (s, r), v in candidate.items()},
           "candidate_by_mapping": {f"set{s}|ro{r}": v
                                    for (s, r), v in candidate_by_mapping.items()},
           "no_learning": {f"set{s}|ro{r}": v for (s, r), v in eta0.items()},
           "no_learning_by_mapping": {f"set{s}|ro{r}": v
                                      for (s, r), v in no_learning_by_mapping.items()},
           "shuffled_target": {f"set{s}|ro{r}": v for (s, r), v in shuffled.items()},
           "shuffled_target_by_mapping": {f"set{s}|ro{r}": v
                                          for (s, r), v in shuffled_by_mapping.items()},
           "criterion": judged, "pass": bool(judged["pass"]),
           "block_len": args.block_len,
           "gpu_duty_target": args.gpu_duty,
           "checkpoint_granularity": "one simulation tick",
           "models": model_files,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rep["manifest"] = args.manifest_path
    (RUNS / "gate_g_bptt_control.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"criterion": judged, "pass": rep["pass"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=("smoke", "pilot", "evaluate"))
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.50,
                    help="cap this process's PyTorch GPU allocator (default: 50%%)")
    ap.add_argument("--gpu-duty", type=float, default=0.6,
                    help="target mean GPU duty across training blocks (default: 60%%)")
    ap.add_argument("--block-len", type=int, default=BLOCK_LEN,
                    help="truncated-BPTT decision horizon for pilot/evaluate (default: 1)")
    ap.add_argument("--smoke-trials", type=int, default=SMOKE_TRIALS,
                    help="smoke-only trial count; no learning claim is produced")
    ap.add_argument("--smoke-block-len", type=int, default=SMOKE_BLOCK_LEN,
                    help="smoke-only truncated-BPTT block length")
    ap.add_argument("--run-tag", default="",
                    help="optional tag to keep this run's report, checkpoints and live status separate")
    ap.add_argument("--no-optimizer-step", action="store_true",
                    help="smoke-only: inspect gradients without changing weights")
    ap.add_argument("--diagnose-gradients", action="store_true",
                    help="smoke-only: record per-tick state and gradient finiteness/maxima")
    ap.add_argument("--detect-anomaly", action="store_true",
                    help="smoke-only: ask PyTorch to trace the backward operation producing NaNs")
    ap.add_argument("--state-grad-clip-norm", type=float, default=None,
                    help="optional global-norm cap applied at each membrane/spike state during BPTT")
    args = ap.parse_args()
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        ap.error("--gpu-memory-fraction must be in (0, 1]")
    if not 0.0 < args.gpu_duty <= 1.0:
        ap.error("--gpu-duty must be in (0, 1]")
    if args.state_grad_clip_norm is not None and args.state_grad_clip_norm <= 0:
        ap.error("--state-grad-clip-norm must be positive")
    if args.block_len <= 0:
        ap.error("--block-len must be positive")
    if args.run_tag and not all(c.isalnum() or c in "-_" for c in args.run_tag):
        ap.error("--run-tag may contain only letters, digits, hyphens and underscores")
    if args.mode != "smoke" and (args.no_optimizer_step or args.diagnose_gradients or args.detect_anomaly):
        ap.error("gradient audit flags are smoke-only")
    if args.diagnose_gradients and not args.no_optimizer_step:
        ap.error("gradient diagnostics require --no-optimizer-step")
    run_id = f"gate_g_bptt_{args.mode}" + (f"_{args.run_tag}" if args.run_tag else "")
    if args.run_tag:
        global CHECKPOINTS
        CHECKPOINTS = RUNS / "checkpoints" / f"gate_g_bptt_{args.run_tag}"
    if str(args.device).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("the requested GPU is not available")
        # Leave VRAM headroom for the display compositor and other desktop use.
        memory_device = torch.device(args.device)
        if memory_device.index is None:
            memory_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,
                                                   device=memory_device)
    S = setup()
    pilot = args.mode == "pilot"
    smoke = args.mode == "smoke"
    output_name = {"smoke": f"gate_g_bptt_smoke_t{args.smoke_trials}_b{args.smoke_block_len}.json",
                   "pilot": "gate_g_bptt_pilot.json",
                   "evaluate": "gate_g_bptt_control.json"}[args.mode]
    if args.run_tag:
        output_name = output_name.removesuffix(".json") + f"_{args.run_tag}.json"
    output = RUNS / output_name
    args.report_path = output
    outputs = {"report": output}
    if smoke:
        seed_registry = {"master": 20260923, "train_seed_set": 1,
                         "readout_seed": READOUTS[0], "mapping": 0,
                         "stream_derivation": "SeedSequence([MASTER, seed_set, purpose]); spawn noise, cue, shuffled"}
    elif pilot:
        outputs.update({f"pilot_lr_{lr:g}": CHECKPOINTS / f"pilot_lr_{lr:g}.pt"
                        for lr in LR_GRID})
        outputs["staged_witness_map1"] = CHECKPOINTS / "staged_witness_map1.pt"
        seed_registry = {"master": 20260923, "train_seed_set": 1,
                         "readout_seed": READOUTS[0], "mapping": 0,
                         "stream_derivation": "SeedSequence([MASTER, seed_set, purpose]); spawn noise, cue, shuffled"}
    else:
        outputs.update({f"model_{s}_{r}_{mapping}": CHECKPOINTS /
                        f"set{s}_ro{r}_map{mapping}.pt"
                        for s in SETS for r in READOUTS for mapping in (0, 1)})
        outputs.update({f"shuffled_{s}_{r}": CHECKPOINTS /
                        f"set{s}_ro{r}_shuffled.pt" for s in SETS for r in READOUTS})
        seed_registry = {"master": 20260923, "seed_sets": list(SETS),
                         "readout_seeds": list(READOUTS), "mappings": [0, 1],
                         "stream_derivation": "SeedSequence([MASTER, seed_set, purpose]); spawn noise, cue, shuffled"}
        outputs["pilot_config"] = RUNS / "gate_g_bptt_pilot.json"
    live = RUNS / "live" / run_id
    outputs.update({"live_status": live / "status.json", "live_series": live / "series.jsonl"})
    inputs = dataset_files(str(config.get("connectome.dataset")),
                           int(config.get("connectome.min_syn")))
    inputs.update({"bptt_script": Path(__file__),
                   "bptt_implementation": ROOT / "src/flymario/training/bptt.py",
                   "gate_task": ROOT / "scripts/gates/gate_g_task.py",
                   "two_cue": ROOT / "src/flymario/training/two_cue.py",
                   "plastic_brain": ROOT / "src/flymario/training/plastic.py",
                   "base_config": ROOT / "configs/base.yaml",
                   "gate_plan": ROOT / "docs/GATE_G_PLAN.md",
                   "gate_pair": RUNS / "gate_g_pair.json"})
    if args.mode == "evaluate":
        inputs["pilot_report"] = RUNS / "gate_g_bptt_pilot.json"
    manifest = RunManifest(
        run_id=run_id, study="Gate G",
        phase=f"surrogate BPTT controllability {args.mode}",
        configuration={"arguments": vars(args), "learning_rates": list(LR_GRID),
                       "runtime_debug": {"HIP_LAUNCH_BLOCKING":
                                         os.environ.get("HIP_LAUNCH_BLOCKING", "0")},
                       "gpu_memory_fraction": args.gpu_memory_fraction,
                       "burn_in_gradient": "detached before first scored decision",
                       "n_train": args.smoke_trials if smoke else PILOT_TRIALS if pilot else N_TRAIN,
                       "n_test_per_mapping": N_TEST if pilot else 0 if smoke else N_TEST,
                       "block_len": args.smoke_block_len if smoke else args.block_len,
                       "smoke_trials": args.smoke_trials if smoke else None,
                       "optimizer_step_applied": not args.no_optimizer_step if smoke else None,
                       "diagnose_gradients": args.diagnose_gradients if smoke else False,
                       "detect_anomaly": args.detect_anomaly if smoke else False,
                       "state_grad_clip_norm": args.state_grad_clip_norm,
                       "checkpoint_granularity": "one simulation tick",
                       "gpu_duty_target": args.gpu_duty,
                       "warmup_ticks": WARMUP_TICKS,
                       "ticks_per_decision": TICKS_PER_DECISION,
                       "fixed_readout": True, "trainable_subset": "incoming motor edges"},
        seeds=seed_registry, dataset=str(config.get("connectome.dataset")),
        input_files=inputs, output_files=outputs)
    args.manifest_path = str(manifest.path)
    ctl = RunControl(run_id,
                     total=1 if smoke else 8 if pilot else 18,
                     checkpoint=lambda: None, unit="model", meta={
        "question": "Can the fixed-readout task be controlled by training exactly the allowed motor edges?",
        "method": "Hard-spike forward dynamics with surrogate BPTT and projected Adam; same fixed readout and incoming-motor-edge mask as Gate G."})
    try:
        mode = {"smoke": mode_smoke, "pilot": mode_pilot,
                "evaluate": mode_evaluate}[args.mode]
        mode(S, args, ctl)
    except StopRequested:
        ctl.close("stopped")
        manifest.finish("stopped")
        return
    except BaseException as exc:
        ctl.close("failed")
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    ctl.close("complete")
    manifest.finish("complete")


if __name__ == "__main__":
    main()
