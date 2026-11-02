#!/usr/bin/env python
"""Training-only, matched-state probe of surrogate-gradient update directions.

No optimizer is run and no learned checkpoint or Gate verdict is produced.
Every proposed change is replayed from the same frozen-network state, first
with the noise used to compute the direction, then with independent training
noise. A one-decision result cannot establish two-cue controllability.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flymario import config
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.bptt import SurrogateBPTT, _SurrogateSpike
from flymario.training.two_cue import MASTER, streams

VARIANTS = {
    "baseline": {"surrogate_slope": 1.0, "detach_reset": False},
    "narrow_detached": {"surrogate_slope": 25.0, "detach_reset": True},
}
DEFAULT_STEPS = (1e-4, 1e-3, 1e-2, 1e-1)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def relative_direction(m0, gradient):
    """Return RMS-one direction in r=m/m0 coordinates, or reject bad input.

    Double precision prevents a finite but large float32 gradient overflowing
    when multiplied by a large synapse count or when squared for its norm.
    """
    import torch
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("non-finite plastic-edge gradient")
    relative = m0.double() * gradient.double()
    rms = relative.square().mean().sqrt()
    if not bool(torch.isfinite(rms)) or float(rms) <= 0:
        raise FloatingPointError("zero or non-finite relative-coordinate gradient")
    return (relative / rms).to(m0.dtype), float(rms)


def propose_weights(m0, direction, alpha: float):
    """Projected relative update; alpha's units are relative RMS before bounds."""
    import torch
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("relative step must be finite and positive")
    if not bool(torch.isfinite(direction).all()):
        raise FloatingPointError("non-finite update direction")
    raw = 1.0 + float(alpha) * direction
    magnitudes = m0 * raw.clamp(0.0, 2.0)
    actual = (magnitudes.double() - m0.double()) / m0.double()
    return magnitudes, {
        "requested_relative_rms": float(alpha),
        "realized_relative_rms": float(actual.square().mean().sqrt()),
        "realized_relative_max_abs": float(actual.abs().max()),
        "projected_fraction": float(((raw < 0) | (raw > 2)).float().mean()),
        "unchanged_fraction": float((magnitudes == m0).float().mean()),
    }


def replay(model, initial_state, magnitudes, cues, targets, noise,
           *, differentiate=False, ctl=None, gpu_duty=1.0):
    """Replay an immutable state/noise sample and return CE plus decision data."""
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    if noise.ndim != 4 or noise.shape[0] != len(cues):
        raise ValueError("noise must have shape (decisions, ticks, neurons, 1)")
    if len(cues) != len(targets) or not len(cues):
        raise ValueError("one target required per nonempty cue sequence")
    state = tuple(value.detach().clone() for value in initial_state)
    losses, margins, scores_log, correct = [], [], [], []
    clock = [time.monotonic()]
    with torch.set_grad_enabled(differentiate):
        matrix = model._motor_matrix(magnitudes)
        tick = 0
        for cue, target, decision_noise in zip(cues, targets, noise):
            current = model.cues[:, int(cue)]
            for xi in decision_noise:
                if differentiate:
                    state = checkpoint(model._tick, *state, current, xi, matrix,
                                       use_reentrant=False)
                    if model.state_grad_clip_norm is not None:
                        for kind, value in (("membrane", state[0]), ("spikes", state[2])):
                            if value.requires_grad:
                                value.register_hook(
                                    lambda grad, kind=kind, tick=tick:
                                    model._state_gradient_hook(kind, tick, grad))
                else:
                    state = model._tick(*state, current, xi, matrix)
                tick += 1
                model._pace_gpu(clock, gpu_duty)
                if ctl is not None:
                    ctl.sub(tick, int(noise.shape[0] * noise.shape[1]), "replay tick")
            scores = model.readout @ state[3]
            label = torch.tensor([int(target)], device=model.dev)
            losses.append(F.cross_entropy(scores.T / model.temperature, label))
            margins.append(scores[int(target), 0] - scores[1 - int(target), 0])
            scores_log.append(scores[:, 0].detach().cpu().tolist())
            correct.append(int(scores[:, 0].argmax()) == int(target))
        loss = torch.stack(losses).mean()
        metrics = {
            "loss": float(loss.detach()),
            "mean_target_margin": float(torch.stack(margins).mean().detach()),
            "accuracy": float(np.mean(correct)),
            "decision_scores_right_left": scores_log,
            "decision_target_margins": [float(m.detach()) for m in margins],
        }
    if not np.isfinite(metrics["loss"]) or not np.isfinite(metrics["mean_target_margin"]):
        raise FloatingPointError("non-finite forward objective")
    return loss, metrics


def compare_to_zero(metrics, zero):
    return {**metrics, "loss_delta_vs_zero": metrics["loss"] - zero["loss"],
            "target_margin_delta_vs_zero":
            metrics["mean_target_margin"] - zero["mean_target_margin"],
            "accuracy_delta_vs_zero": metrics["accuracy"] - zero["accuracy"]}


def main() -> None:
    import torch

    from flymario.connectome import datasets
    from flymario.env.retro_env import ACTION_NAMES
    from flymario.policy.readout import FixedRandomProjection, block_seeds
    from flymario.training.two_cue import cue_currents

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-tag", default="direction-v1")
    parser.add_argument("--seed-set", type=int, default=1)
    parser.add_argument("--mapping", type=int, choices=(0, 1), default=0)
    parser.add_argument("--trial-starts", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--decisions", type=int, default=1)
    parser.add_argument("--steps", type=float, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--match-random-rms", action="store_true",
                        help="match random control's realized RMS to projected negative gradient")
    parser.add_argument("--state-grad-clip-norm", type=float, default=1e9,
                        help="state-adjoint cap; zero disables it")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    parser.add_argument("--gpu-duty", type=float, default=0.6)
    args = parser.parse_args()
    if (args.decisions <= 0 or min(args.trial_starts) < 0 or
            len(set(args.trial_starts)) != len(args.trial_starts)):
        parser.error("decisions must be positive and trial starts distinct/nonnegative")
    if any(not np.isfinite(x) or x <= 0 for x in args.steps):
        parser.error("step sizes must be finite and positive")
    if not 0 < args.gpu_memory_fraction <= 1 or not 0 < args.gpu_duty <= 1:
        parser.error("GPU memory fraction/duty must be in (0, 1]")
    if args.state_grad_clip_norm < 0 or not np.isfinite(args.state_grad_clip_norm):
        parser.error("state-gradient cap must be finite and nonnegative")
    if not args.run_tag or not all(c.isalnum() or c in "-_" for c in args.run_tag):
        parser.error("run tag requires letters/digits/hyphens/underscores")
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("the requested GPU is unavailable")
        dev = torch.device(args.device)
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)

    total_trials = max(args.trial_starts) + args.decisions
    stream = streams(args.seed_set, "train", total_trials)
    tpd, warm = int(config.get("timing.ticks_per_decision")), int(config.get("dynamics.warmup_ticks"))
    dataset = str(config.get("connectome.dataset"))
    min_syn = int(config.get("connectome.min_syn"))
    readout_seed = block_seeds(3)[0]
    noise_seed = int(np.random.SeedSequence(
        [MASTER, args.seed_set, 20260927, 341]).generate_state(1)[0] % (2**31 - 1))
    random_seed = int(np.random.SeedSequence(
        [MASTER, args.seed_set, 20260927, 342]).generate_state(1)[0] % (2**31 - 1))
    run_id = f"gate_g_bptt_direction_{args.run_tag}"
    output = RUNS / f"{run_id}.json"
    if output.exists():
        raise FileExistsError(f"preserve existing diagnostic: choose a fresh run tag ({output})")
    report = {
        "what": "Training-only matched-state surrogate-gradient direction diagnostic",
        "complete": False, "training_only": True, "learning_run": False,
        "gate_verdict": None, "arguments": vars(args),
        "variants": {name: VARIANTS[name] for name in args.variants},
        "hypotheses": {
            "useful_local_direction": "negative-gradient CE decreases versus zero, positive-gradient and matched-step random directions",
            "noise_transfer": "direction computed on original noise also decreases CE under independent training noise from the same initial state",
            "limitation": "own-noise descent is in-sample; two noise replays and a few windows do not establish learning, generalization, or a Gate pass",
        },
        "coordinate_system": "r=m/m0, d=m0*(dL/dm)/RMS(m0*dL/dm); proposal m=m0*clip(1+alpha*d,0,2)",
        "objective": "mean cross-entropy of fixed RIGHT/LEFT scores, temperature 1",
        "state_grad_clip_norm": args.state_grad_clip_norm or None,
        "initial_state": "frozen original weights, continuous training-stream prefix; identical state for every replay of a window",
        "noise_sampling": "warmup drawn once, then one (ticks,neurons,1) tensor per decision; independent replay noise uses a separate training-only seed",
        "seeds": {"master": MASTER, "seed_set": args.seed_set,
                  "readout_seed": readout_seed, "original_training_noise": stream["noise_seed"],
                  "independent_training_noise": noise_seed, "random_direction": random_seed},
        "timing": {"warmup_ticks": warm, "ticks_per_decision": tpd},
        "windows": [],
    }
    inputs = dataset_files(dataset, min_syn)
    inputs.update({"script": Path(__file__), "bptt": ROOT / "src/flymario/training/bptt.py",
                   "base_config": ROOT / "configs/base.yaml", "pair": RUNS / "gate_g_pair.json",
                   "plan": ROOT / "docs/GATE_G_PLAN.md"})
    manifest = RunManifest(run_id, "Gate G", "training-only direction diagnostic",
                           vars(args), seeds=report["seeds"], dataset=dataset,
                           input_files=inputs, output_files={"report": output})
    report["manifest"] = str(manifest.path)
    save = lambda: atomic_json(output, report)
    ctl = RunControl(run_id, total=len(args.trial_starts) * len(args.variants),
                     checkpoint=save, unit="direction probe", meta={"question": report["what"]})
    save()
    started = time.monotonic()
    try:
        graph = datasets.load(dataset, min_syn)
        W, motor = graph.to_sparse(), datasets.motor_index(graph, dataset)
        pair = json.loads((RUNS / "gate_g_pair.json").read_text())["chosen"]["pair"]
        cues = cue_currents(graph, pair)
        rows = FixedRandomProjection(len(motor), len(ACTION_NAMES), readout_seed).W[
            [ACTION_NAMES.index("RIGHT"), ACTION_NAMES.index("LEFT")]]
        model = SurrogateBPTT(W, motor, cues, rows,
                              LIFParams(gain=float(config.get("dynamics.gain")),
                                        noise_std=float(config.get("dynamics.noise_std"))),
                              stream["noise_seed"], device=args.device, block_len=args.decisions)
        model.state_grad_clip_norm = args.state_grad_clip_norm or None
        model.warm_up(int(stream["cues"][0]), warm, gpu_duty=args.gpu_duty)
        frozen_states, original_noise = {}, []
        state = tuple(x.detach().clone() for x in model.state)
        clock = [time.monotonic()]
        with torch.no_grad():
            matrix = model._motor_matrix(model.m0)
            for trial, cue in enumerate(stream["cues"]):
                if trial in args.trial_starts:
                    frozen_states[trial] = tuple(x.clone() for x in state)
                noise = model._noise(tpd)
                original_noise.append(noise)
                for xi in noise:
                    state = model._tick(*state, model.cues[:, int(cue)], xi, matrix)
                    model._pace_gpu(clock, args.gpu_duty)
                ctl.sub(trial + 1, total_trials, "frozen training prefix")
        independent_rng = torch.Generator(device=model.dev).manual_seed(noise_seed)
        random_rng = torch.Generator(device="cpu").manual_seed(random_seed)
        for trial_start in args.trial_starts:
            window_cues = stream["cues"][trial_start:trial_start + args.decisions]
            targets = window_cues ^ args.mapping
            initial = frozen_states[trial_start]
            primary_noise = torch.stack(original_noise[trial_start:trial_start + args.decisions])
            independent_noise = torch.randn(primary_noise.shape, generator=independent_rng,
                                             device=model.dev) * model.params.noise_std
            noises = {"direction_noise": primary_noise, "independent_training_noise": independent_noise}
            random_direction = (torch.randint(0, 2, model.m0.shape, generator=random_rng) * 2 - 1).to(
                device=model.dev, dtype=model.m0.dtype)
            window = {"trial_start": trial_start, "cues": window_cues.tolist(),
                      "targets": targets.tolist(), "variants": {}}
            report["windows"].append(window)
            normalized_directions = {}
            for variant in args.variants:
                ctl.begin(f"trial={trial_start} variant={variant}")
                specification = VARIANTS[variant]
                model.surrogate_slope = specification["surrogate_slope"]
                model.detach_reset = specification["detach_reset"]
                model.spike = _SurrogateSpike.function(model.surrogate_slope).apply
                zero = {name: replay(model, initial, model.m0, window_cues, targets, noise,
                                     ctl=ctl, gpu_duty=args.gpu_duty)[1]
                        for name, noise in noises.items()}
                entry = {"specification": specification, "zero": zero, "proposals": []}
                window["variants"][variant] = entry
                save()
                magnitudes = model.m0.detach().clone().requires_grad_(True)
                loss, forward = replay(model, initial, magnitudes, window_cues, targets,
                                       primary_noise, differentiate=True, ctl=ctl,
                                       gpu_duty=args.gpu_duty)
                if forward != zero["direction_noise"]:
                    raise RuntimeError("differentiated and no-grad forward replays disagree")
                gradient, = torch.autograd.grad(loss, magnitudes)
                entry["gradient"] = model._audit_tensor(gradient)
                try:
                    direction, rms = relative_direction(model.m0, gradient)
                except FloatingPointError as exc:
                    entry["status"] = "unusable_gradient"
                    entry["failure"] = str(exc)
                    save()
                    ctl.done({"trial_start": trial_start, "variant": variant,
                              "status": entry["status"]})
                    del loss, gradient, magnitudes
                    continue
                entry["relative_gradient_rms"] = rms
                normalized_directions[variant] = direction.detach().cpu()
                del loss, gradient, magnitudes
                for label, vector in (("negative_gradient", -direction),
                                      ("positive_gradient", direction),
                                      ("random", random_direction)):
                    for alpha in args.steps:
                        requested = alpha
                        if label == "random" and args.match_random_rms:
                            _, negative_displacement = propose_weights(model.m0, -direction, alpha)
                            requested = negative_displacement["realized_relative_rms"]
                        weights, displacement = propose_weights(model.m0, vector, requested)
                        proposal = {"direction": label, "comparison_step": alpha,
                                    **displacement, "replays": {}}
                        for name, noise in noises.items():
                            metrics = replay(model, initial, weights, window_cues, targets, noise,
                                             ctl=ctl, gpu_duty=args.gpu_duty)[1]
                            proposal["replays"][name] = compare_to_zero(metrics, zero[name])
                        entry["proposals"].append(proposal)
                        save()
                entry["status"] = "complete"
                save()
                ctl.done({"trial_start": trial_start, "variant": variant,
                          "relative_gradient_rms": rms})
            names = list(normalized_directions)
            window["relative_direction_cosines"] = {
                f"{a}|{b}": float(torch.nn.functional.cosine_similarity(
                    normalized_directions[a].double(), normalized_directions[b].double(), dim=0))
                for i, a in enumerate(names) for b in names[i + 1:]}
            save()
        report["complete"] = True
        report["status"] = "complete"
        if model.dev.type == "cuda":
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(model.dev) / 2**20
            report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(model.dev) / 2**20
        report["seconds"] = time.monotonic() - started
        save()
        ctl.close("complete")
        manifest.finish("complete")
    except StopRequested:
        report["status"] = "stopped"
        save()
        ctl.close("stopped")
        manifest.finish("stopped")
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        save()
        ctl.close("failed")
        manifest.finish("failed", error=report["failure"])
        raise
    print(json.dumps({"report": str(output), "status": report["status"],
                      "gate_verdict": None}, indent=2))


if __name__ == "__main__":
    main()
