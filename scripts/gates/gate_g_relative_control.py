#!/usr/bin/env python
"""Exploratory supervised motor-weight control; no Gate G learning-rule claim.

One averaged relative-coordinate surrogate direction is selected on frozen
training prefixes. Step size is selected on an independent training stream and
probed on a third, continuous stream. All hard-spike dynamics and anatomical
constraints are the actual TwoCue implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gate_g_bptt_direction import propose_weights, relative_direction, replay
from gate_g_task import setup

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.run_manifest import RunManifest, dataset_files, sha256_file
from flymario.runctl import RunControl, StopRequested
from flymario.training.bptt import SurrogateBPTT
from flymario.training.two_cue import Learner, TwoCue, streams

STEPS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
GRADIENT_TRIALS = 80
EVALUATION_TRIALS = 200
TRAINING_SET, SELECTION_SET, PROBE_SET = 2, 3, 4


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False))
    part.replace(path)


def weight_at_step(m0, direction, step: float):
    if step == 0:
        return m0.clone(), {"requested_relative_rms": 0.0,
                            "realized_relative_rms": 0.0,
                            "projected_fraction": 0.0}
    return propose_weights(m0, direction, step)


def summaries(run: TwoCue, learners: list[Learner]) -> dict:
    values = {}
    for index, learner in enumerate(learners):
        cues = run.cues[index].cpu().numpy().astype(int)
        target = cues ^ learner.mapping
        margin = run.log['margin'][:, index]
        target_margin = np.where(target == 0, margin, -margin)
        action = run.log['action'][:, index].astype(int)
        loss = np.logaddexp(0, -target_margin)
        values[learner.name] = {
            'accuracy': float(run.log['correct'][:, index].mean()),
            'loss': float(loss.mean()),
            'mean_target_margin': float(target_margin.mean()),
            'cue_counts': {str(c): {'n': int((cues == c).sum()),
                                   'left': int(((cues == c) & (action == 1)).sum()),
                                   'correct': int(((cues == c) & (action == target)).sum())}
                           for c in (0, 1)},
        }
    return values


def run_continuous(S: dict, seed_set: int, candidates: dict,
                   name: str, ctl: RunControl, checkpoint: Path, gpu_duty: float):
    """A resumable fixed-weight TwoCue simulation on one complete stream."""
    import torch
    labels = list(candidates)
    learners = [Learner(label, seed_set, mapping=options['mapping'],
                        readout_seed=block_seeds(3)[0])
                for label, options in candidates.items()]
    run = TwoCue(S['W'], S['motor'], S['cues'], learners, 'train',
                 EVALUATION_TRIALS, cache_key=('gate_g_relative_control', name))
    weights = torch.stack([candidates[label]['weights'] for label in labels])
    if run.brain.m0.shape != weights.shape[1:]:
        raise AssertionError('plastic-edge shape changed')
    if not torch.equal(run.brain.m0, candidates[labels[0]]['m0']):
        raise AssertionError('plastic-edge order differs between BPTT and TwoCue')
    run.brain.set_weights(weights)
    if checkpoint.exists():
        run.load(checkpoint)
    while run.trial < EVALUATION_TRIALS:
        run.run(min(50, EVALUATION_TRIALS - run.trial), ctl=ctl,
                gpu_duty=gpu_duty)
        run.save(checkpoint)
    return summaries(run, learners)


def main() -> None:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu-duty', type=float, default=0.6)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.50)
    args = parser.parse_args()
    if not args.run_tag or not all(x.isalnum() or x in '-_' for x in args.run_tag):
        parser.error('run tag must consist of letters, digits, hyphens, underscores')
    if not 0 < args.gpu_duty <= 1 or not 0 < args.gpu_memory_fraction <= 0.5:
        parser.error('GPU duty in (0,1] and memory fraction in (0,.5] required')
    if not torch.cuda.is_available():
        raise RuntimeError('the declared GPU is unavailable')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()

    run_id = f'gate_g_relative_control_{args.run_tag}'
    output = RUNS / f'{run_id}.json'
    ckpt_dir = RUNS / 'checkpoints' / run_id
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; choose a new tag or --resume')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    files = {'control': Path(__file__),
             'bptt': ROOT / 'src/flymario/training/bptt.py',
             'plastic': ROOT / 'src/flymario/training/plastic.py',
             'two_cue': ROOT / 'src/flymario/training/two_cue.py',
             'direction': ROOT / 'scripts/gates/gate_g_bptt_direction.py',
             'config': ROOT / 'configs/base.yaml',
             'protocol': ROOT / 'docs/GATE_G_PLAN.md'}
    hashes = {k: sha256_file(v) for k, v in files.items()}
    design = {'training_seed_set': TRAINING_SET, 'selection_seed_set': SELECTION_SET,
              'probe_seed_set': PROBE_SET, 'training_decisions': GRADIENT_TRIALS,
              'selection_decisions': EVALUATION_TRIALS,
              'probe_decisions': EVALUATION_TRIALS, 'steps': list(STEPS),
              'mapping': [0, 1], 'readout_seed': block_seeds(3)[0],
              'surrogate_slope': 25, 'detach_reset': True,
              'state_gradient_cap': 1e9,
              'gpu_memory_fraction': args.gpu_memory_fraction,
              'gpu_duty': args.gpu_duty, 'source_hashes': hashes}
    if output.exists():
        report = json.loads(output.read_text())
        if report['design'] != design:
            raise RuntimeError('source or design changed; use a new run tag')
    else:
        report = {'what': 'exploratory supervised internal-weight control',
                  'complete': False, 'gate_g_pass': False,
                  'scope': 'one readout; training-purpose streams only; not a local rule',
                  'design': design, 'stages': {},
                  'utc': datetime.now(UTC).isoformat(timespec='seconds')}
    def save() -> None:
        atomic_json(output, report)
    save()
    manifest = RunManifest(run_id, 'Gate G', 'exploratory supervised motor control',
                           design, seeds={'training': TRAINING_SET,
                                          'selection': SELECTION_SET,
                                          'probe': PROBE_SET},
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(), **files,
                                        'cue_pair': RUNS / 'gate_g_pair.json'},
                           output_files={'report': output})
    report['manifest'] = str(manifest.path)
    save()
    ctl = RunControl(run_id, total=3, checkpoint=save, unit='stage',
                     already_done=len(report['stages']), fresh=not args.resume,
                     meta={'question': 'Can bounded internal weights control the fixed readout?'})
    start = time.monotonic()
    try:
        S = setup()
        stream = streams(TRAINING_SET, 'train', GRADIENT_TRIALS)
        rows = FixedRandomProjection(len(S['motor']), len(ACTION_NAMES),
                                     block_seeds(3)[0]).W
        rows = rows[[ACTION_NAMES.index('RIGHT'), ACTION_NAMES.index('LEFT')]]
        model = SurrogateBPTT(S['W'], S['motor'], S['cues'], rows,
                              LIFParams(), stream['noise_seed'],
                              block_len=1, surrogate_slope=25,
                              detach_reset=True)
        model.state_grad_clip_norm = 1e9
        gradient_ckpt = ckpt_dir / 'gradient.pt'
        if gradient_ckpt.exists():
            snapshot = torch.load(gradient_ckpt, map_location='cpu', weights_only=True)
            state = tuple(x.to(model.dev) for x in snapshot['state'])
            model.noise_rng.set_state(snapshot['noise_rng_state'])
            direction_sum = snapshot['direction_sum'].to(model.dev)
            trial_start = int(snapshot['trial'])
        else:
            model.warm_up(int(stream['cues'][0]),
                          int(config.get('dynamics.warmup_ticks')),
                          gpu_duty=args.gpu_duty)
            state = tuple(x.detach().clone() for x in model.state)
            direction_sum = torch.zeros_like(model.m0)
            trial_start = 0
        if 'gradient' not in report['stages']:
            ctl.begin('80 frozen-state surrogate gradients')
            frozen_matrix = model._motor_matrix(model.m0)
            tpd = int(config.get('timing.ticks_per_decision'))
            for trial in range(trial_start, GRADIENT_TRIALS):
                cue = int(stream['cues'][trial])
                noise = model._noise(tpd)
                magnitudes = model.m0.detach().clone().requires_grad_(True)
                loss, _ = replay(model, state, magnitudes, [cue], [cue],
                                 noise[None, ...], differentiate=True,
                                 gpu_duty=args.gpu_duty)
                gradient, = torch.autograd.grad(loss, magnitudes)
                unit, _ = relative_direction(model.m0, gradient)
                direction_sum.sub_(unit)
                del loss, gradient, magnitudes
                with torch.no_grad():
                    for xi in noise:
                        state = model._tick(*state, model.cues[:, cue], xi,
                                            frozen_matrix)
                if (trial + 1) % 10 == 0 or trial + 1 == GRADIENT_TRIALS:
                    partial = gradient_ckpt.with_suffix('.pt.part')
                    torch.save({'state': tuple(x.detach().cpu() for x in state),
                                'noise_rng_state': model.noise_rng.get_state(),
                                'direction_sum': direction_sum.detach().cpu(),
                                'trial': trial + 1}, partial)
                    partial.replace(gradient_ckpt)
                ctl.sub(trial + 1, GRADIENT_TRIALS,
                        'gradient decisions (checkpoint every 10)')
            ctl.done({'gradients': GRADIENT_TRIALS})
            report['stages']['gradient'] = {'decisions': GRADIENT_TRIALS,
                                            'cue_counts': np.bincount(
                                                stream['cues'], minlength=2).tolist()}
            save()
        direction = direction_sum / direction_sum.square().mean().sqrt()
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError('aggregated relative direction is non-finite')
        m0 = model.m0.detach()
        if 'selection' not in report['stages']:
            ctl.begin('continuous training-only step selection')
            candidates = {}
            for mapping in (0, 1):
                for step in STEPS:
                    label = f'm{mapping}_step{step:g}'
                    weights, movement = weight_at_step(
                        m0, direction if mapping == 0 else -direction, step)
                    candidates[label] = {'mapping': mapping, 'weights': weights,
                                         'movement': movement, 'm0': m0}
            validation = run_continuous(S, SELECTION_SET, candidates,
                                        'selection', ctl,
                                        ckpt_dir / 'selection.pt', args.gpu_duty)
            chosen = {}
            for mapping in (0, 1):
                labels = [f'm{mapping}_step{step:g}' for step in STEPS]
                best = max(labels, key=lambda key: (validation[key]['accuracy'],
                                                   -validation[key]['loss'],
                                                   -float(key.split('step')[1])))
                chosen[str(mapping)] = best
            report['stages']['selection'] = {
                'scores': validation, 'chosen': chosen,
                'movements': {k: v['movement'] for k, v in candidates.items()}}
            save()
            ctl.done({'chosen': chosen})
        chosen = report['stages']['selection']['chosen']
        if 'probe' not in report['stages']:
            ctl.begin('independent continuous training-purpose probe')
            random_rng = torch.Generator(device=model.dev).manual_seed(202609274)
            candidates = {}
            for mapping in (0, 1):
                step = float(chosen[str(mapping)].split('step')[1])
                weights, movement = weight_at_step(
                    m0, direction if mapping == 0 else -direction, step)
                candidates[f'm{mapping}_candidate'] = {
                    'mapping': mapping, 'weights': weights,
                    'movement': movement, 'm0': m0}
                candidates[f'm{mapping}_frozen'] = {
                    'mapping': mapping, 'weights': m0.clone(),
                    'movement': weight_at_step(m0, direction, 0)[1], 'm0': m0}
                random_direction = (torch.randint(0, 2, m0.shape, generator=random_rng,
                                                  device=model.dev) * 2 - 1).to(m0.dtype)
                random_weights, random_movement = weight_at_step(
                    m0, random_direction, movement['realized_relative_rms'])
                candidates[f'm{mapping}_random'] = {
                    'mapping': mapping, 'weights': random_weights,
                    'movement': random_movement, 'm0': m0}
            probe = run_continuous(S, PROBE_SET, candidates, 'probe', ctl,
                                   ckpt_dir / 'probe.pt', args.gpu_duty)
            se = max(float(np.sqrt(max(x['accuracy'] * (1-x['accuracy']), 1e-12)
                                   / EVALUATION_TRIALS)) for x in probe.values())
            checks = {
                'both_mappings_at_least_0.75': all(
                    probe[f'm{m}_candidate']['accuracy'] >= .75 for m in (0, 1)),
                'beats_frozen_by_2se': all(
                    probe[f'm{m}_candidate']['accuracy'] -
                    probe[f'm{m}_frozen']['accuracy'] > 2*se for m in (0, 1)),
                'beats_matched_random_by_2se': all(
                    probe[f'm{m}_candidate']['accuracy'] -
                    probe[f'm{m}_random']['accuracy'] > 2*se for m in (0, 1)),
            }
            report['stages']['probe'] = {'scores': probe, 'checks': checks,
                                         'largest_binomial_se': se,
                                         'pass': all(checks.values()),
                                         'movements': {k: v['movement'] for k,v in candidates.items()}}
            save()
            ctl.done({'provisional_pass': all(checks.values())})
        report['complete'] = True
        report['seconds_this_invocation'] = time.monotonic() - start
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        save()
        ctl.close('complete')
        manifest.finish('complete')
    except StopRequested:
        save()
        ctl.close('stopped')
        manifest.finish('stopped')
        return
    except BaseException as exc:
        report['failure'] = f'{type(exc).__name__}: {exc}'
        save()
        ctl.close('failed')
        manifest.finish('failed', error=report['failure'])
        raise
    print(json.dumps({'report': str(output),
                      'provisional_positive_control': report['stages']['probe']['pass'],
                      'checks': report['stages']['probe']['checks']}, indent=2))


if __name__ == '__main__':
    main()
