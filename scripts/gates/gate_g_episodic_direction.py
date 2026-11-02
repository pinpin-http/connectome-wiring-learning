"""Supervised one-step motor-weight control on the observable episodic task."""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_g_bptt_direction import propose_weights, relative_direction, replay
from gate_g_task import setup

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.bptt import SurrogateBPTT
from flymario.training.plastic import PlasticLIF
from flymario.training.two_cue import streams

STEPS = (0.0, .03, .1, .3, 1.0)
GRAD_SEEDS = (101, 102)
SELECT_SEEDS = tuple(range(125, 137))
PROBE_SEEDS = tuple(range(137, 161))


def atomic_json(path, value):
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False))
    part.replace(path)


def evaluate(S, rows, candidates, seed_sets, ctl, label):
    import torch

    M, B = len(S['motor']), len(seed_sets)
    noise_seeds = [streams(s, 'train', 1)['noise_seed'] for s in seed_sets]
    brain = PlasticLIF(S['W'], S['motor'], LIFParams(), noise_seeds,
                       cache_key=('gate_g_episodic_direction', len(seed_sets)))
    cue_cur = torch.as_tensor(S['cues'], dtype=torch.float32, device=brain.dev)
    if not torch.equal(brain.m0, candidates[0]['m0']):
        raise AssertionError('plastic-edge order differs between engines')
    beta = float(np.exp(-brain.p.dt_ms / 5.0))
    warm = int(config.get('dynamics.warmup_ticks'))
    tpd = int(config.get('timing.ticks_per_decision'))
    results = {}
    for j, candidate in enumerate(candidates):
        brain.set_weights(candidate['weights'][None, :].expand(B, -1).contiguous())
        actions, margins, losses = [], [], []
        start = time.monotonic()
        for cue in (0, 1):
            brain.reset(noise_seeds)
            trace = torch.zeros((M, B), device=brain.dev)
            current = cue_cur[cue]
            for tick in range(warm + tpd):
                motor_spikes = brain.step(current).index_select(0, brain.motor).float()
                trace.mul_(beta).add_(motor_spikes)
                if tick % 50 == 0:
                    ctl.sub(2*j + cue + tick/(warm+tpd), 2*len(candidates), label)
            scores = torch.as_tensor(rows, device=brain.dev) @ trace
            target = cue ^ candidate['mapping']
            action = (scores[1] > scores[0]).int().cpu().numpy()
            target_margin = (scores[target] - scores[1-target]).cpu().numpy()
            actions.append(action)
            margins.append(target_margin)
            losses.append(np.logaddexp(0, -target_margin))
        torch.cuda.synchronize(brain.dev)
        elapsed = time.monotonic() - start
        time.sleep(elapsed * (1/.6 - 1))
        correct = [int(np.sum(a == (c ^ candidate['mapping'])))
                   for c, a in enumerate(actions)]
        results[candidate['name']] = {
            'accuracy': float(sum(correct)/(2*B)),
            'loss': float(np.concatenate(losses).mean()),
            'mean_target_margin': float(np.concatenate(margins).mean()),
            'cue_counts': {str(c): {'n': B, 'correct': correct[c],
                                   'left': int(np.sum(actions[c] == 1))}
                           for c in (0, 1)},
            'movement': candidate['movement'],
        }
    return results


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('declared GPU unavailable')
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.cuda.reset_peak_memory_stats()
    run_id = 'gate_g_episodic_direction_k25-20260927'
    output = RUNS / f'{run_id}.json'
    if output.exists():
        raise FileExistsError(output)
    S = setup()
    readout_seed = block_seeds(3)[0]
    full = FixedRandomProjection(len(S['motor']), len(ACTION_NAMES), readout_seed).W
    rows = np.asarray(full[[ACTION_NAMES.index('RIGHT'), ACTION_NAMES.index('LEFT')]])
    design = {'task': 'episodic reset before each cue',
              'gradient_seeds': GRAD_SEEDS, 'selection_seeds': SELECT_SEEDS,
              'probe_seeds': PROBE_SEEDS, 'purpose': 'train',
              'surrogate_slope': 25, 'detach_reset': True,
              'state_gradient_cap': 1e9, 'relative_rms_steps': STEPS,
              'readout_seed': readout_seed, 'warmup_ticks': 150,
              'ticks_per_decision': 16, 'gpu_memory_fraction': .5,
              'gpu_duty': .6}
    manifest = RunManifest(run_id, 'Gate G', 'episodic supervised motor direction',
                           design, seeds={'gradient': GRAD_SEEDS,
                                          'selection': SELECT_SEEDS,
                                          'probe': PROBE_SEEDS},
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(), 'script': Path(__file__),
                                        'bptt': ROOT / 'src/flymario/training/bptt.py',
                                        'plastic': ROOT / 'src/flymario/training/plastic.py',
                                        'direction': ROOT / 'scripts/gates/gate_g_bptt_direction.py',
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'pair': RUNS / 'gate_g_pair.json'},
                           output_files={'report': output})
    report = {'what': 'supervised episodic internal-weight direction pilot',
              'complete': False, 'gate_g_pass': False, 'design': design,
              'stages': {}, 'manifest': str(manifest.path),
              'utc': datetime.now(UTC).isoformat(timespec='seconds')}
    save = lambda: atomic_json(output, report)
    save()
    ctl = RunControl(run_id, total=3, checkpoint=save, unit='stage',
                     meta={'question': 'Can the fixed readout be steered by internal synapses?'})
    ctl.chart('accuracy', 'Episodic action accuracy', 'relative RMS request',
              'correct fraction', ['mapping0', 'mapping1'], yfmt='pct',
              refs=[{'axis': 'y', 'value': .5, 'label': 'chance'},
                    {'axis': 'y', 'value': .75, 'label': 'pilot target'}],
              note='Each point uses an independent noise stream after a fresh reset.')
    try:
        model = SurrogateBPTT(S['W'], S['motor'], S['cues'], rows, LIFParams(),
                              seed=0, block_len=1, surrogate_slope=25,
                              detach_reset=True)
        model.state_grad_clip_norm = 1e9
        ctl.begin('four reset-episode surrogate directions')
        direction_sum = torch.zeros_like(model.m0)
        gradient_log = []
        for seed_set in GRAD_SEEDS:
            for cue in (0, 1):
                model.noise_rng.manual_seed(streams(seed_set, 'train', 1)['noise_seed'])
                model.state = model._zero_state()
                model.warm_done = False
                model.warm_up(cue, 150, gpu_duty=.6)
                noise = model._noise(16)
                magnitudes = model.m0.detach().clone().requires_grad_(True)
                loss, metrics = replay(model, model.state, magnitudes,
                                       [cue], [cue], noise[None],
                                       differentiate=True, gpu_duty=.6)
                gradient, = torch.autograd.grad(loss, magnitudes)
                unit, rms = relative_direction(model.m0, gradient)
                direction_sum.sub_(unit)
                gradient_log.append({'seed': seed_set, 'cue': cue,
                                     'loss': metrics['loss'], 'relative_gradient_rms': rms})
                ctl.sub(len(gradient_log), 4, 'gradient episode')
        rms = direction_sum.square().mean().sqrt()
        if not bool(torch.isfinite(rms)) or float(rms) <= 0:
            raise FloatingPointError('aggregate relative gradient is zero/nonfinite')
        direction = direction_sum / rms
        m0 = model.m0.detach()
        report['stages']['gradient'] = {'episodes': gradient_log,
                                        'direction_sum_rms': float(rms)}
        save()
        ctl.done({'gradient_episodes': 4})

        ctl.begin('select projected step on independent episodes')
        candidates = []
        for mapping in (0, 1):
            for step in STEPS:
                if step == 0:
                    weights, movement = m0.clone(), {'requested_relative_rms': 0.,
                                                      'realized_relative_rms': 0.}
                else:
                    weights, movement = propose_weights(
                        m0, direction if mapping == 0 else -direction, step)
                candidates.append({'name': f'm{mapping}_step{step:g}',
                                   'mapping': mapping, 'weights': weights,
                                   'movement': movement, 'm0': m0})
        selection = evaluate(S, rows, candidates, SELECT_SEEDS, ctl, 'step selection')
        chosen = {}
        for mapping in (0, 1):
            keys = [f'm{mapping}_step{s:g}' for s in STEPS]
            chosen[str(mapping)] = max(keys, key=lambda k: (
                selection[k]['accuracy'], -selection[k]['loss'],
                -float(k.split('step')[1])))
            for step in STEPS:
                ctl.point('accuracy', f'mapping{mapping}', step,
                          selection[f'm{mapping}_step{step:g}']['accuracy'])
        report['stages']['selection'] = {'scores': selection, 'chosen': chosen}
        save()
        ctl.done({'chosen': chosen})

        ctl.begin('confirm against frozen and matched random')
        rng = torch.Generator(device=model.dev).manual_seed(20260927917)
        controls = []
        for mapping in (0, 1):
            name = chosen[str(mapping)]
            selected = next(c for c in candidates if c['name'] == name)
            controls.append({**selected, 'name': f'm{mapping}_candidate'})
            controls.append({'name': f'm{mapping}_frozen', 'mapping': mapping,
                             'weights': m0.clone(), 'm0': m0,
                             'movement': {'realized_relative_rms': 0.}})
            random_direction = (torch.randint(0, 2, m0.shape,
                                              generator=rng, device=model.dev)*2-1).float()
            random_weights, movement = propose_weights(
                m0, random_direction,
                max(selected['movement']['realized_relative_rms'], 1e-12))
            controls.append({'name': f'm{mapping}_random', 'mapping': mapping,
                             'weights': random_weights, 'm0': m0,
                             'movement': movement})
        probe = evaluate(S, rows, controls, PROBE_SEEDS, ctl, 'fresh confirmation')
        se = max(float(np.sqrt(max(v['accuracy']*(1-v['accuracy']), 1e-12) /
                               (2*len(PROBE_SEEDS)))) for v in probe.values())
        checks = {'both_mappings_at_least_0.75': all(
            probe[f'm{m}_candidate']['accuracy'] >= .75 for m in (0, 1)),
                  'beats_frozen_by_2se': all(
            probe[f'm{m}_candidate']['accuracy']-probe[f'm{m}_frozen']['accuracy'] > 2*se
            for m in (0, 1)),
                  'beats_random_by_2se': all(
            probe[f'm{m}_candidate']['accuracy']-probe[f'm{m}_random']['accuracy'] > 2*se
            for m in (0, 1))}
        report['stages']['probe'] = {'scores': probe, 'largest_binomial_se': se,
                                     'checks': checks, 'provisional_pass': all(checks.values())}
        report['complete'] = True
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        save()
        ctl.done({'provisional_pass': all(checks.values())})
        ctl.close('complete')
        manifest.finish('complete')
        print(json.dumps({'chosen': chosen, 'checks': checks,
                          'probe': {k: v['accuracy'] for k, v in probe.items()}}, indent=2))
    except StopRequested:
        save()
        ctl.close('stopped')
        manifest.finish('stopped')
    except BaseException as exc:
        save()
        ctl.close('failed')
        manifest.finish('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
