"""Probe whether two cue-specific motor gradients can remove action bias."""
from __future__ import annotations

import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_g_bptt_direction import relative_direction, replay
from gate_g_episodic_direction import GRAD_SEEDS, PROBE_SEEDS, SELECT_SEEDS, atomic_json, evaluate
from gate_g_task import setup

from flymario import config
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.bptt import SurrogateBPTT
from flymario.training.two_cue import streams

COEFFICIENTS = (0., .03, .1)


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('declared GPU unavailable')
    torch.cuda.set_per_process_memory_fraction(.5)
    run_id = 'gate_g_episodic_cue_directions_grid3-20260927'
    output = RUNS / f'{run_id}.json'
    if output.exists():
        raise FileExistsError(output)
    S = setup()
    readout_seed = block_seeds(3)[0]
    full = FixedRandomProjection(len(S['motor']), len(ACTION_NAMES), readout_seed).W
    rows = np.asarray(full[[ACTION_NAMES.index('RIGHT'), ACTION_NAMES.index('LEFT')]])
    design = {'task': 'episodic reset', 'mapping': 0,
              'gradient_seeds': GRAD_SEEDS, 'selection_seeds': SELECT_SEEDS,
              'probe_seeds': PROBE_SEEDS, 'coefficients': COEFFICIENTS,
              'surrogate_slope': 25, 'detach_reset': True,
              'state_gradient_cap': 1e9, 'readout_seed': readout_seed,
              'warmup_ticks': 150, 'ticks_per_decision': 16,
              'gpu_memory_fraction': .5, 'gpu_duty': .6}
    manifest = RunManifest(run_id, 'Gate G', 'separate cue motor directions',
                           design, seeds={'gradient': GRAD_SEEDS,
                                          'selection': SELECT_SEEDS,
                                          'probe': PROBE_SEEDS},
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(), 'script': Path(__file__),
                                        'bptt': ROOT / 'src/flymario/training/bptt.py',
                                        'evaluator': ROOT / 'scripts/gates/gate_g_episodic_direction.py',
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'pair': RUNS / 'gate_g_pair.json'},
                           output_files={'report': output})
    report = {'what': 'cue-specific surrogate direction grid on reset task',
              'complete': False, 'gate_g_pass': False, 'design': design,
              'stages': {}, 'manifest': str(manifest.path),
              'utc': datetime.now(UTC).isoformat(timespec='seconds')}
    save = lambda: atomic_json(output, report)
    save()
    ctl = RunControl(run_id, total=3, checkpoint=save, unit='stage',
                     meta={'question': 'Can cue-specific directions separate the two actions?'})
    ctl.chart('selection', 'Cue-specific projected updates', 'coefficient pair',
              'correct fraction', ['mapping0'], kind='points', yfmt='pct',
              refs=[{'axis': 'y', 'value': .75, 'label': 'pilot target'}],
              note='Selection uses fresh episodic noise; confirmation uses another seed set.')
    try:
        model = SurrogateBPTT(S['W'], S['motor'], S['cues'], rows, LIFParams(),
                              seed=0, block_len=1, surrogate_slope=25,
                              detach_reset=True)
        model.state_grad_clip_norm = 1e9
        ctl.begin('cue-specific gradients')
        sums = [torch.zeros_like(model.m0), torch.zeros_like(model.m0)]
        details = []
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
                sums[cue].sub_(unit)
                details.append({'seed': seed_set, 'cue': cue,
                                'loss': metrics['loss'], 'relative_gradient_rms': rms})
                ctl.sub(len(details), 4, 'gradient episode')
        directions = [s / s.square().mean().sqrt() for s in sums]
        if not all(bool(torch.isfinite(d).all()) for d in directions):
            raise FloatingPointError('non-finite cue direction')
        cosine = float(torch.nn.functional.cosine_similarity(
            directions[0].double(), directions[1].double(), dim=0))
        m0 = model.m0.detach()
        report['stages']['gradient'] = {'episodes': details, 'direction_cosine': cosine}
        save()
        ctl.done({'direction_cosine': cosine})

        ctl.begin('nine two-cue combinations')
        candidates = []
        for a0, a1 in itertools.product(COEFFICIENTS, repeat=2):
            change = a0*directions[0] + a1*directions[1]
            weights = m0 * (1 + change).clamp(0, 2)
            relative = (weights.double() - m0.double()) / m0.double()
            movement = {'coefficients': [a0, a1],
                        'realized_relative_rms': float(relative.square().mean().sqrt()),
                        'projected_fraction': float(((1+change < 0) | (1+change > 2))
                                                    .float().mean())}
            candidates.append({'name': f'a{a0:g}_b{a1:g}', 'mapping': 0,
                               'weights': weights, 'movement': movement, 'm0': m0})
        selection = evaluate(S, rows, candidates, SELECT_SEEDS, ctl, 'grid selection')
        for name, score in selection.items():
            ctl.point('selection', 'mapping0', name, score['accuracy'])
        chosen = max(candidates, key=lambda c: (
            selection[c['name']]['accuracy'], -selection[c['name']]['loss'],
            -c['movement']['realized_relative_rms'],
            -c['movement']['coefficients'][0], -c['movement']['coefficients'][1]))
        report['stages']['selection'] = {'scores': selection,
                                         'chosen': chosen['name']}
        save()
        ctl.done({'chosen': chosen['name']})

        ctl.begin('fresh-seed confirmation')
        rng = torch.Generator(device=model.dev).manual_seed(20260927918)
        random_direction = (torch.randint(0, 2, m0.shape, device=model.dev,
                                          generator=rng)*2-1).float()
        realized = chosen['movement']['realized_relative_rms']
        random_weights = m0 * (1+realized*random_direction).clamp(0, 2)
        controls = [{**chosen, 'name': 'candidate'},
                    {'name': 'frozen', 'mapping': 0, 'weights': m0.clone(), 'm0': m0,
                     'movement': {'realized_relative_rms': 0.}},
                    {'name': 'random', 'mapping': 0, 'weights': random_weights, 'm0': m0,
                     'movement': {'realized_relative_rms': realized}}]
        probe = evaluate(S, rows, controls, PROBE_SEEDS, ctl, 'confirmation')
        se = max(float(np.sqrt(max(v['accuracy']*(1-v['accuracy']), 1e-12) /
                               (2*len(PROBE_SEEDS)))) for v in probe.values())
        checks = {'at_least_0.75': probe['candidate']['accuracy'] >= .75,
                  'beats_frozen_by_2se': probe['candidate']['accuracy'] -
                  probe['frozen']['accuracy'] > 2*se,
                  'beats_random_by_2se': probe['candidate']['accuracy'] -
                  probe['random']['accuracy'] > 2*se}
        report['stages']['probe'] = {'scores': probe, 'largest_binomial_se': se,
                                     'checks': checks, 'provisional_pass': all(checks.values())}
        report['complete'] = True
        save()
        ctl.done({'provisional_pass': all(checks.values())})
        ctl.close('complete')
        manifest.finish('complete')
        print(json.dumps({'direction_cosine': cosine, 'chosen': chosen['name'],
                          'probe': {k: v['accuracy'] for k, v in probe.items()},
                          'checks': checks}, indent=2))
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
