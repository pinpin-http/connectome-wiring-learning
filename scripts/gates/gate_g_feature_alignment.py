#!/usr/bin/env python
"""Compare continuous motor spike counts and fixed-readout trace on paired draws."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from gate_g_task import setup

from flymario import config
from flymario.paths import RUNS
from flymario.policy.readout import block_seeds
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.two_cue import Learner, TwoCue

N, TRAIN_SET, PROBE_SET = 200, 3, 4
SHUFFLE_SEED = 20260927999


def decode(Xtr, ytr, Xte, yte, shuffled=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr = np.asarray(Xtr, np.float64)
    Xte = np.asarray(Xte, np.float64)
    scaler = StandardScaler().fit(Xtr)
    labels = (np.random.default_rng(SHUFFLE_SEED).permutation(ytr)
              if shuffled else ytr)
    classifier = LogisticRegression(C=.05, max_iter=2000,
                                    random_state=0).fit(scaler.transform(Xtr), labels)
    return {'train_accuracy': float(classifier.score(scaler.transform(Xtr), labels)),
            'probe_accuracy': float(classifier.score(scaler.transform(Xte), yte))}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu-duty', type=float, default=.6)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.5)
    args = parser.parse_args()
    if not args.run_tag or not all(c.isalnum() or c in '-_' for c in args.run_tag):
        parser.error('invalid run tag')
    if not 0 < args.gpu_duty <= 1 or not 0 < args.gpu_memory_fraction <= .5:
        parser.error('invalid GPU duty or allocator fraction')
    if not torch.cuda.is_available():
        raise RuntimeError('the declared GPU is unavailable')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    run_id = f'gate_g_feature_alignment_{args.run_tag}'
    output = RUNS / f'{run_id}.json'
    checkpoint = RUNS / 'checkpoints' / f'{run_id}.pt'
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    S = setup()
    readout = block_seeds(3)[0]
    learners = [Learner(f'frozen_set{s}', s, readout_seed=readout)
                for s in (TRAIN_SET, PROBE_SET)]
    run = TwoCue(S['W'], S['motor'], S['cues'], learners, 'train', N,
                 record=True, cache_key=('gate_g_feature_alignment', 0))
    counts = np.zeros((N, len(S['motor']), 2), np.float32)
    if checkpoint.exists():
        snapshot = torch.load(checkpoint, map_location=run.dev, weights_only=False)
        run.load_state_dict(snapshot['run'])
        counts[:] = snapshot['counts']
    def save_checkpoint():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        part = checkpoint.with_suffix('.pt.part')
        torch.save({'run': run.state_dict(), 'counts': counts}, part)
        part.replace(checkpoint)
    design = {'training_seed': TRAIN_SET, 'probe_seed': PROBE_SET,
              'purpose': 'train', 'decisions': N,
              'ticks_per_decision': int(config.get('timing.ticks_per_decision')),
              'warmup_ticks': int(config.get('dynamics.warmup_ticks')),
              'readout_seed': readout, 'shuffle_seed': SHUFFLE_SEED,
              'decoder': 'StandardScaler(float64), LogisticRegression(C=.05, random_state=0)',
              'gpu_duty': args.gpu_duty,
              'gpu_memory_fraction': args.gpu_memory_fraction}
    manifest = RunManifest(run_id, 'Gate G', 'paired frozen motor features',
                           design, seeds=[TRAIN_SET, PROBE_SET],
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(),
                                        'script': Path(__file__),
                                        'two_cue': ROOT / 'src/flymario/training/two_cue.py',
                                        'config': ROOT / 'configs/base.yaml',
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'pair': RUNS / 'gate_g_pair.json'},
                           output_files={'report': output, 'checkpoint': checkpoint})
    ctl = RunControl(run_id, total=4, already_done=run.trial // 50,
                     fresh=not args.resume, unit='50 decisions',
                     checkpoint=save_checkpoint,
                     meta={'question': 'Does a 16-tick spike count keep cue information lost by the trace?'})
    try:
        run.run(0, gpu_duty=args.gpu_duty)
        accumulator = torch.zeros((len(S['motor']), 2), device=run.dev)
        original_tick = run._tick
        def capture_tick(current):
            motor_spikes = original_tick(current)
            accumulator.add_(motor_spikes)
            return motor_spikes
        run._tick = capture_tick
        while run.trial < N:
            start = run.trial
            ctl.begin(f'decisions {start}-{min(start + 50, N)}')
            for _ in range(min(50, N - start)):
                accumulator.zero_()
                run.run(1, ctl=ctl, gpu_duty=args.gpu_duty)
                counts[run.trial - 1] = accumulator.cpu().numpy()
            save_checkpoint()
            ctl.done({'completed_decisions': run.trial})
        ytr, yte = [run.cues[i].cpu().numpy().astype(int) for i in (0, 1)]
        rep = {'what': 'paired continuous cue observability by motor feature',
               'complete': True, 'gate_g_pass': False,
               'scope': 'no update, training-purpose streams; decoder is cheating',
               'design': design,
               'trace': {'true_labels': decode(run.features[:,:,0], ytr,
                                              run.features[:,:,1], yte),
                         'shuffled_labels': decode(run.features[:,:,0], ytr,
                                                   run.features[:,:,1], yte, True)},
               'spike_count': {'true_labels': decode(counts[:,:,0], ytr,
                                                    counts[:,:,1], yte),
                               'shuffled_labels': decode(counts[:,:,0], ytr,
                                                         counts[:,:,1], yte, True)},
               'manifest': str(manifest.path), 'checkpoint': str(checkpoint),
               'utc': datetime.now(UTC).isoformat(timespec='seconds')}
        part = output.with_suffix('.json.part')
        part.write_text(json.dumps(rep, indent=2, allow_nan=False))
        part.replace(output)
        ctl.close('complete')
        manifest.finish('complete')
        print(json.dumps(rep, indent=2))
    except StopRequested:
        ctl.close('stopped')
        manifest.finish('stopped')
    except BaseException as exc:
        ctl.close('failed')
        manifest.finish('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
