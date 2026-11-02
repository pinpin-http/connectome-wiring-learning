#!/usr/bin/env python
"""No-update motor-cue decodability in continuous Gate G dynamics."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from gate_g_task import accuracy, setup

from flymario import config
from flymario.paths import RUNS
from flymario.policy.readout import block_seeds
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.two_cue import Learner, TwoCue

N = 200
TRAIN_SET, VALIDATION_SET = 3, 4
PERMUTATION_SEED = 20260927999


def main():
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu-duty', type=float, default=0.6)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.50)
    parser.add_argument('--ticks-per-decision', type=int,
                        default=int(config.get('timing.ticks_per_decision')))
    args = parser.parse_args()
    if not args.run_tag or not all(c.isalnum() or c in '-_' for c in args.run_tag):
        parser.error('invalid run tag')
    if not 0 < args.gpu_duty <= 1:
        parser.error('GPU duty must be in (0,1]')
    if not 0 < args.gpu_memory_fraction <= 0.5:
        parser.error('GPU memory fraction must be in (0,.5]')
    if args.ticks_per_decision <= 0:
        parser.error('ticks per decision must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('the configured GPU is unavailable')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)

    run_id = f'gate_g_continuous_decode_{args.run_tag}'
    output = RUNS / f'{run_id}.json'
    checkpoint = RUNS / 'checkpoints' / f'{run_id}.pt'
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    S = setup()
    readout_seed = block_seeds(3)[0]
    learners = [Learner(f'frozen_set{s}', s, readout_seed=readout_seed)
                for s in (TRAIN_SET, VALIDATION_SET)]
    run = TwoCue(S['W'], S['motor'], S['cues'], learners, 'train', N,
                 record=True, cache_key=('gate_g_continuous_decode', 0))
    run.tpd = args.ticks_per_decision
    if checkpoint.exists():
        run.load(checkpoint)
    config_record = {'seed_sets': [TRAIN_SET, VALIDATION_SET], 'purpose': 'train',
                     'readout_seed': readout_seed, 'decisions': N,
                     'warmup_ticks': int(config.get('dynamics.warmup_ticks')),
                     'ticks_per_decision': args.ticks_per_decision,
                     'permutation_seed': PERMUTATION_SEED,
                     'decoder': 'StandardScaler + LogisticRegression(C=.05, random_state=0)',
                     'gpu_memory_fraction': args.gpu_memory_fraction,
                     'gpu_duty': args.gpu_duty}
    manifest = RunManifest(run_id, 'Gate G', 'continuous frozen motor decodability',
                           config_record, seeds=config_record['seed_sets'],
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(),
                                        'script': Path(__file__),
                                        'two_cue': ROOT / 'src/flymario/training/two_cue.py',
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'config': ROOT / 'configs/base.yaml',
                                        'pair': RUNS / 'gate_g_pair.json'},
                           output_files={'report': output, 'traces': checkpoint})
    ctl = RunControl(run_id, total=4, unit='50 decisions',
                     checkpoint=lambda: run.save(checkpoint),
                     already_done=run.trial//50, fresh=not args.resume,
                     meta={'question': 'Do continuous frozen motor traces retain the cue?'})
    try:
        while run.trial < N:
            ctl.begin(f'decisions {run.trial}-{run.trial + 50}')
            run.run(50, ctl=ctl, gpu_duty=args.gpu_duty)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            run.save(checkpoint)
            ctl.done({'completed_decisions': run.trial})
        # The trace contains decayed float32 subnormals (down to ~1e-45).
        # StandardScaler computes nonzero ~1e-46 scales in float64, but
        # transforms float32 by casting those scales to zero. Convert before
        # fitting and transforming to preserve the finite measured features.
        Xtr = run.features[:, :, 0].astype(np.float64)
        Xte = run.features[:, :, 1].astype(np.float64)
        ytr = run.cues[0].cpu().numpy().astype(int)
        yte = run.cues[1].cpu().numpy().astype(int)
        scale = StandardScaler().fit(Xtr)
        train_scaled, test_scaled = scale.transform(Xtr), scale.transform(Xte)
        true_decoder = LogisticRegression(C=.05, max_iter=2000,
                                          random_state=0).fit(train_scaled, ytr)
        shuffled = np.random.default_rng(PERMUTATION_SEED).permutation(ytr)
        negative_decoder = LogisticRegression(C=.05, max_iter=2000,
                                              random_state=0).fit(train_scaled, shuffled)
        rep = {'what': 'continuous frozen motor cue observability',
               'complete': True, 'gate_g_pass': False,
               'scope': 'cheating decoder, no updates, training-purpose streams',
               'configuration': config_record,
               'cue_decode_train': float(true_decoder.score(train_scaled, ytr)),
               'cue_decode_independent': float(true_decoder.score(test_scaled, yte)),
               'shuffled_label_decode_independent': float(
                   negative_decoder.score(test_scaled, yte)),
               'motor_trace_variance_mean_train': float(Xtr.var(axis=0).mean()),
               'fixed_readout': {f'set{s}': {f'mapping{m}':
                   accuracy(run.features[:, :, i],
                            run.cues[i].cpu().numpy().astype(int), readout_seed, m)
                   for m in (0, 1)} for i, s in enumerate((TRAIN_SET, VALIDATION_SET))},
               'manifest': str(manifest.path),
               'traces_checkpoint': str(checkpoint),
               'utc': datetime.now(UTC).isoformat(timespec='seconds')}
        temporary = output.with_suffix('.json.part')
        temporary.write_text(json.dumps(rep, indent=2, allow_nan=False))
        temporary.replace(output)
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
