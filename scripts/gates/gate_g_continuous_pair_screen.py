"""Screen frozen motor traces for a transferable cue pair in continuous play."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from flymario import config
from flymario.agent import AgentConfig
from flymario.connectome import datasets
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl, StopRequested
from flymario.training.plastic import PlasticLIF
from flymario.training.two_cue import MASTER, cue_currents, streams

SEEDS = (3, 4, 5)
N = 400
SHUFFLE_SEED = 20260927999


def scene_stream(seed: int, count: int, n_trials: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, seed, 3, 20260927]))
    blocks = -(-n_trials // count)
    return np.concatenate([rng.permutation(count) for _ in range(blocks)])[:n_trials]


def fit_score(Xtrain, ytrain, Xtest, ytest, shuffled=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtrain, Xtest = np.asarray(Xtrain, np.float64), np.asarray(Xtest, np.float64)
    labels = (np.random.default_rng(SHUFFLE_SEED).permutation(ytrain)
              if shuffled else ytrain)
    scaler = StandardScaler().fit(Xtrain)
    model = LogisticRegression(C=.05, max_iter=2000, random_state=0).fit(
        scaler.transform(Xtrain), labels)
    proba = model.predict_proba(scaler.transform(Xtest))
    prediction = model.classes_[proba.argmax(axis=1)]
    true_col = np.searchsorted(model.classes_, ytest)
    return {'accuracy': float(np.mean(prediction == ytest)),
            'mean_true_probability': float(np.mean(proba[np.arange(len(ytest)), true_col])),
            'n': len(ytest)}


def pair_data(features, cues, pair):
    keep = np.isin(cues, pair)
    return features[keep], np.searchsorted(pair, cues[keep])


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
        parser.error('invalid GPU limits')
    if not torch.cuda.is_available():
        raise RuntimeError('declared GPU unavailable')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    run_id = f'gate_g_continuous_pair_screen_{args.run_tag}'
    output = RUNS / f'{run_id}.json'
    checkpoint = RUNS / 'checkpoints' / f'{run_id}.pt'
    if output.exists() and not args.resume:
        raise FileExistsError(output)

    g = datasets.load(datasets.DEFAULT)
    W = g.to_sparse()
    motor = datasets.motor_index(g, datasets.DEFAULT)
    names = json.loads((RUNS / 'gate_g_pair.json').read_text())['scenes']
    currents = cue_currents(g, names)
    sequences = np.stack([scene_stream(s, len(names), N) for s in SEEDS], axis=1)
    noise_seeds = [streams(s, 'train', N)['noise_seed'] for s in SEEDS]
    params = LIFParams(gain=float(config.get('dynamics.gain')),
                       noise_std=float(config.get('dynamics.noise_std')))
    brain = PlasticLIF(W, motor, params, seeds=noise_seeds, cache_key=(run_id, 0))
    dev = brain.dev
    cue_cur = torch.as_tensor(currents.T, device=dev, dtype=torch.float32)
    cue_index = torch.as_tensor(sequences, device=dev, dtype=torch.long)
    beta = float(np.exp(-params.dt_ms / AgentConfig().trace_tau_ms))
    trace = torch.zeros((len(motor), len(SEEDS)), device=dev)
    features = np.zeros((N, len(motor), len(SEEDS)), np.float32)
    tpd = int(config.get('timing.ticks_per_decision'))
    warm = int(config.get('dynamics.warmup_ticks'))
    trial, warm_done = 0, False
    if checkpoint.exists() and args.resume:
        snap = torch.load(checkpoint, map_location=dev, weights_only=False)
        if snap['seeds'] != SEEDS or snap['names'] != names:
            raise ValueError('checkpoint design mismatch')
        brain.load_state_dict(snap['brain'])
        trace.copy_(snap['trace'])
        features[:snap['trial']] = snap['features']
        trial, warm_done = snap['trial'], snap['warm_done']

    def save():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        part = checkpoint.with_suffix('.pt.part')
        torch.save({'brain': brain.state_dict(), 'trace': trace.clone(),
                    'features': features[:trial].copy(), 'trial': trial,
                    'warm_done': warm_done, 'seeds': SEEDS, 'names': names}, part)
        part.replace(checkpoint)

    design = {'seed_sets': SEEDS, 'purpose': 'train', 'scenes': names,
              'decisions_per_seed': N, 'balanced_block_size': len(names),
              'warmup_ticks': warm, 'ticks_per_decision': tpd,
              'feature': 'decayed motor trace',
              'selection_seed': 4, 'confirmation_seed': 5,
              'decoder': 'float64 StandardScaler, logistic C=.05',
              'shuffle_seed': SHUFFLE_SEED,
              'gpu_duty': args.gpu_duty, 'gpu_memory_fraction': args.gpu_memory_fraction}
    manifest = RunManifest(run_id, 'Gate G', 'continuous ten-scene observability',
                           design, seeds=list(SEEDS),
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(), 'script': Path(__file__),
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'config': ROOT / 'configs/base.yaml',
                                        'pair': RUNS / 'gate_g_pair.json'},
                           output_files={'report': output, 'checkpoint': checkpoint})
    ctl = RunControl(run_id, total=N // 50, already_done=trial // 50,
                     fresh=not args.resume, unit='50 decisions', checkpoint=save,
                     meta={'question': 'Does any visual cue pair remain readable on a continuous stream?'})
    try:
        def tick(cur):
            spikes = brain.step(cur).index_select(0, brain.motor).float()
            trace.mul_(beta).add_(spikes)

        if not warm_done:
            current = cue_cur.index_select(1, cue_index[0])
            for t in range(warm):
                tick(current)
                if t % 50 == 0:
                    ctl.sub(t, warm, 'warm-up')
            warm_done = True
        while trial < N:
            end = min(trial + 50, N)
            ctl.begin(f'decisions {trial}-{end}')
            while trial < end:
                current = cue_cur.index_select(1, cue_index[trial])
                start = time.monotonic()
                for _ in range(tpd):
                    tick(current)
                torch.cuda.synchronize(dev)
                elapsed = time.monotonic() - start
                if args.gpu_duty < 1:
                    time.sleep(elapsed * (1 / args.gpu_duty - 1))
                features[trial] = trace.cpu().numpy()
                trial += 1
                if trial % 10 == 0:
                    ctl.sub(trial, N, 'decision')
            save()
            ctl.done({'completed_decisions_per_seed': trial})
        rows = []
        for pair in itertools.combinations(range(len(names)), 2):
            Xtr, ytr = pair_data(features[:, :, 0], sequences[:, 0], pair)
            Xsel, ysel = pair_data(features[:, :, 1], sequences[:, 1], pair)
            score = fit_score(Xtr, ytr, Xsel, ysel)
            rows.append({'pair': [names[i] for i in pair], 'indices': pair,
                         'selection': score})
        rows.sort(key=lambda r: (-r['selection']['accuracy'],
                                 -r['selection']['mean_true_probability'], r['pair']))
        selected = rows[0]
        pair = selected['indices']
        Xtr, ytr = pair_data(features[:, :, 0], sequences[:, 0], pair)
        Xconfirm, yconfirm = pair_data(features[:, :, 2], sequences[:, 2], pair)
        confirmation = fit_score(Xtr, ytr, Xconfirm, yconfirm)
        negative = fit_score(Xtr, ytr, Xconfirm, yconfirm, shuffled=True)
        se = float(np.sqrt(.25 / confirmation['n']))
        exploratory_candidate = (confirmation['accuracy'] >= .75
                                 and abs(negative['accuracy'] - .5) <= 2 * se)
        report = {'what': 'continuous ten-scene motor-trace pair screen',
                  'complete': True, 'gate_g_pass': False, 'design': design,
                  'pair_scores': rows, 'selected_pair': selected['pair'],
                  'confirmation': confirmation, 'shuffled_confirmation': negative,
                  'chance_2se': 2 * se,
                  'exploratory_candidate': exploratory_candidate,
                  'scope': 'frozen brain; selected pair requires actual two-cue validation',
                  'manifest': str(manifest.path), 'checkpoint': str(checkpoint),
                  'utc': datetime.now(UTC).isoformat(timespec='seconds')}
        part = output.with_suffix('.json.part')
        part.write_text(json.dumps(report, indent=2, allow_nan=False))
        part.replace(output)
        ctl.close('complete')
        manifest.finish('complete')
        print(json.dumps({k: report[k] for k in ('selected_pair', 'confirmation',
                                                   'shuffled_confirmation',
                                                   'exploratory_candidate')}, indent=2))
    except StopRequested:
        ctl.close('stopped')
        manifest.finish('stopped')
    except BaseException as exc:
        ctl.close('failed')
        manifest.finish('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
