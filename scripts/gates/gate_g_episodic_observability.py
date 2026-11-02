"""Test the original visual cue pair with a reset before every episode."""
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

from gate_g_continuous_pair_screen import fit_score
from gate_g_restricted_decoder import one_feature_score

from flymario import config
from flymario.agent import AgentConfig
from flymario.connectome import datasets
from flymario.env.retro_env import ACTION_NAMES
from flymario.models.spiking_reservoir import LIFParams
from flymario.paths import RUNS
from flymario.policy.readout import FixedRandomProjection, block_seeds
from flymario.run_manifest import RunManifest, dataset_files
from flymario.runctl import RunControl
from flymario.training.plastic import PlasticLIF
from flymario.training.two_cue import cue_currents, streams

SEEDS = tuple(range(101, 125))


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('declared GPU unavailable')
    torch.cuda.set_per_process_memory_fraction(.5)
    run_id = 'gate_g_episodic_observability_fresh-seeds-20260927'
    output = RUNS / f'{run_id}.json'
    if output.exists():
        raise FileExistsError(output)
    g = datasets.load(datasets.DEFAULT)
    W = g.to_sparse()
    motor = datasets.motor_index(g, datasets.DEFAULT)
    pair_file = RUNS / 'gate_g_pair.json'
    names = json.loads(pair_file.read_text())['chosen']['pair']
    currents = cue_currents(g, names)
    noise_seeds = [streams(s, 'train', 1)['noise_seed'] for s in SEEDS]
    params = LIFParams(gain=float(config.get('dynamics.gain')),
                       noise_std=float(config.get('dynamics.noise_std')))
    brain = PlasticLIF(W, motor, params, noise_seeds, cache_key=(run_id, 0))
    dev = brain.dev
    cue_cur = torch.as_tensor(currents, dtype=torch.float32, device=dev)
    beta = float(np.exp(-params.dt_ms / AgentConfig().trace_tau_ms))
    warm = int(config.get('dynamics.warmup_ticks'))
    tpd = int(config.get('timing.ticks_per_decision'))
    M, B = len(motor), len(SEEDS)
    trace_features = np.zeros((2, B, M), np.float32)
    count_features = np.zeros_like(trace_features)
    readout = block_seeds(3)[0]
    wr = FixedRandomProjection(M, len(ACTION_NAMES), readout).W
    rows = np.asarray(wr[[ACTION_NAMES.index('RIGHT'), ACTION_NAMES.index('LEFT')]])
    design = {'task': 'episodic reset before each visual cue', 'pair': names,
              'seeds': SEEDS, 'purpose': 'train', 'fit_seeds': SEEDS[:12],
              'probe_seeds': SEEDS[12:], 'warmup_ticks': warm,
              'ticks_per_decision': tpd, 'readout_seed': readout,
              'decoder': 'float64 StandardScaler + logistic C=.05; full and one-feature',
              'shuffle_seed': 20260927999, 'gpu_memory_fraction': .5,
              'gpu_duty': .6}
    manifest = RunManifest(run_id, 'Gate G', 'episodic pair observability',
                           design, seeds=list(SEEDS),
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(), 'script': Path(__file__),
                                        'pair': pair_file,
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'config': ROOT / 'configs/base.yaml'},
                           output_files={'report': output})
    ctl = RunControl(run_id, total=2, checkpoint=lambda: None, unit='scene',
                     meta={'question': 'Does resetting between episodes restore cue information?'})
    try:
        for c in range(2):
            ctl.begin(names[c])
            brain.reset(noise_seeds)
            trace = torch.zeros((M, B), device=dev)
            count = torch.zeros_like(trace)
            current = cue_cur[c]
            start = time.monotonic()
            for t in range(warm + tpd):
                spikes = brain.step(current).index_select(0, brain.motor).float()
                trace.mul_(beta).add_(spikes)
                if t >= warm:
                    count.add_(spikes)
                if t % 25 == 0:
                    ctl.sub(t, warm + tpd, 'tick')
            torch.cuda.synchronize(dev)
            elapsed = time.monotonic() - start
            time.sleep(elapsed * (1 / .6 - 1))
            trace_features[c] = trace.t().cpu().numpy()
            count_features[c] = count.t().cpu().numpy()
            ctl.done({'scene': names[c], 'seeds': B})
        y = np.repeat(np.arange(2), B)
        train = np.tile(np.arange(B) < 12, 2)
        results = {}
        for name, X in [('trace', trace_features.reshape(2 * B, M)),
                        ('spike_count', count_features.reshape(2 * B, M))]:
            Xtr, Xte, ytr, yte = X[train], X[~train], y[train], y[~train]
            actions = ((X @ rows[1]) > (X @ rows[0])).astype(int)
            results[name] = {
                'full_decoder': fit_score(Xtr, ytr, Xte, yte),
                'full_shuffled': fit_score(Xtr, ytr, Xte, yte, shuffled=True),
                'one_feature': one_feature_score(Xtr, ytr, Xte, yte),
                'one_feature_shuffled': one_feature_score(Xtr, ytr, Xte, yte,
                                                          shuffled=True),
                'fixed_readout_mapping0': float(np.mean(actions[~train] == yte)),
                'fixed_readout_mapping1': float(np.mean(actions[~train] == (1-yte))),
            }
        report = {'what': 'episodic visual two-cue observability on fresh seeds',
                  'complete': True, 'gate_g_pass': False, 'design': design,
                  'results': results, 'scope': 'no learning; task candidate only',
                  'manifest': str(manifest.path),
                  'utc': datetime.now(UTC).isoformat(timespec='seconds')}
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
        ctl.close('complete')
        manifest.finish('complete')
        print(json.dumps(results, indent=2))
    except BaseException as exc:
        ctl.close('failed')
        manifest.finish('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
