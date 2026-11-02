"""Matched one-motor-feature assay on frozen continuous streams and a synthetic control."""
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

from gate_g_continuous_pair_screen import SEEDS, N, pair_data, scene_stream

from flymario import config
from flymario.paths import RUNS
from flymario.run_manifest import RunManifest, dataset_files

SHUFFLE_SEED = 20260927999


def one_feature_score(Xtr, ytr, Xte, yte, shuffled=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr = np.asarray(Xtr, np.float64)
    Xte = np.asarray(Xte, np.float64)
    labels = (np.random.default_rng(SHUFFLE_SEED).permutation(ytr)
              if shuffled else ytr)
    scaler = StandardScaler().fit(Xtr)
    scaled = scaler.transform(Xtr)
    difference = np.abs(scaled[labels == 1].mean(axis=0)
                        - scaled[labels == 0].mean(axis=0))
    difference[np.ptp(Xtr, axis=0) <= 1e-6] = -np.inf
    if not np.isfinite(difference).any():
        raise ValueError('no dynamic motor feature')
    coordinate = int(np.argmax(difference))
    model = LogisticRegression(C=.05, max_iter=2000, random_state=0).fit(
        scaled[:, [coordinate]], labels)
    probabilities = model.predict_proba(scaler.transform(Xte)[:, [coordinate]])
    prediction = model.classes_[probabilities.argmax(axis=1)]
    return {'accuracy': float(np.mean(prediction == yte)),
            'mean_true_probability': float(np.mean(
                probabilities[np.arange(len(yte)), np.searchsorted(model.classes_, yte)])),
            'coordinate': coordinate, 'n': len(yte)}


def screen(features, names, sequences):
    rows = []
    for pair in itertools.combinations(range(len(names)), 2):
        Xtr, ytr = pair_data(features[:, :, 0], sequences[:, 0], pair)
        Xsel, ysel = pair_data(features[:, :, 1], sequences[:, 1], pair)
        rows.append({'pair': [names[i] for i in pair], 'indices': pair,
                     'selection': one_feature_score(Xtr, ytr, Xsel, ysel)})
    rows.sort(key=lambda r: (-r['selection']['accuracy'],
                             -r['selection']['mean_true_probability'], r['pair']))
    selected = rows[0]
    pair = selected['indices']
    Xtr, ytr = pair_data(features[:, :, 0], sequences[:, 0], pair)
    Xte, yte = pair_data(features[:, :, 2], sequences[:, 2], pair)
    confirmed = one_feature_score(Xtr, ytr, Xte, yte)
    shuffled = one_feature_score(Xtr, ytr, Xte, yte, shuffled=True)
    chance_2se = float(2 * np.sqrt(.25 / len(yte)))
    return {'pair_scores': rows, 'selected_pair': selected['pair'],
            'confirmation': confirmed, 'shuffled_confirmation': shuffled,
            'chance_2se': chance_2se,
            'exploratory_candidate': (confirmed['accuracy'] >= .75
                                      and abs(shuffled['accuracy'] - .5) <= chance_2se)}


def main():
    import torch

    source = RUNS / 'gate_g_continuous_pair_screen_ten-scenes-20260927.json'
    source_report = json.loads(source.read_text())
    checkpoint = Path(source_report['checkpoint'])
    snapshot = torch.load(checkpoint, map_location='cpu', weights_only=False)
    names = source_report['design']['scenes']
    sequences = np.stack([scene_stream(s, len(names), N) for s in SEEDS], axis=1)
    original = np.asarray(snapshot['features'], np.float64)
    if original.shape != (N, 2227, 3) or snapshot['trial'] != N:
        raise ValueError('incomplete or mismatched source features')
    target = names.index('1-1_open')
    coordinate = int(np.argmax(np.std(original[:, :, 0], axis=0)))
    amplitude = float(4 * np.std(original[:, coordinate, 0]))
    synthetic = original.copy()
    for b in range(len(SEEDS)):
        synthetic[sequences[:, b] == target, coordinate, b] += amplitude

    original_result = screen(original, names, sequences)
    synthetic_result = screen(synthetic, names, sequences)
    run_id = 'gate_g_restricted_decoder_1feature-20260927'
    output = RUNS / f'{run_id}.json'
    if output.exists():
        raise FileExistsError(output)
    design = {'source_report': str(source), 'source_checkpoint': str(checkpoint),
              'training_seed': 3, 'selection_seed': 4, 'confirmation_seed': 5,
              'decisions_per_seed': N, 'scenes': names,
              'feature_selection': 'largest abs standardized between-class mean difference; training range >1e-6',
              'classifier': 'logistic C=.05 on exactly one selected feature',
              'synthetic_scene': names[target], 'synthetic_coordinate': coordinate,
              'synthetic_amplitude': amplitude, 'synthetic_rule': '4 × seed-3 pooled SD',
              'shuffle_seed': SHUFFLE_SEED}
    manifest = RunManifest(run_id, 'Gate G', 'matched one-feature continuous decode',
                           design, seeds=list(SEEDS),
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(), 'script': Path(__file__),
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'source_report': source, 'source_checkpoint': checkpoint},
                           output_files={'report': output})
    report = {'what': 'one-feature continuous pair screen with matched synthetic positive control',
              'complete': True, 'gate_g_pass': False, 'design': design,
              'original': original_result, 'synthetic': synthetic_result,
              'assay_positive_control_pass': synthetic_result['exploratory_candidate'],
              'scope': 'restricted single-feature observability; no learning',
              'manifest': str(manifest.path),
              'utc': datetime.now(UTC).isoformat(timespec='seconds')}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    manifest.finish('complete')
    print(json.dumps({key: {k: report[key][k] for k in
                             ('selected_pair', 'confirmation', 'shuffled_confirmation',
                              'exploratory_candidate')}
                      for key in ('original', 'synthetic')}, indent=2))


if __name__ == '__main__':
    main()
