"""Positive assay control: a declared synthetic cue signal in saved motor traces."""
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

from gate_g_continuous_pair_screen import SEEDS, N, fit_score, pair_data, scene_stream

from flymario import config
from flymario.paths import RUNS
from flymario.run_manifest import RunManifest, dataset_files


def main():
    import torch

    original = RUNS / 'gate_g_continuous_pair_screen_ten-scenes-20260927.json'
    base = json.loads(original.read_text())
    checkpoint = Path(base['checkpoint'])
    snapshot = torch.load(checkpoint, map_location='cpu', weights_only=False)
    names = base['design']['scenes']
    sequences = np.stack([scene_stream(s, len(names), N) for s in SEEDS], axis=1)
    features = np.asarray(snapshot['features'], np.float64).copy()
    if features.shape != (N, 2227, 3) or snapshot['trial'] != N:
        raise ValueError('incomplete or mismatched source features')
    target_scene = names.index('1-1_open')
    coordinate = int(np.argmax(np.std(features[:, :, 0], axis=0)))
    amplitude = float(4 * np.std(features[:, coordinate, 0]))
    if amplitude <= 0:
        raise ValueError('selected feature has zero variance')
    for b in range(len(SEEDS)):
        features[sequences[:, b] == target_scene, coordinate, b] += amplitude

    rows = []
    for pair in itertools.combinations(range(len(names)), 2):
        Xtr, ytr = pair_data(features[:, :, 0], sequences[:, 0], pair)
        Xsel, ysel = pair_data(features[:, :, 1], sequences[:, 1], pair)
        rows.append({'pair': [names[i] for i in pair], 'indices': pair,
                     'selection': fit_score(Xtr, ytr, Xsel, ysel)})
    rows.sort(key=lambda r: (-r['selection']['accuracy'],
                             -r['selection']['mean_true_probability'], r['pair']))
    selected = rows[0]
    pair = selected['indices']
    Xtr, ytr = pair_data(features[:, :, 0], sequences[:, 0], pair)
    Xte, yte = pair_data(features[:, :, 2], sequences[:, 2], pair)
    confirmation = fit_score(Xtr, ytr, Xte, yte)
    shuffled = fit_score(Xtr, ytr, Xte, yte, shuffled=True)
    chance_2se = float(2 * np.sqrt(.25 / confirmation['n']))
    run_id = 'gate_g_continuous_screen_control_synthetic4sd-20260927'
    output = RUNS / f'{run_id}.json'
    if output.exists():
        raise FileExistsError(output)
    design = {'source_report': str(original), 'source_checkpoint': str(checkpoint),
              'target_scene': names[target_scene], 'feature_coordinate': coordinate,
              'offset': amplitude, 'offset_rule': '4 × seed-3 pooled SD',
              'seeds': SEEDS, 'purpose': 'train',
              'decoder_and_selection': base['design']['decoder']}
    manifest = RunManifest(run_id, 'Gate G', 'matched continuous decoder positive control',
                           design, seeds=list(SEEDS),
                           dataset=str(config.get('connectome.dataset')),
                           input_files={**dataset_files(), 'script': Path(__file__),
                                        'screen_script': ROOT / 'scripts/gates/gate_g_continuous_pair_screen.py',
                                        'plan': ROOT / 'docs/GATE_G_PLAN.md',
                                        'source_report': original, 'source_checkpoint': checkpoint},
                           output_files={'report': output})
    report = {'what': 'synthetic positive control matched to continuous pair screen',
              'complete': True, 'gate_g_pass': False, 'design': design,
              'selection_scores': rows, 'selected_pair': selected['pair'],
              'confirmation': confirmation, 'shuffled_confirmation': shuffled,
              'chance_2se': chance_2se,
              'assay_positive_control_pass': (confirmation['accuracy'] >= .75
                                              and abs(shuffled['accuracy'] - .5) <= chance_2se),
              'scope': 'synthetic offset only; not a brain or learning result',
              'manifest': str(manifest.path),
              'utc': datetime.now(UTC).isoformat(timespec='seconds')}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    manifest.finish('complete')
    print(json.dumps({k: report[k] for k in ('selected_pair', 'confirmation',
                                               'shuffled_confirmation',
                                               'assay_positive_control_pass')}, indent=2))


if __name__ == '__main__':
    main()
