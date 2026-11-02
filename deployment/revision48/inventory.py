"""Locate and hash the proposed 21-graph design without changing any input."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/soheib/Downloads/campagne')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    archive = next(BASE.rglob('COMPLEMENTS_complet_2026-10-21.zip'))
    with ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('matched_activity_full/plan.json'))
        selected = ['connectome'] + json.loads(z.read(name))['graphs']
    files = list(BASE.rglob('artifacts/cache/nulls/*.npz'))
    source = BASE / 'SMB_Flywire_Speedrun_final_2026-10-20_1036UTC-002/SMB_Flywire_Speedrun'
    rows = []
    for spec in selected:
        if spec == 'connectome':
            matches = [source / 'data/connectome/malecns.npz']
        else:
            family, seed = spec.split(':')
            variant = 'depth_stratified_lateral' + ('_instrength' if family == 'nullin' else '')
            matches = [p for p in files if p.name.endswith(f'_{variant}_s{seed}.npz')]
        matches = [p for p in matches if p.is_file()]
        if len(matches) != 1:
            raise RuntimeError(f'{spec}: expected exactly one cached source; found {len(matches)}')
        p = matches[0]
        rows.append(dict(graph=spec, path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    data = dict(status='DRAFT proposed historical subset; not frozen or newly randomized',
                selection_source=str(archive), selection_member=name,
                inputs=rows, total_bytes=sum(r['bytes'] for r in rows))
    out = ROOT / 'artifacts/revision48_prep/inputs.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(graphs=len(rows), total_gb=data['total_bytes']/1e9, manifest=str(out)), indent=2))


if __name__ == '__main__':
    main()
