"""RETINO_ALIGN_V1 checks of concrete scientific risks (mission §17). Each test can fail on a real defect.

The retina and simulator tests use the real MaleCNS graph (skipped when the data or a GPU is absent).
"""
from __future__ import annotations

import numpy as np
import pytest

from flymario.retino_align import analysis, decode, retina, simulate, stimuli


@pytest.fixture(scope="module")
def real():
    from flymario.connectome import datasets
    from flymario.env.vision import RetinotopicVision
    try:
        g = datasets.load(datasets.DEFAULT)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"connectome not available: {e}")
    vis = RetinotopicVision(g)
    return g, vis, retina.build(g, vis)


# 1-2. permutation: bijection, inverse, types, sides, ON/OFF together -------------------------------------------------
def test_permutation_is_a_type_preserving_bijection_with_exact_inverse(real):
    _g, _vis, ret = real
    pi = retina.column_permutation(ret, 7)
    sigma = retina.cell_permutation(ret, pi)
    assert np.array_equal(np.sort(sigma), np.arange(ret.n_cells))
    x = np.random.default_rng(0).normal(size=(5, ret.n_cells)).astype(np.float32)
    px = x[:, sigma]
    assert np.array_equal(px[:, retina.inverse(sigma)], x)
    assert np.array_equal(ret.types[sigma], ret.types)              # no type mixing
    assert np.array_equal(np.sort(px, 1), np.sort(x, 1))            # value multiset kept
    for t in retina.LAMINA:                                          # ... within each type
        m = ret.types == t
        assert np.array_equal(np.sort(px[:, m], 1), np.sort(x[:, m], 1))
    # signature groups only: a fixed column stays fixed
    fixed = ~np.isin(np.arange(ret.n_cols), np.concatenate(list(ret.groups().values())))
    assert np.array_equal(pi[fixed], np.flatnonzero(fixed))
    assert np.array_equal(ret.col_sig[pi], ret.col_sig)


def test_on_and_off_cells_of_a_column_move_together(real):
    _g, _vis, ret = real
    pi = retina.column_permutation(ret, 3)
    sigma = retina.cell_permutation(ret, pi)
    src = ret.col[sigma]
    for c in range(ret.n_cols):
        assert np.all(src[ret.col == c] == pi[c])
    rep = retina.report(ret, pi, sigma)
    assert rep["on_off_same_displacement"] and rep["types_preserved"]
    # a defective map that permutes ON and OFF independently must be caught by the same check
    bad = sigma.copy()
    on = np.flatnonzero(np.isin(ret.types, retina.ON_TYPES))
    bad[on] = sigma[on][np.random.default_rng(1).permutation(len(on))]
    assert not retina._on_off_together(ret, bad)


def test_permutations_differ_and_depend_on_the_seed_only(real):
    _g, _vis, ret = real
    p1, p2, p1b = (retina.column_permutation(ret, s) for s in (1, 2, 1))
    assert not np.array_equal(p1, p2)
    assert np.array_equal(p1, p1b)


# 3-4. datasets: balance, disjoint partitions ---------------------------------------------------------------------------
def test_every_partition_and_every_budget_is_balanced_on_the_16_combinations():
    gp = stimuli.GenParams()
    for name, per in (("train", 64), ("val", 32), ("test", 160)):
        b = stimuli.draw_bank(name, per, 5, gp)
        combo = b.orient.astype(int) * 4 + b.lum
        assert np.array_equal(np.bincount(combo, minlength=16), np.full(16, per))
    b = stimuli.draw_bank("train", 64, 5, gp)
    combo = b.orient.astype(int) * 4 + b.lum
    sub = decode.subsets(combo, [11, 12, 13])
    for r in range(sub.shape[0]):
        prev = set()
        for k, m in enumerate((2, 4, 8, 16, 32)):
            sel = sub[r, k][sub[r, k] >= 0]
            assert np.array_equal(np.bincount(combo[sel], minlength=16), np.full(16, m))
            assert prev <= set(sel.tolist())                          # nested
            prev = set(sel.tolist())


def test_partitions_share_no_example_and_no_latent_draw():
    gp = stimuli.GenParams()
    banks = [stimuli.draw_bank(n, 4, s, gp) for n, s in (("train", 1), ("val", 2), ("test", 3), ("dev", 4))]
    uids = np.concatenate([b.uid for b in banks])
    assert len(np.unique(uids)) == len(uids)
    lat = np.concatenate([np.stack([b.latent[k] for k in ("phase", "period", "amp", "lum_offset")], 1)
                          for b in banks])
    assert len(np.unique(lat, axis=0)) == len(lat)
    noise = np.concatenate([b.latent["noise_seed"] for b in banks])
    assert len(np.unique(noise)) == len(noise)


# 5-6. simulator: frozen weights, no carry-over, readout before projection --------------------------------------------
@pytest.fixture(scope="module")
def net(real):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from flymario import config
    from flymario.connectome import datasets
    from flymario.models.spiking_reservoir import LIFParams
    g, vis, ret = real
    p = LIFParams(gain=float(config.get("dynamics.gain")), noise_std=float(config.get("dynamics.noise_std")))
    motor = datasets.motor_index(g, datasets.DEFAULT)
    gp = stimuli.GenParams(amp_range=(0.2, 0.25))
    b = stimuli.draw_bank("dev", 1, 77, gp)
    X, _ = stimuli.encode_bank(b, gp, vis, ret.inj, ret.col_uv)
    return simulate.FrozenNet(g.to_sparse(), motor, ret.inj, p, 16, batch=16), X, b.uid, motor


def test_weights_unchanged_and_no_state_carried_between_images(net):
    fn, X, uid, _ = net
    d0 = fn.device_operator_digest()
    a = fn.run(X, uid, 0).counts
    assert fn.device_operator_digest() == d0 == fn.digest
    # image 0 alone, then after 15 other images in a different order, then in a batch of 3
    alone = fn.run(X[:1], uid[:1], 0).counts
    order = np.r_[np.arange(15, 0, -1), 0]
    shuffled = fn.run(X[order], uid[order], 0).counts
    assert np.array_equal(alone[0], a[0]) and np.array_equal(shuffled[-1], a[0])
    fn3 = simulate.FrozenNet.__new__(simulate.FrozenNet)
    fn3.__dict__.update(fn.__dict__)
    fn3.B = 3
    assert np.array_equal(fn3.run(X, uid, 0).counts, a)             # batch size does not change the data
    assert a.sum() > 0, "a silent readout would make these tests vacuous"


def test_counts_are_the_readout_neurons_spikes_before_any_projection(net, real):
    import torch

    from flymario.models.spiking_reservoir import LIFParams
    from flymario.models.torch_lif import TorchLIF
    fn, X, uid, motor = net
    g, _vis, ret = real
    p0 = LIFParams(gain=fn.p.gain, noise_std=0.0)
    ref = TorchLIF(g.to_sparse(), p0, batch=4, normalize=False, seeds=[0, 1, 2, 3], noise="torch")
    cur = torch.zeros((ref.n, 4), device="cuda")
    cur[torch.as_tensor(ret.inj, device="cuda")] = torch.as_tensor(X[:4], device="cuda").t()
    c = torch.zeros((len(motor), 4), device="cuda")
    for _ in range(16):
        c += ref.step(cur)[torch.as_tensor(motor, device="cuda").long()].float()
    mine = simulate.FrozenNet(g.to_sparse(), motor, ret.inj, p0, 16, batch=4).run(X[:4], uid[:4], 0)
    assert mine.counts.shape == (4, 2227)
    assert np.array_equal(mine.counts, c.t().cpu().numpy().astype(np.uint8))


# 7-9. decoder ----------------------------------------------------------------------------------------------------------
def _toy(n=96, d=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(4), n // 4)
    X = rng.normal(size=(n, d)) + np.eye(4)[y] @ rng.normal(size=(4, d)) * 0.8
    X[:, 3] = 2.0                                                     # a constant column
    return X, y


def test_direct_decoder_is_invariant_to_a_feature_permutation():
    X, y = _toy()
    Xv, yv = _toy(64, seed=1)
    perm = np.random.default_rng(2).permutation(X.shape[1])
    f1, f2 = decode.fit_all_lambdas(X, y), decode.fit_all_lambdas(X[:, perm], y)
    s1, s2 = decode.scores(f1, Xv), decode.scores(f2, Xv[:, perm])
    assert np.max(np.abs(s1 - s2)) < 1e-9
    assert np.array_equal(s1.argmax(2), s2.argmax(2))
    # an anisotropic penalty would break the invariance: the check can fail
    scale = np.linspace(1, 3, X.shape[1])
    f3 = decode.fit_all_lambdas(X * scale, y)
    assert np.max(np.abs(decode.scores(f3, Xv * scale) - s1)) < 1e-9  # standardisation removes scale ...
    assert not np.allclose(decode.scores(decode.Fit(f1.W[:, perm], f1.b, f1.mu, f1.sd), Xv), s1)


def test_lambda_convention_matches_the_normalised_objective_and_sklearn():
    from sklearn.linear_model import Ridge
    X, y = _toy()
    n = len(X)
    f = decode.fit_all_lambdas(X, y)
    Z = (X - f.mu) / f.sd
    Y = np.eye(4)[y]
    for li, lam in enumerate(decode.LAMBDAS):
        W, b = f.W[li], f.b
        grad_W = (2 / n) * Z.T @ (Z @ W + b - Y) + 2 * lam * W         # gradient of the declared objective
        grad_b = (2 / n) * (Z @ W + b - Y).sum(0)
        assert np.abs(grad_W).max() < 1e-8 and np.abs(grad_b).max() < 1e-8
        sk = Ridge(alpha=decode.alpha_of(lam, n), fit_intercept=True).fit(Z, Y)
        assert np.allclose(sk.coef_.T, W, atol=1e-7) and np.allclose(sk.intercept_, b, atol=1e-7)
        if lam == 1.0:   # the naive alpha = lambda would be a different fit
            assert not np.allclose(Ridge(alpha=lam).fit(Z, Y).coef_.T, W, atol=1e-4)


def test_standardisation_uses_the_training_subset_only():
    X, y = _toy()
    f = decode.fit_all_lambdas(X[:48], y[:48])
    assert np.allclose(f.mu, X[:48].mean(0))
    Xo = X.copy()
    Xo[48:] += 100.0                                                  # outside the subset: no effect on the fit
    g = decode.fit_all_lambdas(Xo[:48], y[:48])
    assert np.array_equal(f.mu, g.mu) and np.array_equal(f.sd, g.sd) and np.array_equal(f.W, g.W)
    assert np.all(np.isfinite(f.W)) and np.allclose(f.W[:, 3, :], 0.0)  # constant column: no division by zero


def test_selection_ties_go_to_the_largest_lambda():
    assert decode.select([0.5, 0.7, 0.7, 0.6, 0.7, 0.2]) == 4
    assert decode.select([0.9, 0.1, 0.1, 0.1, 0.1, 0.1]) == 0


# 10-11. analysis ---------------------------------------------------------------------------------------------------
def _synthetic_cbar(K=20, n=1600, effect=0.08, seed=0):
    """cbar with known M: nulls lose 0.10 under permutation, the connectome 0.10 + effect (orientation only)."""
    rng = np.random.default_rng(seed)
    G = K + 1
    M = np.zeros((G, 4, 2))
    base = 0.6 + rng.normal(0, 0.01, G)
    M[:, 0, :] = base[:, None]
    M[:, 1:, :] = base[:, None, None] - 0.10
    M[0, 1:, 0] -= effect
    strata = np.repeat(np.arange(16), n // 16)
    cbar = np.zeros((G, 4, 2, n))
    for idx in np.ndindex(G, 4, 2):          # exact per-stratum means, so M is reproduced exactly
        k = int(round(M[idx] * (n // 16)))
        row = np.zeros(n)
        for s in range(16):
            row[np.flatnonzero(strata == s)[:k]] = 1.0
        cbar[idx] = row
        M[idx] = row.mean()
    return cbar, strata, M


def test_D_G_S_on_synthetic_data_with_a_known_contrast():
    cbar, strata, M = _synthetic_cbar()
    c = analysis.contrasts(analysis.scores(cbar))
    L = M[:, 0, :] - M[:, 1:, :].mean(1)
    assert c["D_orientation"] == pytest.approx(L[0, 0] - L[1:, 0].mean())
    assert c["D_orientation"] == pytest.approx(0.08, abs=0.011)
    assert c["D_brightness"] == pytest.approx(0.0, abs=0.011)
    assert c["S"] == pytest.approx(c["D_orientation"] - c["D_brightness"])
    assert c["G"] == pytest.approx(M[0, 0, 0] - M[1:, 0, 0].mean())


def test_bootstrap_resamples_the_same_images_everywhere():
    cbar, strata, _ = _synthetic_cbar(effect=0.0)
    cbar[0] = cbar[1]                                                 # connectome identical to null 1
    cbar[2:] = cbar[1]                                                # ... and to every null
    bs = analysis.bootstrap(cbar, strata, 200, seed=3)
    # paired resampling: identical graphs give exactly zero in every replicate; unpaired would not
    assert np.all(np.abs(bs["D_orientation"]) < 1e-12) and np.all(np.abs(bs["G"]) < 1e-12)
    cbar2, strata2, _ = _synthetic_cbar(effect=0.08)
    bs2 = analysis.bootstrap(cbar2, strata2, 400, seed=4)
    lo, hi = analysis.interval(bs2["D_orientation"])
    assert lo < 0.08 < hi + 0.011


def test_reading_follows_the_mission_table():
    d = 0.05
    assert analysis.reading(0.06, 0.2, d).startswith("positive differential")
    assert analysis.reading(-0.03, 0.04, d).startswith("compatible with practical equivalence")
    assert analysis.reading(0.01, 0.09, d).startswith("positive effect compatible")
    assert analysis.reading(-0.2, -0.06, d).startswith("important differential effect in the opposite")
    assert analysis.reading(-0.08, 0.03, d).startswith("a positive effect of the targeted size is excluded")
    assert analysis.reading(-0.02, 0.09, d).startswith("indeterminate")


# 12. resume -------------------------------------------------------------------------------------------------------
def test_an_interrupted_lot_is_recomputed_identically_and_valid_lots_are_untouched(tmp_path):
    """On a copy of the pilot run: one lot loses its record (as if killed before the rename) and leaves a stray
    .part; `extract` recomputes only that lot, bit for bit, and leaves every complete lot's file as it was."""
    import hashlib
    import json
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    import torch
    root = Path(__file__).resolve().parents[2]
    src = root / "artifacts/retino_align/pilot_e2e"
    if not (src / "manifest.json").exists() or not torch.cuda.is_available():
        pytest.skip("needs the pilot run and a GPU")
    needed = [src / "inputs/inputs.npz", src / "permutations/permutations.npz"]
    if not all(p.exists() for p in needed) or len(list((src / "features").rglob("*.npz"))) < 4:
        pytest.skip("needs the pilot's .npz inputs, permutations and features (not in git)")
    # Bit-identity holds only on the card that extracted the pilot (docs/REPRODUCIBILITY.md, GPU non-determinism).
    pilot_gpu = json.loads((src / "manifest.json").read_text())["env"]["gpu"]
    if torch.cuda.get_device_name(0) != pilot_gpu:
        pytest.skip(f"bit-identity is checked on the pilot's card ({pilot_gpu}), not {torch.cuda.get_device_name(0)}")
    name = "_test_resume_tmp"
    dst = root / "artifacts/retino_align" / name
    shutil.rmtree(dst, ignore_errors=True)
    try:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("decoders", "predictions", "metrics", "figures"))
        man = (dst / "manifest.json")
        man.write_text(man.read_text().replace('"run": "pilot_e2e"', f'"run": "{name}"'))
        feats = sorted((dst / "features").rglob("*.npz"))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
        before = {p: sha(p) for p in feats}
        mtimes = {p: p.stat().st_mtime_ns for p in feats}
        victim = feats[3]
        victim.with_suffix(".json").unlink()
        victim.with_name(victim.name + ".part").write_bytes(b"truncated")
        r = subprocess.run([sys.executable, str(root / "scripts/retino_align/run_study.py"), "extract", "--run", name],
                           capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-2000:]
        assert r.stdout.count(" done=") == 1                      # exactly one lot recomputed
        for p in feats:
            assert sha(p) == before[p]                          # the recomputed lot is bit-identical
            if p != victim:
                assert p.stat().st_mtime_ns == mtimes[p]        # complete lots were not rewritten
        assert victim.with_suffix(".json").exists()
    finally:
        shutil.rmtree(dst, ignore_errors=True)
