"""ACTIVITY_MATCH_V1: calibration rules, contrasts, bootstrap pairing, decoder isolation from test labels and the gain
intervention on a tiny network (GPU tests skipped without CUDA)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/activity_match"))

from flymario.activity_match import analysis, calibrate  # noqa: E402

RULES = {"minimum_active_fraction": 0.01, "maximum_saturation_fraction": 0.05}


def pt(r: float, valid: bool = True) -> dict:
    return {"rate_hz": r, "valid": valid}


def test_max_count_from_refractoriness():
    assert calibrate.max_count(16, 1) == 8
    assert calibrate.max_count(16, 2) == 6
    assert calibrate.max_count(15, 1) == 8


def test_activity_rate_and_fractions():
    C = np.zeros((4, 10), np.uint8)
    C[0, :5] = 8
    a = calibrate.activity(C, window_s=0.05, cmax=8, sat_frac=0.9)
    assert a["rate_hz"] == pytest.approx(40 / (4 * 10 * 0.05))
    assert a["active_frac_mean"] == pytest.approx(0.5 / 4)
    assert a["saturated_frac_mean"] == pytest.approx(0.5 / 4)
    assert a["n_constant_features"] == 5


def test_validity_rules():
    ok = {"rate_hz": 30.0, "active_frac_mean": 0.7, "saturated_frac_mean": 0.001}
    assert calibrate.combine([ok, ok], True, RULES)["valid"]
    sat = dict(ok, saturated_frac_mean=0.2)
    v = calibrate.combine([sat, sat], True, RULES)
    assert not v["valid"] and v["invalid_reasons"] == ["saturated fraction above maximum"]
    assert not calibrate.combine([ok, ok], False, RULES)["valid"]
    silent = dict(ok, rate_hz=0.0, active_frac_mean=0.0)
    assert not calibrate.combine([silent], True, RULES)["valid"]


def test_common_range_and_targets():
    curves = {"a": {0.5: pt(10), 1.0: pt(20), 2.0: pt(40)}, "b": {0.5: pt(15), 1.0: pt(30), 2.0: pt(60, False)}}
    cr = calibrate.common_range(curves)
    assert cr["exists"] and cr["r_min"] == 15 and cr["r_max"] == 30
    t = calibrate.targets(15, 30, {"CL": 0.25, "CM": 0.5, "CH": 0.75})
    assert t == {"CL": 18.75, "CM": 22.5, "CH": 26.25}
    assert not calibrate.common_range({"a": {1.0: pt(10)}, "b": {1.0: pt(20)}})["exists"]


def test_search_monotone_converges_within_budget():
    f = lambda g: 20.0 * g ** 0.4                                         # noqa: E731
    pts = {g: pt(f(g)) for g in (0.5, 0.75, 1.0, 1.5, 2.0)}
    calls = []

    def ev(g):
        calls.append(g)
        return pt(f(g))

    tg = {"CM": 21.3, "CL": 17.0, "CH": 24.9}
    log = calibrate.search(pts, tg, ev, budget=9, per_target=3, fit_tol=0.005)
    assert len(calls) <= 9
    for t in tg.values():
        g, _ = calibrate.select(pts, t, 0.005)
        assert calibrate.rel_err(pts[g]["rate_hz"], t) <= 0.005
    assert "extra evaluations used" in log.notes[-1]


def test_search_stops_on_non_monotone_branch():
    vals = {0.5: 10.0, 1.0: 30.0}
    pts = {g: pt(v) for g, v in vals.items()}
    log = calibrate.search(pts, {"CM": 20.0}, lambda g: pt(50.0), budget=9, per_target=3, fit_tol=0.01)
    assert len(log.evaluations) == 1 and any("not monotone" in n for n in log.notes)


def test_selection_ties():
    pts = {0.9: pt(20.0), 1.1: pt(20.0), 1.5: pt(20.1)}
    assert calibrate.select(pts, 20.0, 0.02)[0] == 0.9                      # equal distance to 1: smaller gamma
    pts = {0.5: pt(18.0), 2.0: pt(22.5)}
    assert calibrate.select(pts, 20.0, 0.02)[0] == 0.5                      # none within tolerance: smaller error


def synthetic_cbar(n=160, K=40, seed=0):
    rng = np.random.default_rng(seed)
    conds = ["C0", "CM"]
    p = np.zeros((K + 1, 2, 2))
    p[0] = [[0.30, 0.80], [0.35, 0.80]]
    p[1:] = [[0.36, 0.75], [0.36, 0.75]]
    cbar = (rng.random((K + 1, 2, 2, n)) < p[..., None]).astype(float)
    return cbar, conds, np.repeat(np.arange(16), n // 16)


def test_contrast_identities():
    cbar, conds, _ = synthetic_cbar()
    c = analysis.contrasts(cbar.mean(-1), conds)
    assert c["R_orient"] == pytest.approx(c["T_real_orient"] - c["T_nulls_orient"])
    assert c["G_CM_orient"] == pytest.approx(c["M_connectome_CM_orient"] - c["M_nulls_CM_orient"])


def test_bootstrap_pairing_gives_zero_for_identical_graphs():
    cbar, conds, strata = synthetic_cbar()
    same = np.repeat(cbar[:1], cbar.shape[0], 0)
    bs = analysis.bootstrap(same, strata, conds, 200, 1)
    for k in ("G_C0_orient", "G_CM_orient", "R_orient", "G_CM_lum"):
        assert np.all(np.abs(bs[k]) < 1e-12)


def test_bootstrap_recovers_known_contrasts():
    cbar, conds, strata = synthetic_cbar(n=1600)
    bs = analysis.bootstrap(cbar, strata, conds, 1000, 2)
    iv = analysis.intervals(bs)
    for k, truth in (("G_C0_orient", -0.06), ("G_CM_orient", -0.01), ("R_orient", 0.05)):
        assert iv[k][0] - 0.02 < truth < iv[k][1] + 0.02
    c = analysis.strong_criteria(iv, 0.02, 0.30, True)
    assert set(c) >= {"all", "G_CM_within_margin", "R_positive_interval_excludes_zero"}


def test_reading_table():
    assert analysis.reading_G(-0.01, 0.015, 0.02).startswith("compatible")
    assert analysis.reading_G(-0.06, -0.03, 0.02).startswith("persistent")
    assert analysis.reading_G(0.03, 0.05, 0.02).startswith("advantage")
    assert analysis.reading_G(-0.03, 0.01, 0.02).startswith("practical importance not settled")


def test_decoder_never_reads_test_labels():
    import run_am
    rng = np.random.default_rng(3)
    part = np.repeat([0, 1, 2], [256, 128, 128])
    orient = np.tile(np.arange(4), 128)
    lum = np.repeat(np.arange(4), 128)
    F = rng.normal(size=(512, 30)) + orient[:, None] * 0.3
    from flymario.retino_align import decode
    tr = part == 0
    subs = decode.subsets(orient[tr] * 4 + lum[tr], [1, 2], (2, 4))
    y = {"orient": orient.copy(), "lum": lum.copy()}
    a = run_am.fit_predict(F, y, part, subs, list(decode.LAMBDAS))
    y2 = {k: v.copy() for k, v in y.items()}
    for k in y2:
        y2[k][part == 2] = rng.permutation(y2[k][part == 2])
    b = run_am.fit_predict(F, y2, part, subs, list(decode.LAMBDAS))
    for k in ("W", "b", "lam_idx", "pred", "val_acc"):
        assert np.array_equal(a[k], b[k])


def test_gain_multiplies_recurrent_input_only_once():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    import dataclasses

    import run_am
    import scipy.sparse as sp

    from flymario.retino_align import simulate
    base = dataclasses.replace(run_am.base_params(), noise_std=0.0)
    W = sp.csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]], np.float32))
    net = simulate.FrozenNet(W, np.array([1]), np.array([0]), base, 2, batch=2)
    for g in (0.5, 2.0, 1.0, 2.0):
        run_am.set_gamma(net, base, g)
        v = net.run(np.array([[5.0]], np.float32), np.array([0]), 0).v_readout_mean[0]
        assert v == pytest.approx(float(base.gain) * g, abs=1e-6)
    assert base.gain == run_am.base_params().gain                            # the base is never modified


def test_v2_contrasts_with_unchanged_connectome():
    """ACTIVITY_MATCH_V2: the connectome is identical in C0 and CN, so T_real = 0 and R = -T_nulls exactly."""
    cbar, _, strata = synthetic_cbar()
    cbar[0, 1] = cbar[0, 0]                                                  # connectome unchanged
    conds = ["C0", "CN"]
    c = analysis.contrasts(cbar.mean(-1), conds, matched="CN")
    assert c["T_real_orient"] == 0.0
    assert c["R_orient"] == pytest.approx(-c["T_nulls_orient"])
    bs = analysis.bootstrap(cbar, strata, conds, 100, 3, matched="CN")
    iv = analysis.intervals(bs)
    assert np.all(bs["T_real_orient"] == 0.0)
    flags = analysis.criteria_v2(iv, 0.02, 0.27, True, True)
    assert set(flags) >= {"all", "G_CN_within_margin", "deficit_at_C0_below_minus_delta"}
