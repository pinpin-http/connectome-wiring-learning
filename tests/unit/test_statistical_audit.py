"""Independent checks of the vectorized statistics used by the report audit."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'statistical_audit', Path(__file__).resolve().parents[2] / 'tools/audits/statistical_20261019.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def brute_z(x):
    return np.stack([(x[i] - np.delete(x, i, axis=0).mean(0)) /
                     np.delete(x, i, axis=0).std(0, ddof=1) for i in range(len(x))])


@pytest.mark.parametrize('k,m', [(20, 2), (20, 8), (40, 2), (40, 8)])
def test_batched_statistics_match_independent_delete_one_implementation(k, m):
    x = np.random.default_rng(1729).normal(size=(3, k + 1, m))
    z = np.stack([brute_z(rows) for rows in x])
    np.testing.assert_allclose(audit.leave_one_out_z(x), z, rtol=1e-12, atol=1e-12)
    expected_raw = (abs(z) >= abs(z[:, :1])).mean(axis=1)
    expected_adj = (abs(z).max(axis=2)[:, :, None] >= abs(z[:, :1])).mean(axis=1)
    raw, adjusted = audit.symmetric_p(x)
    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_array_equal(adjusted, expected_adj)
    expected_historical = []
    for rows in x:
        zc = (rows[0] - rows[1:].mean(0)) / rows[1:].std(0, ddof=1)
        maxima = abs(brute_z(rows[1:])).max(axis=1)
        expected_historical.append((1 + (maxima[:, None] >= abs(zc)).sum(0)) / (k + 1))
    np.testing.assert_array_equal(audit.historical_p(x), expected_historical)


def test_endpoint_units_and_reference_order_do_not_change_probabilities():
    rng = np.random.default_rng(52)
    x = rng.normal(size=(41, 8))
    transformed = x * np.array([.1, -2, 4, 8, -.5, 3, 5, -1]) + np.arange(8)
    transformed = transformed[np.r_[0, rng.permutation(np.arange(1, 41))]]
    for method in (audit.historical_p, lambda rows: audit.symmetric_p(rows)[1]):
        np.testing.assert_array_equal(method(x), method(transformed))


def test_constant_endpoint_cannot_create_a_rejection():
    x = np.random.default_rng(92).normal(size=(41, 8))
    x[:, 3] = 7
    # The frozen simulation implementation forms 0*inf in an unused np.where branch.
    with np.errstate(invalid='ignore'):
        raw, adj = audit.symmetric_p(x)
        historical = audit.historical_p(x)
    assert raw[3] == adj[3] == historical[3] == 1
    assert np.isfinite(raw).all() and np.isfinite(adj).all()
    assert (adj >= raw).all()


def test_isolated_outlier_against_constant_references_is_finite_and_extreme():
    x = np.ones((41, 2)); x[0] = 10
    with np.errstate(invalid='ignore'):
        raw, adj = audit.symmetric_p(x)
        historical = audit.historical_p(x)
    np.testing.assert_allclose(raw, 1 / 41)
    np.testing.assert_allclose(adj, 1 / 41)
    np.testing.assert_allclose(historical, 1 / 41)
