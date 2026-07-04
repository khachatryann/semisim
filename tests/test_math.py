"""Tests for the pure-NumPy Lambert W implementation."""

import numpy as np
import pytest

from semisim._math import lambertw_log


@pytest.mark.parametrize(
    "x,expected",
    [
        (1.0, 0.5671432904097838),
        (np.e, 1.0),
        (10.0, 1.7455280027406994),
        (0.1, 0.09127652716086226),
    ],
)
def test_known_values(x, expected):
    assert float(lambertw_log(np.log(x))) == pytest.approx(expected, rel=1e-10)


def test_tiny_argument_limit():
    # W(x) ~ x as x -> 0
    assert float(lambertw_log(-40.0)) == pytest.approx(np.exp(-40.0), rel=1e-6)


def test_defining_identity_large_argument():
    # For log-space arguments far beyond float overflow (x = e^500),
    # verify w + log(w) == log_x.
    log_x = 500.0
    w = float(lambertw_log(log_x))
    assert w + np.log(w) == pytest.approx(log_x, rel=1e-12)


def test_vectorized_identity():
    log_x = np.linspace(-30.0, 100.0, 57)
    w = lambertw_log(log_x)
    # w * e^w = x  <=>  log(w) + w = log_x (for w not underflowing)
    mask = w > 1e-12
    residual = np.log(w[mask]) + w[mask] - log_x[mask]
    assert np.max(np.abs(residual)) < 1e-10
    assert w.shape == log_x.shape
    assert np.all(np.diff(w) > 0)  # strictly increasing
