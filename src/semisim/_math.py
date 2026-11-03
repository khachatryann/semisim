"""Numerical helpers (pure NumPy, no SciPy dependency).

Currently provides the principal branch of the Lambert W function for
non-negative arguments, evaluated from the *logarithm* of the argument so
that the diode series-resistance solution never overflows even at large
forward bias.
"""

import numpy as np


def lambertw_log(log_x: np.ndarray | float) -> np.ndarray:
    """Principal-branch Lambert W of ``exp(log_x)``.

    Solves ``w * exp(w) = exp(log_x)`` (equivalently ``w + log(w) = log_x``
    for ``w > 0``) using Newton iteration. Taking the argument in log space
    avoids overflow for the diode equation, where the argument grows as
    ``exp(V / n V_T)``.

    Parameters
    ----------
    log_x : float or ndarray
        Natural log of the (positive) argument. May be any real value;
        very negative values return ``W ~ exp(log_x)``.

    Returns
    -------
    ndarray
        ``W0(exp(log_x))``, same shape as ``log_x``.

    Notes
    -----
    Accuracy is ~1e-12 relative over the tested range. Only the principal
    branch for non-negative arguments is implemented, which is all the
    diode model requires.
    """
    log_x = np.asarray(log_x, dtype=float)
    w = np.empty_like(log_x)

    # Below -37, W(x) = x to full double precision (relative error ~x).
    tiny = log_x < -37.0
    small = (log_x >= -37.0) & (log_x <= 1.0)
    large = log_x > 1.0

    w[tiny] = np.exp(log_x[tiny])
    # For moderate arguments W(e^L) ~ e^L is an excellent starting point
    # (exact as L -> -inf); Newton then converges in a few steps.
    w[small] = np.exp(log_x[small])
    # Asymptotic first guess: W ~ L - log(L) for large arguments.
    ll = log_x[large]
    w[large] = ll - np.log(np.maximum(ll, 1.1))

    # Newton iteration on f(w) = w + log(w) - log_x (skip the tiny branch,
    # already accurate to ~1e-9 there and log(w) would underflow).
    active = ~tiny
    wa = w[active]
    la = log_x[active]
    for _ in range(50):
        f = wa + np.log(wa) - la
        step = f * wa / (wa + 1.0)
        wa = np.maximum(wa - step, 1e-300)
        if np.all(np.abs(f) < 1e-14):
            break
    w[active] = wa
    return w


def bisect_monotonic(func, target, lo, hi, iterations: int = 200) -> np.ndarray:
    """Vectorized bisection for ``func(x) = target`` with ``func`` increasing.

    Parameters
    ----------
    func : callable
        Element-wise, monotonically non-decreasing function of an ndarray.
        May return ``inf``/``-inf`` (treated as above/below the target).
    target, lo, hi : ndarray or float
        Target values and bracketing bounds (broadcast together). The root
        must lie in ``[lo, hi]``.
    iterations : int, optional
        Maximum number of halvings; stops early once the interval has
        collapsed to machine precision everywhere.

    Returns
    -------
    ndarray
        Midpoint of the final bracket.
    """
    target, lo, hi = np.broadcast_arrays(
        np.asarray(target, dtype=float), np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    )
    lo = lo.copy()
    hi = hi.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if np.all((mid == lo) | (mid == hi)):
            break
        with np.errstate(over="ignore", invalid="ignore"):
            above = func(mid) > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)
