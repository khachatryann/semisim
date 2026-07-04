"""Matplotlib plotting helpers for device characteristics.

All functions accept an optional ``ax`` and return the axes they drew on,
so plots can be composed into larger figures.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .devices.diode import PNJunction
from .devices.mosfet import MOSFET


def plot_diode_iv(
    diodes: PNJunction | Sequence[PNJunction],
    voltages: np.ndarray,
    labels: Sequence[str] | None = None,
    log: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot diode I-V characteristics.

    Parameters
    ----------
    diodes : PNJunction or sequence of PNJunction
        One or more diodes to plot.
    voltages : ndarray
        Voltage sweep [V].
    labels : sequence of str, optional
        Legend labels (defaults to material @ temperature).
    log : bool, optional
        Plot |I| on a log scale (default False).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if isinstance(diodes, PNJunction):
        diodes = [diodes]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for idx, diode in enumerate(diodes):
        label = (
            labels[idx]
            if labels is not None
            else f"{diode.material.name}, {diode.temperature:.0f} K"
        )
        current = diode.current(voltages)
        if log:
            ax.semilogy(voltages, np.abs(current), label=label)
        else:
            ax.plot(voltages, current * 1e3, label=label)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("|Current| [A]" if log else "Current [mA]")
    ax.set_title("Diode I-V characteristic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_band_diagram(
    diode: PNJunction, bias: float = 0.0, ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot the band diagram of a PN junction in the depletion approximation.

    Parameters
    ----------
    diode : PNJunction
        Diode to plot.
    bias : float, optional
        Applied voltage [V] (must be below the built-in potential).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    bd = diode.band_diagram(bias)
    x_um = bd["x"] * 1e4
    ax.plot(x_um, bd["ec"], "b-", label="$E_C$")
    ax.plot(x_um, bd["ev"], "r-", label="$E_V$")
    ax.plot(x_um, bd["efp"], "k--", lw=1, label="$E_{Fp}$")
    ax.plot(x_um, bd["efn"], "k-.", lw=1, label="$E_{Fn}$")
    ax.set_xlabel("Position [um]")
    ax.set_ylabel("Energy [eV]")
    ax.set_title(
        f"{diode.material.name} PN junction band diagram (V = {bias:g} V)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_mosfet_output(
    mosfet: MOSFET,
    vgs_values: Sequence[float],
    vds_max: float = 3.0,
    points: int = 200,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot MOSFET output characteristics (Id vs Vds for several Vgs).

    Parameters
    ----------
    mosfet : MOSFET
        Transistor to plot.
    vgs_values : sequence of float
        Gate voltages [V], one curve each.
    vds_max : float, optional
        Maximum drain voltage of the sweep [V].
    points : int, optional
        Samples in the Vds sweep.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    vds = np.linspace(0.0, vds_max, points)
    for vgs in vgs_values:
        i_d = mosfet.current(vgs=vgs, vds=vds)
        ax.plot(vds, i_d * 1e3, label=f"$V_{{GS}}$ = {vgs:g} V")
    ax.set_xlabel("$V_{DS}$ [V]")
    ax.set_ylabel("$I_D$ [mA]")
    ax.set_title("MOSFET output characteristics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_mosfet_transfer(
    mosfet: MOSFET,
    vds: float,
    vgs_range: tuple[float, float] = (0.0, 3.0),
    points: int = 300,
    log: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the MOSFET transfer characteristic (Id vs Vgs at fixed Vds).

    Parameters
    ----------
    mosfet : MOSFET
        Transistor to plot.
    vds : float
        Drain voltage [V].
    vgs_range : tuple of float, optional
        (min, max) gate voltage sweep [V].
    points : int, optional
        Samples in the sweep.
    log : bool, optional
        Log-scale current axis (reveals the subthreshold slope).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    vgs = np.linspace(*vgs_range, points)
    i_d = mosfet.current(vgs=vgs, vds=vds)
    if log:
        ax.semilogy(vgs, np.maximum(i_d, 1e-30), label=f"$V_{{DS}}$ = {vds:g} V")
    else:
        ax.plot(vgs, i_d * 1e3, label=f"$V_{{DS}}$ = {vds:g} V")
    ax.axvline(
        mosfet.threshold_voltage(), color="gray", ls=":", label="$V_{th}$"
    )
    ax.set_xlabel("$V_{GS}$ [V]")
    ax.set_ylabel("$I_D$ [A]" if log else "$I_D$ [mA]")
    ax.set_title("MOSFET transfer characteristic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
