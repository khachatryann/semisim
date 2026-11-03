"""Smoke tests for the plotting helpers (Agg backend, no display)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from semisim.devices.diode import PNJunction  # noqa: E402
from semisim.devices.mosfet import MOSFET  # noqa: E402
from semisim.plotting import (  # noqa: E402
    plot_band_diagram,
    plot_diode_iv,
    plot_mosfet_output,
    plot_mosfet_transfer,
)


def teardown_function():
    plt.close("all")


def test_plot_diode_iv_single_and_multiple():
    d1 = PNJunction(na=1e17, nd=1e17)
    d2 = PNJunction(na=1e17, nd=1e17, temperature=350.0)
    v = np.linspace(-0.2, 0.7, 50)
    ax = plot_diode_iv(d1, v)
    assert ax.get_xlabel() == "Voltage [V]"
    ax2 = plot_diode_iv([d1, d2], v, log=True, labels=["300 K", "350 K"])
    assert len(ax2.get_lines()) == 2


def test_plot_band_diagram():
    d = PNJunction(na=1e17, nd=1e16)
    ax = plot_band_diagram(d, bias=0.2)
    assert len(ax.get_lines()) == 4  # Ec, Ev, Efp, Efn


def test_plot_mosfet_output_and_transfer():
    m = MOSFET(body_doping=1e17, tox=1e-6, width=1e-3, length=1e-4)
    ax = plot_mosfet_output(m, vgs_values=[1.0, 1.5, 2.0])
    assert len(ax.get_lines()) == 3
    ax2 = plot_mosfet_transfer(m, vds=1.0, log=True)
    assert ax2.get_yscale() == "log"
