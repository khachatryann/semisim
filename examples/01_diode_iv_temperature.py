"""Diode I-V characteristics at three temperatures.

Demonstrates the strong temperature sensitivity of the saturation current
(via ni^2) and the resulting ~-2 mV/K shift of the forward knee.

Run from the repository root:

    python examples/01_diode_iv_temperature.py
"""

import matplotlib.pyplot as plt
import numpy as np

from semisim import PNJunction
from semisim.plotting import plot_diode_iv


def main() -> None:
    temperatures = [250.0, 300.0, 350.0]
    diodes = [PNJunction(na=1e17, nd=1e16, temperature=t) for t in temperatures]

    print("T [K]   Vbi [V]   Is [A]        W(0V) [um]")
    for d in diodes:
        print(
            f"{d.temperature:5.0f}   {d.built_in_potential:.4f}   "
            f"{d.saturation_current:.3e}   {d.depletion_width() * 1e4:.4f}"
        )

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 5))
    v = np.linspace(-0.2, 0.8, 400)
    plot_diode_iv(diodes, v, ax=ax_lin)
    plot_diode_iv(diodes, v, log=True, ax=ax_log)
    ax_lin.set_ylim(-1, 50)
    fig.tight_layout()
    fig.savefig("diode_iv_temperature.png", dpi=150)
    print("Saved diode_iv_temperature.png")


if __name__ == "__main__":
    main()
