"""Silicon vs. germanium PN junctions: Vbi, leakage, band diagrams.

Shows why Ge diodes turn on around 0.3 V but leak ~10^6 x more than Si
(smaller bandgap -> much larger ni), and compares equilibrium band
diagrams.

Run from the repository root:

    python examples/02_diode_si_vs_ge.py
"""

import matplotlib.pyplot as plt
import numpy as np

from semisim import GERMANIUM, SILICON, PNJunction
from semisim.plotting import plot_band_diagram, plot_diode_iv


def main() -> None:
    si = PNJunction(na=1e17, nd=1e16, material=SILICON)
    ge = PNJunction(na=1e17, nd=1e16, material=GERMANIUM)

    print("Material  Eg [eV]   ni [cm^-3]   Vbi [V]   Is [A]        W(0V) [um]")
    for d in (si, ge):
        print(
            f"{d.material.name:8s}  {d.material.bandgap(d.temperature):.4f}   "
            f"{d.ni:.3e}    {d.built_in_potential:.4f}   "
            f"{d.saturation_current:.3e}   {d.depletion_width() * 1e4:.4f}"
        )
    print(f"\nIs(Ge) / Is(Si) = {ge.saturation_current / si.saturation_current:.3g}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    v = np.linspace(-0.3, 0.8, 400)
    plot_diode_iv([si, ge], v, labels=["Si", "Ge"], log=True, ax=axes[0])
    plot_band_diagram(si, ax=axes[1])
    plot_band_diagram(ge, ax=axes[2])
    fig.tight_layout()
    fig.savefig("diode_si_vs_ge.png", dpi=150)
    print("Saved diode_si_vs_ge.png")


if __name__ == "__main__":
    main()
