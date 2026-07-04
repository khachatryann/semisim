"""MOSFET output/transfer characteristics and Vth extraction.

Sweeps a 1 um NMOS device, prints the model's threshold voltage and
subthreshold swing, and extracts Vth from the transfer curve by linear
extrapolation at the point of maximum transconductance (the standard
experimental method) for comparison.

Run from the repository root:

    python examples/03_mosfet_characteristics.py
"""

import matplotlib.pyplot as plt
import numpy as np

from semisim import MOSFET
from semisim.plotting import plot_mosfet_output, plot_mosfet_transfer


def extract_vth_linear_extrapolation(mosfet: MOSFET, vds: float = 0.05) -> float:
    """Extract Vth by extrapolating Id(Vgs) at max gm to Id = 0 (linear region)."""
    vgs = np.linspace(0.0, 3.0, 601)
    i_d = mosfet.current(vgs=vgs, vds=vds)
    gm = np.gradient(i_d, vgs)
    k = int(np.argmax(gm))
    # Tangent at max-gm point: Id = gm (Vgs - Vgs_k) + Id_k -> zero crossing
    vth_ext = vgs[k] - i_d[k] / gm[k]
    return float(vth_ext - vds / 2.0)  # standard Vds/2 correction


def main() -> None:
    nmos = MOSFET(
        na_substrate=1e17,
        tox=1e-6,  # 10 nm
        width=1e-3,  # 10 um
        length=1e-4,  # 1 um
        lambda_clm=0.05,
    )
    print(nmos.summary())

    vth_model = nmos.threshold_voltage()
    vth_extracted = extract_vth_linear_extrapolation(nmos)
    print(f"\nModel Vth       = {vth_model:.4f} V")
    print(f"Extracted Vth   = {vth_extracted:.4f} V (linear extrapolation)")
    print(f"Subthreshold SS = {nmos.subthreshold_swing() * 1e3:.1f} mV/decade")
    print(f"Body effect: Vth(Vsb=1 V) = {nmos.threshold_voltage(1.0):.4f} V")

    fig, (ax_out, ax_tr) = plt.subplots(1, 2, figsize=(12, 5))
    plot_mosfet_output(nmos, vgs_values=[0.8, 1.2, 1.6, 2.0], vds_max=3.0, ax=ax_out)
    plot_mosfet_transfer(nmos, vds=1.0, vgs_range=(0.0, 2.0), log=True, ax=ax_tr)
    fig.tight_layout()
    fig.savefig("mosfet_characteristics.png", dpi=150)
    print("Saved mosfet_characteristics.png")


if __name__ == "__main__":
    main()
