"""Tour of semisim 2.0 features.

- Diode with recombination-generation current and avalanche breakdown
- Small-signal diode conductance (dynamic resistance)
- CMOS pair: NMOS + PMOS with velocity saturation
- MOSFET small-signal gm / gds / gm/Id

Run from the repository root:

    python examples/04_v2_features.py
"""

import matplotlib.pyplot as plt
import numpy as np

from semisim import MOSFET, PNJunction
from semisim.plotting import plot_diode_iv, plot_mosfet_output


def main() -> None:
    # --- Diode: ideal vs. two-diode + breakdown -----------------------
    ideal = PNJunction(na=1e18, nd=1e17)
    real = PNJunction(na=1e18, nd=1e17, tau_rg=1e-7, breakdown_voltage="auto")
    print(f"Estimated avalanche BV: {real.breakdown_voltage:.1f} V")
    print(f"Dynamic resistance at 0.65 V: {1 / float(real.conductance(0.65)):.1f} ohm")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    v = np.linspace(-1.2 * real.breakdown_voltage, 0.8, 800)
    plot_diode_iv([ideal, real], v, labels=["Shockley only", "+ R-G + breakdown"],
                  log=True, ax=axes[0, 0])

    # --- CMOS pair with velocity saturation ---------------------------
    common = dict(body_doping=1e17, tox=1e-6, width=1e-3, length=2e-5,
                  mobility_model="caughey-thomas", lambda_clm=0.05)
    nmos = MOSFET(**common, vsat=1e7)
    pmos = MOSFET(**common, channel="p", vsat=8e6)
    for m in (nmos, pmos):
        print(f"{m.channel.upper()}MOS: Vth = {m.threshold_voltage():+.3f} V, "
              f"mu = {m.mobility:.0f} cm^2/Vs")
    plot_mosfet_output(nmos, vgs_values=[1.0, 1.5, 2.0], ax=axes[0, 1])
    plot_mosfet_output(pmos, vgs_values=[-1.0, -1.5, -2.0], ax=axes[1, 0])

    # --- gm/Id design curve -------------------------------------------
    vgs = np.linspace(0.0, 2.0, 400)
    ss = nmos.small_signal(vgs=vgs, vds=1.0)
    axes[1, 1].plot(vgs, ss["gm"] / nmos.current(vgs=vgs, vds=1.0))
    axes[1, 1].axhline(1 / (nmos.body_factor() * nmos.vt), ls=":", color="gray",
                       label="1 / (m VT)")
    axes[1, 1].set_xlabel("$V_{GS}$ [V]")
    axes[1, 1].set_ylabel("$g_m / I_D$ [1/V]")
    axes[1, 1].set_title("NMOS transconductance efficiency")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
