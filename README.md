# semisim

[![CI](https://github.com/khachatryann/semisim/actions/workflows/ci.yml/badge.svg)](https://github.com/khachatryann/semisim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

An educational semiconductor device physics simulator in pure Python (NumPy +
Matplotlib). Analytic models of a **PN junction diode** and a **long-channel
MOSFET (NMOS & PMOS)** with configurable materials (Si, Ge, GaAs), doping,
and temperature — numerical results plus publication-style plots of I-V
curves and band diagrams.

> **New in 2.0** — PMOS · GaAs · doping-dependent (Caughey-Thomas) mobility ·
> diode recombination-generation current · avalanche breakdown ·
> MOSFET velocity saturation · small-signal gm/gds/gmb & diode conductance ·
> `semisim` command-line tool. See [CHANGELOG.md](CHANGELOG.md) (incl. migration notes).

This is an engineering-reference and teaching tool built on closed-form
textbook equations (Sze, Pierret). It is deliberately **not** a mesh-based
TCAD solver or a SPICE replacement; see [Model limits](#model-limits).

## Installation

```bash
git clone https://github.com/khachatryann/semisim.git
cd semisim
pip install .          # or: pip install -e ".[dev]" for development
```

Dependencies: NumPy and Matplotlib only.

## Quick start

```python
import numpy as np
from semisim import PNJunction, MOSFET, GERMANIUM

# --- Diode ---
d = PNJunction(na=1e17, nd=1e16, temperature=300.0)
print(d.built_in_potential)        # 0.799 V
print(d.depletion_width() * 1e4)   # 0.33 um
print(d.current(0.6))              # forward current at 0.6 V [A]

d_rs = PNJunction(na=1e17, nd=1e16, series_resistance=5.0)  # exact Lambert-W I-V
d_ge = PNJunction(na=1e17, nd=1e16, material=GERMANIUM)     # germanium diode

# --- MOSFET ---
m = MOSFET(body_doping=1e17, tox=1e-6, width=1e-3, length=1e-4)  # cm units
print(m.threshold_voltage())       # ~0.36 V
i_d = m.current(vgs=1.5, vds=np.linspace(0, 3, 100))
```

New in 2.0:

```python
# Diode: n=2 recombination current + avalanche breakdown (BV estimated)
d = PNJunction(na=1e18, nd=1e17, tau_rg=1e-7, breakdown_voltage="auto")
print(d.breakdown_voltage)         # ~11.8 V
print(1 / d.conductance(0.65))     # dynamic resistance [ohm]

# PMOS with velocity saturation and doping-dependent mobility
p = MOSFET(body_doping=1e17, tox=1e-6, width=2e-3, length=1e-4,
           channel="p", vsat=8e6, mobility_model="caughey-thomas")
print(p.threshold_voltage())       # ~-0.36 V
print(p.current(vgs=-2.0, vds=-2.0))           # negative: flows out of drain
print(p.small_signal(vgs=-2.0, vds=-2.0)["gm"])  # transconductance [S]
```

Plotting:

```python
import matplotlib.pyplot as plt
from semisim.plotting import plot_diode_iv, plot_band_diagram, plot_mosfet_output

plot_band_diagram(d, bias=0.3)
plot_mosfet_output(m, vgs_values=[1.0, 1.5, 2.0])
plt.show()
```

Devices can also be defined in TOML (see `examples/diode.toml`):

```python
from semisim import load_device
d = load_device("examples/diode.toml")
```

…or from the command line:

```bash
semisim info examples/pmos.toml                 # parameter summary
semisim plot examples/zener.toml --log -o z.png # I-V + band diagram
```

### Examples

```bash
python examples/01_diode_iv_temperature.py   # diode I-V at 250/300/350 K
python examples/02_diode_si_vs_ge.py         # Si vs Ge: Vbi, leakage, band diagrams
python examples/03_mosfet_characteristics.py # output/transfer curves, Vth extraction
python examples/04_v2_features.py            # breakdown, CMOS pair, gm/Id
```

## Units

Semiconductor-textbook cm-based units throughout: densities in cm⁻³, lengths
in cm (10 nm oxide = `1e-6`), mobility in cm²/(V·s), capacitance in F/cm².

## Physics

Full derivations and references in [`docs/theory.md`](docs/theory.md).

**Materials** (Si, Ge, GaAs) — Varshni bandgap `Eg(T) = Eg(0) − αT²/(T+β)`;
intrinsic concentration `ni = √(Nc Nv) exp(−Eg/2kT)` with `Nc,Nv ∝ T^1.5`;
power-law mobility temperature scaling; optional Caughey-Thomas doping
dependence `μ = μmin + (μmax−μmin)/(1+(N/Nref)^α)`; Einstein relation `D = μ kT/q`.

**PN junction** (abrupt, full-depletion approximation):

- Built-in potential: `Vbi = (kT/q) ln(Na Nd / ni²)`
- Depletion width: `W = √(2ε(Vbi−V)(Na+Nd)/(q Na Nd))`, junction capacitance `Cj = εA/W`
- Shockley I-V: `I = Is (exp(V/nVT) − 1)` with
  `Is = qA ni² (Dn/(Ln Na) + Dp/(Lp Nd))`
- With series resistance the implicit equation is solved **exactly** via the
  Lambert W function (evaluated in log space — no overflow at any bias):
  `I = (nVT/Rs) · W₀[(Is Rs/nVT) e^((V+Is Rs)/nVT)] − Is`
- Recombination-generation current (opt-in): `Irg = (qA ni W/2τ)(exp(V/2VT) − 1)`
- Reverse breakdown knee (opt-in), BV from Sze-Gibbons:
  `BV ≈ 60 (Eg/1.1)^1.5 (N_B/1e16)^−0.75`
- Analytic small-signal conductance `g = gj/(1 + Rs gj)`
- Band diagrams in the depletion approximation, with quasi-Fermi splitting
  under bias (illustrative).

**MOSFET** (long-channel square law, NMOS and PMOS):

- Threshold: `Vth = Vfb + 2φF + γ√(2φF + Vsb)`, with `φF = VT ln(Na/ni)`,
  `γ = √(2qεs Na)/Cox`, and `Vfb` defaulting to an ideal n⁺-poly gate
- Triode: `Id = μCox(W/L)(Vov − Vds/2)Vds`; saturation: `Id = (μCox/2)(W/L)Vov²`,
  both with optional channel-length modulation `(1 + λVds)`
- Subthreshold: EKV-style softplus interpolation of the overdrive giving
  `Id ∝ exp((Vgs−Vth)/mVT)` with `m = 1 + γ/(2√(2φF))` and a smooth handoff
  at threshold
- Velocity saturation (opt-in): `Ec = 2vsat/μ`,
  `Vdsat = Vov·EcL/(Vov + EcL)`, triode current divided by `(1 + Vds/EcL)`
- PMOS as the exact mirror image (signed voltages, negative drain current)
- Small-signal `gm`, `gds`, `gmb`, `ro`

## Model limits

Simplifications you should know about before trusting a number:

1. Boltzmann statistics only — degenerate doping (> 10¹⁹ cm⁻³) triggers a warning.
2. Breakdown is an empirical exponential knee; the auto BV estimate is ±20 %
   and Zener (tunneling) physics is not modeled.
3. No high-level injection or band-gap narrowing in the diode.
4. Mobility temperature scaling is a single power law (also applied to the
   Caughey-Thomas curve); MOSFET default mobility is the bulk value
   (surface mobility is ~2× lower — override `mobility=` for quantitative work).
5. MOSFET: velocity saturation is included (opt-in) but there is no DIBL,
   Vth roll-off, or vertical-field mobility degradation.
6. Band diagrams under bias use flat quasi-Fermi splits, not a solved
   drift-diffusion profile.

## Architecture

```
src/semisim/
├── constants.py       # physical constants (CODATA), thermal voltage
├── materials.py       # Material dataclass, Si/Ge/GaAs presets, Caughey-Thomas
├── _math.py           # pure-NumPy Lambert W (log-space), vectorized bisection
├── config.py          # TOML -> device instances
├── plotting.py        # I-V, band diagram, output/transfer plots
├── cli.py             # `semisim info|plot` command-line tool
└── devices/
    ├── base.py        # abstract Device (terminals, current(), parameters)
    ├── diode.py       # PNJunction
    └── mosfet.py      # MOSFET (NMOS/PMOS)
```

All devices share the `Device` interface (named terminals, uniform
`current(**bias)`) and small-signal derivatives, so a SPICE-style
netlist/nodal-analysis layer can be added later without touching the physics
classes. See
[`docs/architecture.md`](docs/architecture.md).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

CI runs both on every push/PR across Python 3.10–3.13. Pushing a `v*` tag
builds the wheel/sdist and publishes a GitHub Release automatically. Contributions welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md).

## References

- S. M. Sze, K. K. Ng, *Physics of Semiconductor Devices*, 3rd ed., Wiley, 2007.
- R. F. Pierret, *Semiconductor Device Fundamentals*, Addison-Wesley, 1996.
- T. C. Banwell, A. Jayakumar, "Exact analytical solution for current flow
  through diode with series resistance," *Electron. Lett.* 36(4), 2000.
- Y. Tsividis, C. McAndrew, *Operation and Modeling of the MOS Transistor*,
  3rd ed., Oxford, 2011 (subthreshold interpolation).
- C. Jacoboni et al., "A review of some charge transport properties of
  silicon," *Solid-State Electron.* 20, 1977 (Caughey-Thomas parameters).
- M. Sotoodeh et al., "Empirical low-field mobility model for III-V
  compounds," *J. Appl. Phys.* 87, 2000.
- C. T. Sah, R. N. Noyce, W. Shockley, "Carrier generation and recombination
  in P-N junctions," *Proc. IRE* 45, 1957.

## License

MIT — see [LICENSE](LICENSE).
