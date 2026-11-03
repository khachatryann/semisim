# Changelog

All notable changes to semisim are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-11-03

### Added
- **PMOS transistors** — `MOSFET(..., channel="p")` (or TOML `type = "pmos"`),
  evaluated as the exact mirror image of the NMOS model with a p⁺-poly gate.
- **GaAs** material preset (`GALLIUM_ARSENIDE`, `"GaAs"` in configs).
- **Doping-dependent mobility** (Caughey-Thomas) for Si, Ge, GaAs:
  `Material.mobility_n/p(T, doping)`; opt-in on devices via
  `mobility_model="caughey-thomas"`.
- **Diode recombination-generation current** (`tau_rg=`): reverse generation
  leakage and the n = 2 low-bias component.
- **Diode reverse breakdown** (`breakdown_voltage=` number or `"auto"`),
  plus `PNJunction.avalanche_breakdown_voltage()` (Sze-Gibbons estimate).
- **Small-signal analysis** — `PNJunction.conductance(v)` (analytic) and
  `MOSFET.small_signal(vgs, vds, vsb)` → `gm`, `gds`, `gmb`, `ro`.
- **Velocity saturation** for the MOSFET (`vsat=`), with
  `MOSFET.saturation_voltage()` and `MOSFET.critical_field()`.
- `PNJunction.junction_voltage(v)` (voltage across the junction after Rs drop).
- **Command-line tool** — `semisim info <file.toml>` and
  `semisim plot <file.toml> [-o out.png] [--log]` (also `python -m semisim`).
- Examples: `examples/04_v2_features.py`, `examples/pmos.toml`,
  `examples/zener.toml`.
- Release workflow: pushing a `v*` tag builds and publishes a GitHub Release.
- Python 3.13 support in CI and classifiers.

### Changed
- **Breaking:** `MOSFET`'s first argument is now `body_doping` (the body
  doping magnitude for either channel type). Positional calls are unaffected.
- **Breaking:** `MOSFET.parameters` reports `body_doping [cm^-3]` instead of
  `na_substrate [cm^-3]`, and adds `channel` and `vsat [cm/s]`.
- `"GaAs"` is now a valid material name (it previously raised `KeyError`).
- `plot_mosfet_output` / `plot_mosfet_transfer` sweep negative voltages for
  PMOS automatically; the log transfer plot shows |Id|.
- Development status raised to *Production/Stable*.

### Deprecated
- `MOSFET(na_substrate=...)` and the `MOSFET.na_substrate` attribute — use
  `body_doping`. Still works, emits `DeprecationWarning`; removal planned for 3.0.

### Migration from 0.1
| 0.1 | 2.0 |
|-----|-----|
| `MOSFET(na_substrate=1e17, ...)` | `MOSFET(body_doping=1e17, ...)` |
| `m.parameters["na_substrate [cm^-3]"]` | `m.parameters["body_doping [cm^-3]"]` |
| TOML `na_substrate = 1e17` | TOML `body_doping = 1e17` |

All new physics is opt-in: with default arguments every 0.1 result is
reproduced exactly.

## [0.1.0]

Initial release: analytic PN junction (Shockley + exact Lambert-W series
resistance, band diagrams) and long-channel NMOS (square law, body effect,
CLM, subthreshold) for Si and Ge, TOML configs, and plotting helpers.
