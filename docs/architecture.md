# Architecture

## Layout

```
src/semisim/
├── constants.py       # CODATA constants, thermal_voltage()
├── materials.py       # Material dataclass, Si/Ge/GaAs, Caughey-Thomas
├── _math.py           # numerical helpers (log-space Lambert W, bisection)
├── config.py          # TOML -> Device factory
├── plotting.py        # matplotlib helpers (all return Axes)
├── cli.py             # `semisim info|plot` console script
└── devices/
    ├── base.py        # abstract Device
    ├── diode.py       # PNJunction
    └── mosfet.py      # MOSFET (NMOS/PMOS)
```

## Design decisions

**Physics/presentation split.** Device classes compute; `plotting.py` draws.
Every plot helper takes an optional `ax` and returns it, so figures compose.

**Materials are immutable value objects.** `Material` is a frozen dataclass;
devices hold a reference and evaluate temperature-dependent properties on
demand. Adding a material (e.g. GaAs) means adding one constant and a
registry entry — no device code changes.

**Uniform device interface for future circuit simulation.** `Device`
prescribes:

- `terminals`: ordered terminal names,
- `current(**bias)`: vectorized terminal current,
- `parameters`: an inspectable dict.

A SPICE-style netlist layer would map node voltages to each device's named
terminals and call `current()` inside a Newton loop for nodal analysis.
Since v2 the Jacobian entries exist too: `PNJunction.conductance()` is
analytic and `MOSFET.small_signal()` returns gm/gds/gmb.

**PMOS by mirroring.** Rather than duplicating equations, `MOSFET`
evaluates everything in NMOS-normalized coordinates (`_id_normalized`) and
applies the polarity sign at the boundary. Any future model term is
automatically available to both channel types.

**Robust implicit solves.** The pure Shockley + Rs diode keeps its exact
Lambert-W solution. Once extra junction terms exist (R-G, breakdown) there
is no closed form, so `_math.bisect_monotonic` solves for the junction
voltage: the junction I(V) is strictly increasing, so bisection is
unconditionally convergent and overflow-tolerant (±inf compare correctly).

**Config as a thin factory.** TOML keys map 1:1 to constructor keyword
arguments; validation lives in the constructors so programmatic and config
paths behave identically.

**Errors vs. warnings.** Inputs outside the model's mathematical domain
(negative doping, bias ≥ Vbi in the depletion formula, Vds < 0) raise
`ValueError`. Inputs that are computable but physically questionable
(degenerate doping) emit `UserWarning` and proceed.

**No SciPy.** The only non-elementary function needed is Lambert W for the
diode series-resistance solution; `_math.lambertw_log` implements it in
~30 lines of NumPy, evaluated in log space so arguments like
`exp(V/VT)` at V = 100 V cannot overflow. This keeps the dependency
footprint to NumPy + Matplotlib.

## Extension points

- **New device:** subclass `Device` in `devices/`, register in
  `config._DEVICE_TYPES`, add tests mirroring `tests/test_diode.py`.
- **New material:** add a `Material` constant and registry entry in
  `materials.py`.
- **Better mobility model:** extend `Material.mobility_*` (e.g. vertical-
  field degradation for MOS); devices only call these methods.
- **Circuit level:** a `Netlist` class performing Newton nodal analysis
  over device instances, using `conductance()` / `small_signal()`.
