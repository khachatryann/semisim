# Architecture

## Layout

```
src/semisim/
├── constants.py       # CODATA constants, thermal_voltage()
├── materials.py       # Material dataclass, SILICON/GERMANIUM, registry
├── _math.py           # numerical helpers (log-space Lambert W)
├── config.py          # TOML -> Device factory
├── plotting.py        # matplotlib helpers (all return Axes)
└── devices/
    ├── base.py        # abstract Device
    ├── diode.py       # PNJunction
    └── mosfet.py      # MOSFET
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
terminals and call `current()` inside a Newton loop for nodal analysis. To
support that fully, the natural next step is adding
`conductance(**bias)` (analytic derivatives) to the interface — the models
are closed-form, so the derivatives are straightforward.

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
- **Better mobility model:** replace `Material.mobility_*`
  with a doping-aware model (Caughey-Thomas); the device classes only call
  these methods, so nothing else changes.
- **Circuit level:** add `conductance()` to `Device`, then a `Netlist`
  class performing Newton nodal analysis over device instances.
