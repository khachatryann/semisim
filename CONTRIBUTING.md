# Contributing to semisim

Thanks for your interest in contributing!

## Getting started

```bash
git clone https://github.com/khachatryann/semisim.git
cd semisim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Development workflow

1. Open an issue first for non-trivial changes so the approach can be discussed.
2. Create a feature branch from `main`.
3. Make your changes, keeping commits focused.
4. Run the checks locally before pushing:

```bash
ruff check .
pytest
```

Both must pass; CI runs them on every push and pull request.

## Guidelines

- **Physics first.** Every model change must cite the equation/reference it implements
  (add it to `docs/theory.md`) and state its assumptions and validity range.
- **Tests alongside code.** New physics gets a test that checks it against a hand
  calculation or a textbook value, plus at least one edge case (invalid input,
  extreme temperature/doping).
- **Docstrings** on all public functions and classes, NumPy style, with units for
  every physical parameter. This codebase uses cm-based semiconductor units
  (cm^-3, cm^2/(V·s), F/cm) throughout.
- **No new hard dependencies** without discussion — the tool intentionally depends
  only on NumPy and Matplotlib.
- Keep device models subclasses of `semisim.devices.base.Device` so future
  circuit-level (netlist) extensions can treat all devices uniformly.

## Reporting bugs

Please include: semisim version (`semisim --version`), Python version, a minimal script reproducing the
issue, and — for physics discrepancies — the reference value and its source.

## License

By contributing you agree that your contributions are licensed under the MIT License.
