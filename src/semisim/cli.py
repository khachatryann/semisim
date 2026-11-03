"""Command-line interface: ``semisim info|plot <device.toml>``.

Examples::

    semisim info examples/diode.toml
    semisim plot examples/diode.toml -o diode.png
    semisim plot examples/pmos.toml --log
"""

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .config import load_device
from .devices.diode import PNJunction
from .devices.mosfet import MOSFET


def _cmd_info(args: argparse.Namespace) -> int:
    print(load_device(args.config).summary())
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .plotting import (
        plot_band_diagram,
        plot_diode_iv,
        plot_mosfet_output,
        plot_mosfet_transfer,
    )

    device = load_device(args.config)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if isinstance(device, PNJunction):
        lo = -1.2 * device.breakdown_voltage if device.breakdown_voltage else -1.0
        plot_diode_iv(device, np.linspace(lo, 0.9, 600), log=args.log, ax=ax1)
        plot_band_diagram(device, bias=0.0, ax=ax2)
    elif isinstance(device, MOSFET):
        p = device.polarity
        vth = device.threshold_voltage()
        vgs = [vth + p * dv for dv in (0.5, 1.0, 1.5, 2.0)]
        plot_mosfet_output(device, vgs_values=[round(v, 2) for v in vgs], ax=ax1)
        plot_mosfet_transfer(device, vds=p * 0.1, log=args.log, ax=ax2)
    else:  # pragma: no cover - only reachable with a custom registered device
        print(f"No plot recipe for {type(device).__name__}", file=sys.stderr)
        return 1
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=args.dpi)
        print(f"saved {args.output}")
    else:  # pragma: no cover - interactive
        plt.show()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``semisim`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="semisim", description="Educational semiconductor device simulator."
    )
    parser.add_argument("--version", action="version", version=f"semisim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="print device parameters from a TOML file")
    info.add_argument("config", help="device TOML file")
    info.set_defaults(func=_cmd_info)

    plot = sub.add_parser("plot", help="plot characteristics of a TOML-defined device")
    plot.add_argument("config", help="device TOML file")
    plot.add_argument("-o", "--output", help="save to this image file instead of showing")
    plot.add_argument("--log", action="store_true", help="log-scale current axis")
    plot.add_argument("--dpi", type=int, default=150, help="output resolution (default 150)")
    plot.set_defaults(func=_cmd_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``semisim`` console script."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"semisim: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
