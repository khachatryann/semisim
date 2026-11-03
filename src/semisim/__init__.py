"""semisim: educational semiconductor device physics simulator.

Analytic models of a PN junction diode and a long-channel MOSFET (NMOS and
PMOS) with configurable materials (Si, Ge, GaAs), doping, and temperature. See the README
and ``docs/theory.md`` for the equations and their validity limits.
"""

from .config import device_from_dict, load_device
from .constants import thermal_voltage
from .devices import MOSFET, Device, PNJunction
from .materials import (
    GALLIUM_ARSENIDE,
    GERMANIUM,
    MATERIALS,
    SILICON,
    Material,
    get_material,
)

__version__ = "2.0.0"

__all__ = [
    "GALLIUM_ARSENIDE",
    "GERMANIUM",
    "MATERIALS",
    "MOSFET",
    "SILICON",
    "Device",
    "Material",
    "PNJunction",
    "__version__",
    "device_from_dict",
    "get_material",
    "load_device",
    "thermal_voltage",
]
