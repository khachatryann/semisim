"""semisim: educational semiconductor device physics simulator.

Analytic models of a PN junction diode and a long-channel MOSFET with
configurable materials (Si, Ge), doping, and temperature. See the README
and ``docs/theory.md`` for the equations and their validity limits.
"""

from .config import device_from_dict, load_device
from .constants import thermal_voltage
from .devices import MOSFET, Device, PNJunction
from .materials import GERMANIUM, MATERIALS, SILICON, Material, get_material

__version__ = "0.1.0"

__all__ = [
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
