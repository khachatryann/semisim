"""Build device instances from TOML configuration files.

Example configuration::

    [device]
    type = "pn_junction"          # or "mosfet", "nmos", "pmos"
    material = "Si"               # or "Ge", "GaAs"

    [device.parameters]
    na = 1e17
    nd = 1e16
    area = 1e-4
    temperature = 300.0

``parameters`` keys map directly onto the keyword arguments of
:class:`~semisim.devices.diode.PNJunction` /
:class:`~semisim.devices.mosfet.MOSFET`.
"""

import sys
from pathlib import Path
from typing import Any

from .devices.base import Device
from .devices.diode import PNJunction
from .devices.mosfet import MOSFET
from .materials import get_material

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python < 3.11
    import tomli as tomllib

_DEVICE_TYPES: dict[str, type[Device]] = {
    "pn_junction": PNJunction,
    "diode": PNJunction,
    "mosfet": MOSFET,
    "nmos": MOSFET,
    "pmos": MOSFET,
}


def device_from_dict(data: dict[str, Any]) -> Device:
    """Instantiate a device from a configuration dictionary.

    Parameters
    ----------
    data : dict
        Must contain a ``"device"`` table with ``type``, optional
        ``material``, and a ``parameters`` sub-table of keyword arguments.

    Returns
    -------
    Device

    Raises
    ------
    KeyError
        If the ``device`` table or ``type`` key is missing.
    ValueError
        If the device type is unknown.
    """
    try:
        dev = data["device"]
        dev_type = dev["type"]
    except KeyError as exc:
        raise KeyError("Configuration must contain [device] with a 'type' key") from exc

    key = str(dev_type).strip().lower()
    if key not in _DEVICE_TYPES:
        raise ValueError(
            f"Unknown device type {dev_type!r}; available: {sorted(set(_DEVICE_TYPES))}"
        )

    kwargs: dict[str, Any] = dict(dev.get("parameters", {}))
    if "material" in dev:
        kwargs["material"] = get_material(dev["material"])
    if key in ("nmos", "pmos"):
        if "channel" in kwargs and str(kwargs["channel"]).lower()[0] != key[0]:
            raise ValueError(f"device type {key!r} conflicts with channel={kwargs['channel']!r}")
        kwargs["channel"] = key[0]
    return _DEVICE_TYPES[key](**kwargs)


def load_device(path: str | Path) -> Device:
    """Load a device from a TOML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the TOML file.

    Returns
    -------
    Device
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return device_from_dict(data)
