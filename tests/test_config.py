"""Tests for TOML configuration loading."""

import tempfile
from pathlib import Path

import pytest

from semisim.config import device_from_dict, load_device
from semisim.devices.diode import PNJunction
from semisim.devices.mosfet import MOSFET

DIODE_TOML = """
[device]
type = "pn_junction"
material = "Ge"

[device.parameters]
na = 1e17
nd = 1e16
area = 2e-4
temperature = 320.0
"""

MOSFET_TOML = """
[device]
type = "mosfet"
material = "Si"

[device.parameters]
na_substrate = 1e17
tox = 1e-6
width = 1e-3
length = 1e-4
lambda_clm = 0.05
"""


def _load_from_string(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "device.toml"
        path.write_text(text)
        return load_device(path)


def test_load_diode_from_toml():
    d = _load_from_string(DIODE_TOML)
    assert isinstance(d, PNJunction)
    assert d.material.name == "Ge"
    assert d.na == 1e17
    assert d.nd == 1e16
    assert d.area == 2e-4
    assert d.temperature == 320.0


def test_load_mosfet_from_toml():
    m = _load_from_string(MOSFET_TOML)
    assert isinstance(m, MOSFET)
    assert m.material.name == "Si"
    assert m.lambda_clm == 0.05
    assert m.threshold_voltage() > 0


def test_device_from_dict_diode_alias():
    d = device_from_dict(
        {"device": {"type": "diode", "parameters": {"na": 1e17, "nd": 1e17}}}
    )
    assert isinstance(d, PNJunction)


def test_unknown_device_type_raises():
    with pytest.raises(ValueError):
        device_from_dict({"device": {"type": "bjt", "parameters": {}}})


def test_missing_device_table_raises():
    with pytest.raises(KeyError):
        device_from_dict({"simulation": {}})


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        device_from_dict(
            {
                "device": {
                    "type": "diode",
                    "material": "GaAs",
                    "parameters": {"na": 1e17, "nd": 1e17},
                }
            }
        )
