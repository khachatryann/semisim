"""Tests for materials and constants: bandgap, ni, mobility, validation."""

import numpy as np
import pytest

from semisim.constants import K_B_EV, thermal_voltage
from semisim.materials import (
    GERMANIUM,
    SILICON,
    check_doping,
    get_material,
)


def test_thermal_voltage_300k():
    assert thermal_voltage(300.0) == pytest.approx(0.025852, rel=1e-4)


def test_thermal_voltage_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        thermal_voltage(0.0)
    with pytest.raises(ValueError):
        thermal_voltage(-50.0)


@pytest.mark.parametrize(
    "material,eg_300",
    [(SILICON, 1.1245), (GERMANIUM, 0.6634)],
)
def test_bandgap_at_300k(material, eg_300):
    assert material.bandgap(300.0) == pytest.approx(eg_300, abs=2e-3)


def test_bandgap_decreases_with_temperature():
    assert SILICON.bandgap(400.0) < SILICON.bandgap(300.0) < SILICON.bandgap(200.0)


def test_bandgap_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        SILICON.bandgap(-1.0)


def test_intrinsic_concentration_silicon_order_of_magnitude():
    # Literature: ~1e10 cm^-3; tabulated Nc/Nv give ~6e9. Factor-2 agreement.
    ni = SILICON.ni(300.0)
    assert 4e9 < ni < 1.6e10


def test_intrinsic_concentration_germanium_order_of_magnitude():
    # Literature: ~2.3e13 cm^-3.
    ni = GERMANIUM.ni(300.0)
    assert 1e13 < ni < 4e13


def test_ni_matches_closed_form():
    t = 320.0
    expected = np.sqrt(SILICON.nc(t) * SILICON.nv(t)) * np.exp(
        -SILICON.bandgap(t) / (2 * K_B_EV * t)
    )
    assert SILICON.ni(t) == pytest.approx(expected, rel=1e-12)


def test_ni_increases_strongly_with_temperature():
    assert SILICON.ni(350.0) / SILICON.ni(300.0) > 10.0


def test_effective_dos_scaling():
    assert SILICON.nc(600.0) == pytest.approx(SILICON.nc_300 * 2**1.5, rel=1e-12)
    assert SILICON.nv(150.0) == pytest.approx(SILICON.nv_300 * 0.5**1.5, rel=1e-12)


def test_mobility_decreases_with_temperature():
    assert SILICON.mobility_n(400.0) < SILICON.mobility_n(300.0)
    assert SILICON.mobility_p(400.0) < SILICON.mobility_p(300.0)


def test_einstein_relation():
    t = 300.0
    ratio = SILICON.diffusion_n(t) / SILICON.mobility_n(t)
    assert ratio == pytest.approx(thermal_voltage(t), rel=1e-12)


def test_germanium_has_higher_mobility_and_lower_gap_than_silicon():
    assert GERMANIUM.mu_n_300 > SILICON.mu_n_300
    assert GERMANIUM.bandgap(300.0) < SILICON.bandgap(300.0)


def test_get_material_aliases():
    assert get_material("Si") is SILICON
    assert get_material("silicon") is SILICON
    assert get_material("GERMANIUM") is GERMANIUM


def test_get_material_unknown_raises():
    with pytest.raises(KeyError):
        get_material("InP")


def test_check_doping_rejects_nonpositive():
    with pytest.raises(ValueError):
        check_doping(0.0)
    with pytest.raises(ValueError):
        check_doping(-1e16)


def test_check_doping_warns_for_degenerate():
    with pytest.warns(UserWarning):
        check_doping(5e19)
