"""Tests for the PN junction diode: electrostatics, I-V, band diagram."""

import numpy as np
import pytest

from semisim.constants import Q
from semisim.devices.diode import PNJunction
from semisim.materials import GERMANIUM, SILICON


def make_diode(**overrides):
    kwargs = dict(na=1e17, nd=1e17, area=1e-4, temperature=300.0)
    kwargs.update(overrides)
    return PNJunction(**kwargs)


# ----------------------------------------------------------------------
# Electrostatics
# ----------------------------------------------------------------------
def test_built_in_potential_hand_calculation():
    d = make_diode()
    # Vbi = VT ln(Na Nd / ni^2); hand value with tabulated Si parameters.
    assert d.built_in_potential == pytest.approx(0.8588, abs=5e-3)
    expected = d.vt * np.log(d.na * d.nd / d.ni**2)
    assert d.built_in_potential == pytest.approx(expected, rel=1e-12)


def test_built_in_potential_below_bandgap():
    d = make_diode()
    assert 0.0 < d.built_in_potential < SILICON.bandgap(300.0)


def test_germanium_has_lower_built_in_potential():
    si = make_diode()
    ge = make_diode(material=GERMANIUM)
    assert ge.built_in_potential < si.built_in_potential


def test_depletion_width_hand_calculation():
    d = make_diode()
    # W = sqrt(2 eps Vbi (Na+Nd) / (q Na Nd)) ~ 0.149 um for 1e17/1e17 Si.
    assert d.depletion_width() == pytest.approx(1.490e-5, rel=2e-2)
    eps = SILICON.permittivity
    expected = np.sqrt(
        2 * eps * d.built_in_potential * (d.na + d.nd) / (Q * d.na * d.nd)
    )
    assert d.depletion_width() == pytest.approx(expected, rel=1e-12)


def test_depletion_width_bias_dependence():
    d = make_diode()
    assert d.depletion_width(0.4) < d.depletion_width(0.0) < d.depletion_width(-5.0)


def test_depletion_width_invalid_beyond_vbi():
    d = make_diode()
    with pytest.raises(ValueError):
        d.depletion_width(d.built_in_potential + 0.01)


def test_depletion_edges_charge_neutrality():
    d = make_diode(na=1e18, nd=1e16)
    xp, xn = d.depletion_edges()
    assert xp * d.na == pytest.approx(xn * d.nd, rel=1e-12)
    assert xp + xn == pytest.approx(d.depletion_width(), rel=1e-12)
    # Depletion extends mostly into the lightly doped side.
    assert xn > 10 * xp


def test_junction_capacitance():
    d = make_diode()
    expected = SILICON.permittivity * d.area / d.depletion_width()
    assert d.junction_capacitance() == pytest.approx(expected, rel=1e-12)
    assert d.junction_capacitance(-5.0) < d.junction_capacitance(0.0)


# ----------------------------------------------------------------------
# Shockley I-V (no series resistance)
# ----------------------------------------------------------------------
def test_zero_bias_zero_current():
    assert float(make_diode().current(0.0)) == 0.0


def test_forward_slope_60mv_per_decade():
    d = make_diode()
    ratio = float(d.current(0.46) / d.current(0.40))
    assert ratio == pytest.approx(np.exp(0.06 / d.vt), rel=1e-3)


def test_ideality_factor_halves_slope():
    d = make_diode(ideality=2.0)
    ratio = float(d.current(0.46) / d.current(0.40))
    assert ratio == pytest.approx(np.exp(0.06 / (2 * d.vt)), rel=1e-3)


def test_reverse_current_saturates_at_minus_is():
    d = make_diode()
    assert float(d.current(-0.5)) == pytest.approx(-d.saturation_current, rel=1e-6)


def test_saturation_current_scales_with_area_and_temperature():
    small = make_diode(area=1e-4)
    large = make_diode(area=2e-4)
    assert large.saturation_current == pytest.approx(
        2 * small.saturation_current, rel=1e-12
    )
    hot = make_diode(temperature=350.0)
    assert hot.saturation_current > 10 * small.saturation_current


def test_germanium_leaks_more_than_silicon():
    si = make_diode()
    ge = make_diode(material=GERMANIUM)
    assert ge.saturation_current > 1e3 * si.saturation_current


def test_current_vectorized():
    d = make_diode()
    v = np.linspace(-1.0, 0.7, 25)
    i = d.current(v)
    assert i.shape == v.shape
    assert np.all(np.diff(i) >= 0)  # monotonic


# ----------------------------------------------------------------------
# Series resistance (Lambert W)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("v", [0.2, 0.5, 0.8, 1.0])
def test_lambert_solution_satisfies_implicit_equation(v):
    d = make_diode(series_resistance=10.0)
    i = float(d.current(v))
    i_s = d.saturation_current
    residual = i_s * np.expm1((v - i * d.series_resistance) / d.vt)
    assert i == pytest.approx(residual, rel=1e-8)


def test_lambert_matches_ideal_for_negligible_rs():
    ideal = make_diode()
    tiny_rs = make_diode(series_resistance=1e-9)
    v = 0.6
    assert float(tiny_rs.current(v)) == pytest.approx(
        float(ideal.current(v)), rel=1e-6
    )


def test_high_bias_slope_approaches_one_over_rs():
    rs = 10.0
    d = make_diode(series_resistance=rs)
    slope = float((d.current(2.0) - d.current(1.9)) / 0.1)
    assert slope == pytest.approx(1.0 / rs, rel=0.1)


def test_no_overflow_at_large_forward_bias():
    d = make_diode(series_resistance=1.0)
    i = d.current(np.array([5.0, 20.0, 100.0]))
    assert np.all(np.isfinite(i))
    assert np.all(np.diff(i) > 0)


# ----------------------------------------------------------------------
# Band diagram
# ----------------------------------------------------------------------
def test_band_diagram_equilibrium():
    d = make_diode()
    bd = d.band_diagram()
    eg = SILICON.bandgap(300.0)
    assert np.allclose(bd["ec"] - bd["ev"], eg)
    # Total band bending equals Vbi at equilibrium.
    bending = bd["ec"][0] - bd["ec"][-1]
    assert bending == pytest.approx(d.built_in_potential, rel=1e-6)


def test_band_diagram_forward_bias_reduces_bending_and_splits_fermi_levels():
    d = make_diode()
    bias = 0.3
    bd = d.band_diagram(bias)
    bending = bd["ec"][0] - bd["ec"][-1]
    assert bending == pytest.approx(d.built_in_potential - bias, rel=1e-6)
    overlap = ~(np.isnan(bd["efn"]) | np.isnan(bd["efp"]))
    split = bd["efn"][overlap] - bd["efp"][overlap]
    assert np.allclose(split, bias)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        make_diode(na=-1e17)
    with pytest.raises(ValueError):
        make_diode(area=0.0)
    with pytest.raises(ValueError):
        make_diode(temperature=-10.0)
    with pytest.raises(ValueError):
        make_diode(ideality=0.5)
    with pytest.raises(ValueError):
        make_diode(series_resistance=-1.0)
    with pytest.raises(ValueError):
        make_diode(tau_n=0.0)


def test_degenerate_doping_warns():
    with pytest.warns(UserWarning):
        make_diode(na=5e19)


def test_device_interface():
    d = make_diode()
    assert d.terminals == ("anode", "cathode")
    assert "built_in_potential [V]" in d.parameters
    assert float(d.iv(0.3)) == pytest.approx(float(d.current(0.3)), rel=1e-12)
    assert "PNJunction" in d.summary()
