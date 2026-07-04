"""Tests for the MOSFET square-law model."""

import numpy as np
import pytest

from semisim.constants import EPS_0, EPS_OX_R, Q
from semisim.devices.mosfet import MOSFET
from semisim.materials import SILICON


def make_mosfet(**overrides):
    kwargs = dict(
        na_substrate=1e17,
        tox=1e-6,  # 10 nm
        width=1e-3,  # 10 um
        length=1e-4,  # 1 um
        temperature=300.0,
    )
    kwargs.update(overrides)
    return MOSFET(**kwargs)


# ----------------------------------------------------------------------
# Electrostatics / threshold
# ----------------------------------------------------------------------
def test_oxide_capacitance():
    m = make_mosfet()
    assert m.cox == pytest.approx(EPS_OX_R * EPS_0 / 1e-6, rel=1e-12)
    assert m.cox == pytest.approx(3.453e-7, rel=1e-3)


def test_fermi_potential_hand_calculation():
    m = make_mosfet()
    assert m.phi_f == pytest.approx(0.4294, abs=5e-3)


def test_body_effect_coefficient_hand_calculation():
    m = make_mosfet()
    expected = np.sqrt(2 * Q * SILICON.permittivity * 1e17) / m.cox
    assert m.gamma == pytest.approx(expected, rel=1e-12)
    assert m.gamma == pytest.approx(0.5276, rel=1e-2)


def test_threshold_voltage_hand_calculation():
    m = make_mosfet()
    # Vfb(n+ poly) = -(Eg/2 + phi_F); Vth = Vfb + 2 phi_F + gamma sqrt(2 phi_F)
    vfb = -(SILICON.bandgap(300.0) / 2 + m.phi_f)
    expected = vfb + 2 * m.phi_f + m.gamma * np.sqrt(2 * m.phi_f)
    assert m.threshold_voltage() == pytest.approx(expected, rel=1e-12)
    assert m.threshold_voltage() == pytest.approx(0.356, abs=0.02)


def test_user_flat_band_voltage_override():
    m = make_mosfet(vfb=-0.5)
    assert m.flat_band_voltage == -0.5
    assert m.threshold_voltage() == pytest.approx(
        -0.5 + 2 * m.phi_f + m.gamma * np.sqrt(2 * m.phi_f), rel=1e-12
    )


def test_body_effect_raises_threshold():
    m = make_mosfet()
    vth0 = m.threshold_voltage(0.0)
    vth1 = m.threshold_voltage(1.0)
    delta = m.gamma * (np.sqrt(2 * m.phi_f + 1.0) - np.sqrt(2 * m.phi_f))
    assert vth1 - vth0 == pytest.approx(delta, rel=1e-12)
    assert vth1 > vth0


def test_forward_body_bias_beyond_validity_raises():
    m = make_mosfet()
    with pytest.raises(ValueError):
        m.threshold_voltage(vsb=-(2 * m.phi_f + 0.05))


# ----------------------------------------------------------------------
# Square-law I-V (subthreshold interpolation disabled for exact algebra)
# ----------------------------------------------------------------------
def test_cutoff_zero_current():
    m = make_mosfet(subthreshold=False)
    assert float(m.current(vgs=m.threshold_voltage() - 0.5, vds=1.0)) == 0.0


def test_saturation_current_quadratic_law():
    m = make_mosfet(subthreshold=False)
    vth = m.threshold_voltage()
    k = m.mobility * m.cox * m.width / m.length
    i1 = float(m.current(vgs=vth + 1.0, vds=2.0))
    assert i1 == pytest.approx(0.5 * k, rel=1e-12)
    i2 = float(m.current(vgs=vth + 2.0, vds=3.0))
    assert i2 / i1 == pytest.approx(4.0, rel=1e-12)


def test_triode_current():
    m = make_mosfet(subthreshold=False)
    vth = m.threshold_voltage()
    k = m.mobility * m.cox * m.width / m.length
    i = float(m.current(vgs=vth + 1.0, vds=0.1))
    assert i == pytest.approx(k * (1.0 - 0.05) * 0.1, rel=1e-12)


def test_continuity_at_saturation_boundary():
    m = make_mosfet(subthreshold=False, lambda_clm=0.1)
    vth = m.threshold_voltage()
    vov = 1.0
    below = float(m.current(vgs=vth + vov, vds=vov - 1e-9))
    above = float(m.current(vgs=vth + vov, vds=vov + 1e-9))
    assert below == pytest.approx(above, rel=1e-6)


def test_channel_length_modulation():
    flat = make_mosfet(subthreshold=False, lambda_clm=0.0)
    clm = make_mosfet(subthreshold=False, lambda_clm=0.1)
    vgs = flat.threshold_voltage() + 1.0
    ratio = float(clm.current(vgs=vgs, vds=3.0) / flat.current(vgs=vgs, vds=3.0))
    assert ratio == pytest.approx(1.3, rel=1e-12)


def test_output_current_monotonic_in_vds():
    m = make_mosfet(lambda_clm=0.05)
    vds = np.linspace(0.0, 3.0, 100)
    i = m.current(vgs=1.5, vds=vds)
    assert np.all(np.diff(i) >= 0)


def test_current_increases_with_width_decreases_with_length():
    base = make_mosfet(subthreshold=False)
    wide = make_mosfet(subthreshold=False, width=2e-3)
    long_ = make_mosfet(subthreshold=False, length=2e-4)
    vgs = base.threshold_voltage() + 1.0
    assert float(wide.current(vgs=vgs, vds=2.0)) == pytest.approx(
        2 * float(base.current(vgs=vgs, vds=2.0)), rel=1e-12
    )
    assert float(long_.current(vgs=vgs, vds=2.0)) == pytest.approx(
        0.5 * float(base.current(vgs=vgs, vds=2.0)), rel=1e-12
    )


# ----------------------------------------------------------------------
# Subthreshold behavior
# ----------------------------------------------------------------------
def test_subthreshold_slope_matches_body_factor():
    m = make_mosfet(subthreshold=True)
    vth = m.threshold_voltage()
    v1, dv = vth - 0.3, 0.06
    ratio = float(m.current(vgs=v1 + dv, vds=1.0) / m.current(vgs=v1, vds=1.0))
    expected = np.exp(dv / (m.body_factor() * m.vt))
    assert ratio == pytest.approx(expected, rel=5e-2)


def test_subthreshold_swing_reasonable_at_300k():
    m = make_mosfet()
    ss = m.subthreshold_swing()
    assert 0.060 < ss < 0.120  # V/decade; >= ideal 60 mV/dec


def test_softplus_converges_to_square_law_in_strong_inversion():
    smooth = make_mosfet(subthreshold=True)
    hard = make_mosfet(subthreshold=False)
    vgs = smooth.threshold_voltage() + 2.0
    i_smooth = float(smooth.current(vgs=vgs, vds=3.0))
    i_hard = float(hard.current(vgs=vgs, vds=3.0))
    assert i_smooth == pytest.approx(i_hard, rel=1e-3)


def test_subthreshold_current_positive_and_small():
    m = make_mosfet()
    vth = m.threshold_voltage()
    i_sub = float(m.current(vgs=vth - 0.4, vds=1.0))
    i_on = float(m.current(vgs=vth + 1.0, vds=1.0))
    assert 0.0 < i_sub < 1e-4 * i_on


# ----------------------------------------------------------------------
# Broadcasting and validation
# ----------------------------------------------------------------------
def test_broadcasting():
    m = make_mosfet()
    vds = np.linspace(0.0, 3.0, 7)
    assert m.current(vgs=1.5, vds=vds).shape == (7,)
    vgs = np.array([[1.0], [1.5], [2.0]])
    assert m.current(vgs=vgs, vds=vds).shape == (3, 7)


def test_negative_vds_raises():
    with pytest.raises(ValueError):
        make_mosfet().current(vgs=1.0, vds=-0.1)


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        make_mosfet(na_substrate=-1e17)
    with pytest.raises(ValueError):
        make_mosfet(tox=0.0)
    with pytest.raises(ValueError):
        make_mosfet(width=-1e-3)
    with pytest.raises(ValueError):
        make_mosfet(lambda_clm=-0.1)
    with pytest.raises(ValueError):
        make_mosfet(mobility=-100.0)


def test_degenerate_substrate_warns():
    with pytest.warns(UserWarning):
        make_mosfet(na_substrate=5e19)


def test_device_interface():
    m = make_mosfet()
    assert m.terminals == ("gate", "drain", "source", "body")
    assert "threshold_voltage [V]" in m.parameters
    assert float(m.iv(vgs=1.5, vds=1.0)) == pytest.approx(
        float(m.current(vgs=1.5, vds=1.0)), rel=1e-12
    )
