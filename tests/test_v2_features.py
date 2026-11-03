"""Tests for v2 features: doping-dependent mobility, GaAs, R-G current,
breakdown, conductance, PMOS, velocity saturation, small-signal, CLI."""

import numpy as np
import pytest

from semisim import GALLIUM_ARSENIDE, MOSFET, SILICON, PNJunction, device_from_dict
from semisim._math import bisect_monotonic
from semisim.cli import main as cli_main
from semisim.constants import Q

MOS_KW = dict(body_doping=1e17, tox=1e-6, width=1e-3, length=1e-4)


# ----------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------
def test_caughey_thomas_limits_and_textbook_value():
    assert SILICON.mobility_n(300.0, 1e12) == pytest.approx(SILICON.mu_n_300, rel=1e-3)
    assert SILICON.mobility_n(300.0, 1e21) == pytest.approx(92.0, rel=0.05)
    # Sze & Ng Fig. 1.18: ~800 cm^2/Vs electrons, ~340 holes at 1e17.
    assert SILICON.mobility_n(300.0, 1e17) == pytest.approx(800.0, rel=0.05)
    assert SILICON.mobility_p(300.0, 1e17) == pytest.approx(340.0, rel=0.05)


def test_caughey_thomas_monotonic_in_doping():
    dop = np.logspace(14, 20, 25)
    mu = np.array([SILICON.mobility_n(300.0, n) for n in dop])
    assert np.all(np.diff(mu) < 0)


def test_mobility_without_doping_is_v1_value():
    assert SILICON.mobility_n(300.0) == SILICON.mu_n_300


def test_negative_doping_mobility_raises():
    with pytest.raises(ValueError):
        SILICON.mobility_p(300.0, -1.0)


def test_gaas_properties():
    assert GALLIUM_ARSENIDE.bandgap(300.0) == pytest.approx(1.424, abs=5e-3)
    assert 1e6 < GALLIUM_ARSENIDE.ni(300.0) < 1e7
    assert GALLIUM_ARSENIDE.mobility_n() > 5 * SILICON.mobility_n()


def test_gaas_from_config():
    d = device_from_dict(
        {"device": {"type": "diode", "material": "gaas", "parameters": {"na": 1e17, "nd": 1e17}}}
    )
    assert d.material is GALLIUM_ARSENIDE
    assert d.built_in_potential > 1.2


# ----------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------
def test_bisect_monotonic_cubic():
    target = np.array([-8.0, 0.0, 27.0])
    x = bisect_monotonic(lambda x: x**3, target, -10.0, 10.0)
    np.testing.assert_allclose(x, [-2.0, 0.0, 3.0], atol=1e-12)


# ----------------------------------------------------------------------
# Diode
# ----------------------------------------------------------------------
def test_caughey_thomas_lowers_saturation_current():
    const = PNJunction(na=1e18, nd=1e18)
    ct = PNJunction(na=1e18, nd=1e18, mobility_model="caughey-thomas")
    assert ct.saturation_current < const.saturation_current


def test_invalid_mobility_model_raises():
    with pytest.raises(ValueError):
        PNJunction(na=1e17, nd=1e17, mobility_model="arora")


def test_generation_current_dominates_reverse_leakage_in_si():
    d = PNJunction(na=1e17, nd=1e17, tau_rg=1e-7)
    w = d.depletion_width(-2.0)
    expected = -(Q * d.area * d.ni * w / (2 * d.tau_rg)) - d.saturation_current
    assert float(d.current(-2.0)) == pytest.approx(expected, rel=1e-6)
    assert abs(float(d.current(-2.0))) > 100 * d.saturation_current


def test_recombination_gives_ideality_two_at_low_bias():
    d = PNJunction(na=1e17, nd=1e17, tau_rg=1e-9)
    v1, v2 = 0.15, 0.20
    slope = (v2 - v1) / np.log10(float(d.current(v2) / d.current(v1)))
    assert slope == pytest.approx(2 * d.vt * np.log(10), rel=0.1)  # ~120 mV/dec


def test_rg_current_zero_at_zero_bias():
    d = PNJunction(na=1e17, nd=1e17, tau_rg=1e-7, series_resistance=3.0)
    assert float(d.current(0.0)) == pytest.approx(0.0, abs=1e-25)


def test_avalanche_breakdown_estimate_textbook():
    # One-sided Si junction, N_B = 1e16 -> ~60 V (Sze & Gibbons).
    d = PNJunction(na=1e19, nd=1e16)
    assert d.avalanche_breakdown_voltage() == pytest.approx(57.0, rel=0.1)
    # Heavier doping breaks down earlier.
    assert PNJunction(na=1e18, nd=1e17).avalanche_breakdown_voltage() < 15.0


def test_breakdown_knee_current():
    d = PNJunction(na=1e17, nd=1e17, breakdown_voltage=5.0, breakdown_current=1e-3)
    assert float(d.current(-5.0)) == pytest.approx(-1e-3 - d.saturation_current, rel=1e-6)
    assert abs(float(d.current(-4.0))) < 1e-12
    assert abs(float(d.current(-5.3))) > 0.1


def test_breakdown_auto_and_validation():
    d = PNJunction(na=1e17, nd=1e17, breakdown_voltage="auto")
    assert d.breakdown_voltage == pytest.approx(d.avalanche_breakdown_voltage())
    with pytest.raises(ValueError):
        PNJunction(na=1e17, nd=1e17, breakdown_voltage=-5.0)
    with pytest.raises(ValueError):
        PNJunction(na=1e17, nd=1e17, breakdown_voltage="soon")
    with pytest.raises(ValueError):
        PNJunction(na=1e17, nd=1e17, tau_rg=0.0)


@pytest.mark.parametrize("v", [-6.0, -5.0, -1.0, 0.0, 0.4, 0.7, 1.0, 3.0])
def test_numeric_solution_satisfies_implicit_equation(v):
    d = PNJunction(
        na=1e17, nd=1e17, tau_rg=1e-8, breakdown_voltage=5.0, series_resistance=10.0
    )
    i = float(d.current(v))
    vj = v - i * d.series_resistance
    assert float(d._junction_current(vj)) == pytest.approx(i, rel=1e-9, abs=1e-24)


def test_no_overflow_with_extra_terms_at_extreme_bias():
    d = PNJunction(na=1e17, nd=1e17, tau_rg=1e-7, breakdown_voltage=5.0, series_resistance=1.0)
    i = d.current(np.array([-200.0, 200.0]))
    assert np.all(np.isfinite(i))
    assert i[1] == pytest.approx(199.0, rel=0.01)  # ~ (V - Vj)/Rs


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(series_resistance=5.0),
        dict(tau_rg=1e-8, breakdown_voltage=6.0),
        dict(tau_rg=1e-8, breakdown_voltage=6.0, series_resistance=5.0),
    ],
)
def test_conductance_matches_numerical_derivative(kw):
    d = PNJunction(na=1e17, nd=1e16, **kw)
    v = np.array([-6.2, -2.0, 0.2, 0.5, 0.7])
    h = 1e-6
    numeric = (d.current(v + h) - d.current(v - h)) / (2 * h)
    np.testing.assert_allclose(d.conductance(v), numeric, rtol=1e-4, atol=1e-18)


def test_conductance_limits():
    d = PNJunction(na=1e17, nd=1e17)
    assert float(d.conductance(0.0)) == pytest.approx(d.saturation_current / d.vt, rel=1e-12)
    d_rs = PNJunction(na=1e17, nd=1e17, series_resistance=4.0)
    assert float(d_rs.conductance(5.0)) == pytest.approx(0.25, rel=1e-2)


# ----------------------------------------------------------------------
# MOSFET: PMOS
# ----------------------------------------------------------------------
def test_pmos_is_mirror_of_nmos_with_matched_mobility():
    n = MOSFET(**MOS_KW, mobility=300.0)
    p = MOSFET(**MOS_KW, mobility=300.0, channel="p")
    assert p.flat_band_voltage == pytest.approx(-n.flat_band_voltage)
    assert p.threshold_voltage() == pytest.approx(-n.threshold_voltage())
    assert p.threshold_voltage(vsb=-1.0) == pytest.approx(-n.threshold_voltage(vsb=1.0))
    vgs = np.linspace(0, 3, 13)
    np.testing.assert_allclose(
        p.current(vgs=-vgs, vds=-1.5), -n.current(vgs=vgs, vds=1.5), rtol=1e-12
    )


def test_pmos_defaults_to_hole_mobility():
    p = MOSFET(**MOS_KW, channel="pmos")
    assert p.channel == "p"
    assert p.mobility == pytest.approx(SILICON.mobility_p(300.0))
    assert float(p.current(vgs=-2.0, vds=-2.0)) < 0


def test_pmos_wrong_vds_sign_raises():
    p = MOSFET(**MOS_KW, channel="p")
    with pytest.raises(ValueError):
        p.current(vgs=-2.0, vds=0.5)


def test_invalid_channel_raises():
    with pytest.raises(ValueError):
        MOSFET(**MOS_KW, channel="x")


def test_pmos_from_config():
    m = device_from_dict({"device": {"type": "pmos", "parameters": dict(MOS_KW)}})
    assert m.channel == "p"
    with pytest.raises(ValueError):
        device_from_dict({"device": {"type": "pmos", "parameters": dict(MOS_KW, channel="n")}})


def test_na_substrate_deprecated_alias():
    with pytest.warns(DeprecationWarning):
        m = MOSFET(na_substrate=1e17, tox=1e-6, width=1e-3, length=1e-4)
    assert m.body_doping == 1e17
    assert m.na_substrate == 1e17
    with pytest.raises(TypeError):
        MOSFET(body_doping=1e17, na_substrate=1e17, tox=1e-6, width=1e-3, length=1e-4)


def test_missing_required_argument_raises():
    with pytest.raises(TypeError):
        MOSFET(body_doping=1e17, tox=1e-6)


def test_caughey_thomas_channel_mobility():
    m = MOSFET(**MOS_KW, mobility_model="caughey-thomas")
    assert m.mobility == pytest.approx(SILICON.mobility_n(300.0, 1e17))


# ----------------------------------------------------------------------
# MOSFET: velocity saturation
# ----------------------------------------------------------------------
def test_velocity_saturation_reduces_current_and_vdsat():
    ideal = MOSFET(**MOS_KW, subthreshold=False)
    vs = MOSFET(**MOS_KW, subthreshold=False, vsat=1e7)
    vgs = ideal.threshold_voltage() + 1.5
    assert float(vs.current(vgs=vgs, vds=3.0)) < float(ideal.current(vgs=vgs, vds=3.0))
    assert float(vs.saturation_voltage(vgs)) < float(ideal.saturation_voltage(vgs))
    assert float(ideal.saturation_voltage(vgs)) == pytest.approx(1.5, rel=1e-12)


def test_velocity_saturation_limit_recovers_square_law():
    ideal = MOSFET(**MOS_KW, subthreshold=False)
    fast = MOSFET(**MOS_KW, subthreshold=False, vsat=1e14)
    vgs = ideal.threshold_voltage() + 1.0
    vds = np.linspace(0, 3, 31)
    np.testing.assert_allclose(fast.current(vgs=vgs, vds=vds), ideal.current(vgs=vgs, vds=vds),
                               rtol=1e-5)


def test_velocity_saturation_continuity_and_short_channel_limit():
    m = MOSFET(body_doping=1e17, tox=1e-6, width=1e-3, length=1e-5, subthreshold=False,
               vsat=1e7, lambda_clm=0.05)
    vgs = m.threshold_voltage() + 1.0
    vdsat = float(m.saturation_voltage(vgs))
    below = float(m.current(vgs=vgs, vds=vdsat - 1e-9))
    above = float(m.current(vgs=vgs, vds=vdsat + 1e-9))
    assert below == pytest.approx(above, rel=1e-6)
    # Strong velocity saturation: Id -> W Cox vsat (Vgs - Vth) as Ec L -> 0.
    long_vov = m.threshold_voltage() + 20.0
    i = float(m.current(vgs=long_vov, vds=25.0)) / (1 + 0.05 * 25.0)
    assert i == pytest.approx(m.width * m.cox * m.vsat * 20.0, rel=0.1)


def test_vsat_validation():
    with pytest.raises(ValueError):
        MOSFET(**MOS_KW, vsat=0.0)


# ----------------------------------------------------------------------
# MOSFET: small signal
# ----------------------------------------------------------------------
def test_small_signal_square_law_saturation():
    m = MOSFET(**MOS_KW, subthreshold=False, lambda_clm=0.05)
    vov, vds = 1.0, 2.0
    vgs = m.threshold_voltage() + vov
    ss = m.small_signal(vgs=vgs, vds=vds)
    k = m.mobility * m.cox * m.width / m.length
    assert float(ss["gm"]) == pytest.approx(k * vov * (1 + 0.05 * vds), rel=1e-6)
    assert float(ss["gds"]) == pytest.approx(0.5 * k * vov**2 * 0.05, rel=1e-6)
    assert float(ss["ro"]) == pytest.approx(1 / float(ss["gds"]))
    # gmb = gm * gamma / (2 sqrt(2 phi_F + Vsb))
    eta = m.gamma / (2 * np.sqrt(2 * m.phi_f))
    assert float(ss["gmb"]) == pytest.approx(eta * float(ss["gm"]), rel=1e-4)


def test_small_signal_pmos_positive():
    p = MOSFET(**MOS_KW, channel="p", lambda_clm=0.05)
    ss = p.small_signal(vgs=-2.0, vds=-2.0)
    assert float(ss["gm"]) > 0 and float(ss["gds"]) > 0 and float(ss["gmb"]) > 0


def test_small_signal_subthreshold_gm_over_id():
    m = MOSFET(**MOS_KW)
    vgs = m.threshold_voltage() - 0.3
    ss = m.small_signal(vgs=vgs, vds=1.0)
    gm_id = float(ss["gm"]) / float(m.current(vgs=vgs, vds=1.0))
    assert gm_id == pytest.approx(1 / (m.body_factor() * m.vt), rel=0.05)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_cli_info(tmp_path, capsys):
    cfg = tmp_path / "d.toml"
    cfg.write_text('[device]\ntype = "pmos"\n[device.parameters]\n'
                   "body_doping = 1e17\ntox = 1e-6\nwidth = 1e-3\nlength = 1e-4\n")
    assert cli_main(["info", str(cfg)]) == 0
    assert "PMOS" in capsys.readouterr().out


def test_cli_plot(tmp_path):
    cfg = tmp_path / "d.toml"
    cfg.write_text('[device]\ntype = "diode"\n[device.parameters]\nna = 1e17\nnd = 1e17\n'
                   'breakdown_voltage = "auto"\n')
    out = tmp_path / "out.png"
    assert cli_main(["plot", str(cfg), "--log", "-o", str(out)]) == 0
    assert out.stat().st_size > 1000


def test_cli_error_exit_code(tmp_path, capsys):
    assert cli_main(["info", str(tmp_path / "missing.toml")]) == 2
    assert "error" in capsys.readouterr().err
