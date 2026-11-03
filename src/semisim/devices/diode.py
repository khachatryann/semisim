"""Abrupt PN junction diode model.

Physics implemented (see ``docs/theory.md`` for derivations and references):

- Built-in potential from Boltzmann statistics.
- Depletion width, edges, and junction capacitance in the full-depletion
  (abrupt junction) approximation.
- Saturation current from the ideal-diode (Shockley) long-base
  diffusion model with configurable minority-carrier lifetimes.
- Optional Caughey-Thomas (doping-dependent) minority-carrier mobility.
- I-V from the Shockley equation with ideality factor; with a series
  resistance the implicit equation is solved exactly via the Lambert W
  function (evaluated in log space, so it does not overflow).
- Optional Sah-Noyce-Shockley recombination-generation current in the
  depletion region (the ``n = 2`` component of the two-diode model).
- Optional reverse breakdown (SPICE-style exponential knee) with a
  built-in estimate of the avalanche breakdown voltage (Sze & Gibbons).
- Analytic small-signal conductance ``dI/dV``.
- Equilibrium/biased band diagram in the depletion approximation with
  flat quasi-Fermi level splitting (illustrative, not self-consistent).

Not modeled: high-level injection, band-gap narrowing, tunneling (Zener)
physics beyond the empirical breakdown knee.
"""

from typing import Any

import numpy as np

from .._math import bisect_monotonic, lambertw_log
from ..constants import Q, thermal_voltage
from ..materials import SILICON, Material, check_doping
from .base import Device

_MOBILITY_MODELS = ("constant", "caughey-thomas")


class PNJunction(Device):
    """Abrupt PN junction diode.

    Parameters
    ----------
    na : float
        Acceptor concentration on the p side [cm^-3].
    nd : float
        Donor concentration on the n side [cm^-3].
    material : Material, optional
        Semiconductor material (default: silicon).
    area : float, optional
        Junction area [cm^2] (default 1e-4, i.e. 100 um x 100 um).
    temperature : float, optional
        Operating temperature [K] (default 300).
    ideality : float, optional
        Ideality factor ``n`` in the Shockley equation (default 1.0).
    series_resistance : float, optional
        Ohmic series resistance ``Rs`` [ohm] (default 0).
    tau_n : float, optional
        Minority electron lifetime in the p region [s] (default 1e-6).
    tau_p : float, optional
        Minority hole lifetime in the n region [s] (default 1e-6).
    mobility_model : {"constant", "caughey-thomas"}, optional
        ``"constant"`` (default) uses lightly-doped bulk mobilities;
        ``"caughey-thomas"`` evaluates each minority-carrier mobility at the
        doping of the region it diffuses through.
    tau_rg : float or None, optional
        Effective depletion-region recombination-generation lifetime [s].
        ``None`` (default) disables the recombination-generation current.
    breakdown_voltage : float, "auto", or None, optional
        Reverse breakdown voltage magnitude ``BV`` [V]. ``"auto"`` uses
        :meth:`avalanche_breakdown_voltage`. ``None`` (default) disables
        breakdown (ideal reverse characteristic).
    breakdown_current : float, optional
        Current magnitude at ``V = -BV`` [A] (default 1e-3, the knee).
    breakdown_ideality : float, optional
        Emission coefficient of the breakdown knee (default 1.0); larger
        values give a softer knee.

    Examples
    --------
    >>> d = PNJunction(na=1e17, nd=1e17)
    >>> round(d.built_in_potential, 3)
    0.859
    """

    terminals = ("anode", "cathode")

    def __init__(
        self,
        na: float,
        nd: float,
        material: Material = SILICON,
        area: float = 1e-4,
        temperature: float = 300.0,
        ideality: float = 1.0,
        series_resistance: float = 0.0,
        tau_n: float = 1e-6,
        tau_p: float = 1e-6,
        mobility_model: str = "constant",
        tau_rg: float | None = None,
        breakdown_voltage: float | str | None = None,
        breakdown_current: float = 1e-3,
        breakdown_ideality: float = 1.0,
    ) -> None:
        check_doping(na, "na (p-side acceptor doping)")
        check_doping(nd, "nd (n-side donor doping)")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area} cm^2")
        if ideality < 1.0:
            raise ValueError(f"ideality factor must be >= 1, got {ideality}")
        if series_resistance < 0:
            raise ValueError(f"series_resistance must be >= 0, got {series_resistance}")
        if tau_n <= 0 or tau_p <= 0:
            raise ValueError("minority-carrier lifetimes must be positive")
        if mobility_model not in _MOBILITY_MODELS:
            raise ValueError(
                f"mobility_model must be one of {_MOBILITY_MODELS}, got {mobility_model!r}"
            )
        if tau_rg is not None and tau_rg <= 0:
            raise ValueError(f"tau_rg must be positive or None, got {tau_rg}")
        if breakdown_current <= 0:
            raise ValueError("breakdown_current must be positive")
        if breakdown_ideality <= 0:
            raise ValueError("breakdown_ideality must be positive")

        self.na = na
        self.nd = nd
        self.material = material
        self.area = area
        self.temperature = temperature
        self.ideality = ideality
        self.series_resistance = series_resistance
        self.tau_n = tau_n
        self.tau_p = tau_p
        self.mobility_model = mobility_model
        self.tau_rg = tau_rg
        self.breakdown_current = breakdown_current
        self.breakdown_ideality = breakdown_ideality

        self.vt = thermal_voltage(temperature)  # also validates temperature
        self.ni = material.ni(temperature)

        if isinstance(breakdown_voltage, str):
            if breakdown_voltage.strip().lower() != "auto":
                raise ValueError(
                    "breakdown_voltage must be a number, 'auto', or None; "
                    f"got {breakdown_voltage!r}"
                )
            breakdown_voltage = self.avalanche_breakdown_voltage()
        if breakdown_voltage is not None and breakdown_voltage <= 0:
            raise ValueError(
                f"breakdown_voltage is a magnitude and must be positive, got {breakdown_voltage}"
            )
        self.breakdown_voltage = breakdown_voltage

    # ------------------------------------------------------------------
    # Electrostatics
    # ------------------------------------------------------------------
    @property
    def built_in_potential(self) -> float:
        """Built-in potential ``Vbi = VT ln(Na Nd / ni^2)`` [V]."""
        return self.vt * np.log(self.na * self.nd / self.ni**2)

    def depletion_width(self, bias: float = 0.0) -> float:
        """Total depletion width in the full-depletion approximation.

        ``W = sqrt(2 eps (Vbi - V) (Na + Nd) / (q Na Nd))``

        Parameters
        ----------
        bias : float, optional
            Applied voltage [V], positive = forward. Must be below the
            built-in potential (the approximation collapses at ``Vbi``).

        Returns
        -------
        float
            Depletion width [cm].

        Raises
        ------
        ValueError
            If ``bias >= built_in_potential``.
        """
        vbi = self.built_in_potential
        if bias >= vbi:
            raise ValueError(
                f"Full-depletion approximation invalid for bias {bias} V >= Vbi {vbi:.3f} V"
            )
        eps = self.material.permittivity
        return float(
            np.sqrt(2.0 * eps * (vbi - bias) * (self.na + self.nd) / (Q * self.na * self.nd))
        )

    def depletion_edges(self, bias: float = 0.0) -> tuple[float, float]:
        """Depletion extents into the p and n sides.

        Charge neutrality gives ``xp Na = xn Nd``.

        Returns
        -------
        tuple of float
            ``(xp, xn)`` [cm], both positive; junction at x = 0 with the
            p side at x < 0.
        """
        w = self.depletion_width(bias)
        xp = w * self.nd / (self.na + self.nd)
        xn = w * self.na / (self.na + self.nd)
        return xp, xn

    def junction_capacitance(self, bias: float = 0.0) -> float:
        """Depletion (junction) capacitance ``Cj = eps A / W`` [F].

        Parameters
        ----------
        bias : float, optional
            Applied voltage [V], must be below ``Vbi``.
        """
        return self.material.permittivity * self.area / self.depletion_width(bias)

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def avalanche_breakdown_voltage(self) -> float:
        """Estimated avalanche breakdown voltage of the abrupt junction [V].

        Sze & Gibbons universal approximation for one-sided abrupt junctions::

            BV ~ 60 (Eg / 1.1)^1.5 (N_B / 1e16)^-0.75   [V]

        with ``N_B`` the doping of the lightly-doped side (here the
        reduced doping ``Na Nd / (Na + Nd)``, which reduces to ``N_B`` for
        one-sided junctions). Accuracy is roughly +/-20 % for 1e14-1e18
        cm^-3; below ~5 V the real breakdown is tunneling (Zener) dominated.
        """
        eg = self.material.bandgap(self.temperature)
        n_b = self.na * self.nd / (self.na + self.nd)
        return float(60.0 * (eg / 1.1) ** 1.5 * (n_b / 1e16) ** -0.75)

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def _diffusion_coefficients(self) -> tuple[float, float]:
        """Minority-carrier diffusion coefficients ``(Dn, Dp)`` [cm^2/s]."""
        t = self.temperature
        if self.mobility_model == "caughey-thomas":
            # Electrons are minority in the p side (doping Na), holes in the n side (Nd).
            return self.material.diffusion_n(t, self.na), self.material.diffusion_p(t, self.nd)
        return self.material.diffusion_n(t), self.material.diffusion_p(t)

    @property
    def saturation_current(self) -> float:
        """Ideal-diode (long-base) saturation current [A].

        ``Is = q A ni^2 (Dn / (Ln Na) + Dp / (Lp Nd))``

        with diffusion lengths ``L = sqrt(D tau)``. With
        ``mobility_model="constant"`` the bulk lightly-doped mobilities are
        used; ``"caughey-thomas"`` evaluates them at the local doping.
        """
        dn, dp = self._diffusion_coefficients()
        ln = np.sqrt(dn * self.tau_n)
        lp = np.sqrt(dp * self.tau_p)
        return float(
            Q * self.area * self.ni**2 * (dn / (ln * self.na) + dp / (lp * self.nd))
        )

    def _rg_width(self, vj: np.ndarray) -> np.ndarray:
        """Depletion width used by the R-G term: ``W(min(Vj, 0))`` [cm]."""
        eps = self.material.permittivity
        v_eff = np.minimum(vj, 0.0)
        return np.sqrt(
            2.0 * eps * (self.built_in_potential - v_eff) * (self.na + self.nd)
            / (Q * self.na * self.nd)
        )

    def _has_extra_terms(self) -> bool:
        return self.tau_rg is not None or self.breakdown_voltage is not None

    def _junction_current(self, vj: np.ndarray) -> np.ndarray:
        """Intrinsic junction current at junction voltage ``vj`` (no Rs) [A]."""
        n_vt = self.ideality * self.vt
        with np.errstate(over="ignore"):
            i = self.saturation_current * np.expm1(vj / n_vt)
            if self.tau_rg is not None:
                i_r0 = Q * self.area * self.ni * self._rg_width(vj) / (2.0 * self.tau_rg)
                i = i + i_r0 * np.expm1(vj / (2.0 * self.vt))
            if self.breakdown_voltage is not None:
                nbv_vt = self.breakdown_ideality * self.vt
                i = i - self.breakdown_current * np.exp(
                    -(vj + self.breakdown_voltage) / nbv_vt
                )
        return i

    def _junction_conductance(self, vj: np.ndarray) -> np.ndarray:
        """Analytic ``dI/dVj`` of :meth:`_junction_current` [S]."""
        n_vt = self.ideality * self.vt
        with np.errstate(over="ignore"):
            g = self.saturation_current / n_vt * np.exp(vj / n_vt)
            if self.tau_rg is not None:
                w = self._rg_width(vj)
                i_r0 = Q * self.area * self.ni * w / (2.0 * self.tau_rg)
                # dW/dV = -W / (2 (Vbi - V)) for V < 0, zero for V >= 0.
                dw = np.where(vj < 0, -w / (2.0 * (self.built_in_potential - vj)), 0.0)
                di_r0 = i_r0 * dw / w
                g = g + i_r0 / (2.0 * self.vt) * np.exp(vj / (2.0 * self.vt))
                g = g + di_r0 * np.expm1(vj / (2.0 * self.vt))
            if self.breakdown_voltage is not None:
                nbv_vt = self.breakdown_ideality * self.vt
                g = g + self.breakdown_current / nbv_vt * np.exp(
                    -(vj + self.breakdown_voltage) / nbv_vt
                )
        return g

    def junction_voltage(self, voltage: float | np.ndarray) -> np.ndarray:
        """Voltage across the intrinsic junction, ``Vj = V - I Rs`` [V]."""
        v = np.asarray(voltage, dtype=float)
        rs = self.series_resistance
        if rs == 0.0:
            return v
        if not self._has_extra_terms():
            return v - self.current(v) * rs
        # V = Vj + Rs I(Vj) is strictly increasing in Vj; the root lies
        # between 0 and V.
        return bisect_monotonic(
            lambda vj: vj + rs * self._junction_current(vj),
            v,
            np.minimum(v, 0.0),
            np.maximum(v, 0.0),
        )

    def current(self, voltage: float | np.ndarray) -> np.ndarray:
        """Diode current.

        Junction current (each optional term only when enabled)::

            I(Vj) = Is (exp(Vj/nVT) - 1)                       # diffusion
                  + Ir0(Vj) (exp(Vj/2VT) - 1)                  # R-G
                  - Ibv exp(-(Vj + BV)/(nbv VT))               # breakdown

        with ``Ir0 = q A ni W / (2 tau_rg)``. With series resistance ``Rs``
        the terminal voltage is ``V = Vj + I Rs``. For the pure Shockley
        case this is solved exactly with the Lambert W function::

            I = (n VT / Rs) W0( (Is Rs / (n VT)) exp((V + Is Rs)/(n VT)) ) - Is

        otherwise by vectorized bisection on ``Vj`` (machine precision).

        Parameters
        ----------
        voltage : float or ndarray
            Applied anode-cathode voltage [V], positive = forward.

        Returns
        -------
        ndarray
            Diode current [A].
        """
        v = np.asarray(voltage, dtype=float)
        rs = self.series_resistance
        if rs == 0.0:
            return self._junction_current(v)
        if self._has_extra_terms():
            return self._junction_current(self.junction_voltage(v))
        n_vt = self.ideality * self.vt
        i_s = self.saturation_current
        # Argument of W in log space: log(Is Rs / nVT) + (V + Is Rs) / nVT
        log_arg = np.log(i_s * rs / n_vt) + (v + i_s * rs) / n_vt
        return n_vt / rs * lambertw_log(log_arg) - i_s

    def conductance(self, voltage: float | np.ndarray) -> np.ndarray:
        """Small-signal conductance ``dI/dV`` [S].

        ``g = gj / (1 + Rs gj)`` with ``gj = dI/dVj`` evaluated analytically
        at the junction voltage. Its inverse is the dynamic resistance.

        Parameters
        ----------
        voltage : float or ndarray
            Applied anode-cathode voltage [V].
        """
        gj = self._junction_conductance(self.junction_voltage(voltage))
        rs = self.series_resistance
        if rs == 0.0:
            return gj
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(np.isinf(gj), 1.0 / rs, gj / (1.0 + rs * gj))

    # Alias satisfying the generic Device interface.
    def iv(self, voltage: float | np.ndarray) -> np.ndarray:
        """Alias for :meth:`current` (generic device interface)."""
        return self.current(voltage)

    # ------------------------------------------------------------------
    # Band diagram
    # ------------------------------------------------------------------
    def band_diagram(self, bias: float = 0.0, points: int = 400) -> dict[str, np.ndarray]:
        """Band-edge energies vs. position in the depletion approximation.

        The p-side Fermi level is the energy reference (0 eV). Under bias
        the quasi-Fermi levels are drawn as flat lines split by the applied
        voltage — an illustration, not a solved drift-diffusion profile.

        Parameters
        ----------
        bias : float, optional
            Applied voltage [V], must be below ``Vbi``.
        points : int, optional
            Number of position samples.

        Returns
        -------
        dict of ndarray
            Keys ``x`` [cm], ``ec``, ``ev``, ``efp``, ``efn`` [eV].
            ``efn``/``efp`` are NaN outside their meaningful regions.
        """
        vbi_eff = self.built_in_potential - bias
        xp, xn = self.depletion_edges(bias)
        eps = self.material.permittivity
        kt = self.vt  # eV-per-unit-charge equals volts numerically

        x = np.linspace(-3.0 * xp, 3.0 * xn, points)
        phi = np.zeros_like(x)
        in_p_dep = (x >= -xp) & (x < 0)
        in_n_dep = (x >= 0) & (x <= xn)
        phi[in_p_dep] = Q * self.na * (x[in_p_dep] + xp) ** 2 / (2.0 * eps)
        phi[in_n_dep] = vbi_eff - Q * self.nd * (xn - x[in_n_dep]) ** 2 / (2.0 * eps)
        phi[x > xn] = vbi_eff

        eg = self.material.bandgap(self.temperature)
        # Neutral p side: Ef - Ev = kT ln(Nv / Na), with Ef = 0.
        ev_p = -kt * np.log(self.material.nv(self.temperature) / self.na)
        ec = ev_p + eg - phi
        ev = ec - eg

        efp = np.where(x <= xn, 0.0, np.nan)
        efn = np.where(x >= -xp, bias, np.nan)
        return {"x": x, "ec": ec, "ev": ev, "efp": efp, "efn": efn}

    # ------------------------------------------------------------------
    @property
    def parameters(self) -> dict[str, Any]:
        """Configuration plus key derived quantities."""
        return {
            "material": self.material.name,
            "na [cm^-3]": self.na,
            "nd [cm^-3]": self.nd,
            "area [cm^2]": self.area,
            "temperature [K]": self.temperature,
            "ideality": self.ideality,
            "series_resistance [ohm]": self.series_resistance,
            "built_in_potential [V]": self.built_in_potential,
            "depletion_width [cm]": self.depletion_width(),
            "saturation_current [A]": self.saturation_current,
            "mobility_model": self.mobility_model,
            "tau_rg [s]": self.tau_rg,
            "breakdown_voltage [V]": self.breakdown_voltage,
        }
