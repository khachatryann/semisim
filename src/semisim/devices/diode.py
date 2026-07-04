"""Abrupt PN junction diode model.

Physics implemented (see ``docs/theory.md`` for derivations and references):

- Built-in potential from Boltzmann statistics.
- Depletion width, edges, and junction capacitance in the full-depletion
  (abrupt junction) approximation.
- Saturation current from the ideal-diode (Shockley) long-base
  diffusion model with configurable minority-carrier lifetimes.
- I-V from the Shockley equation with ideality factor; with a series
  resistance the implicit equation is solved exactly via the Lambert W
  function (evaluated in log space, so it does not overflow).
- Equilibrium/biased band diagram in the depletion approximation with
  flat quasi-Fermi level splitting (illustrative, not self-consistent).

Not modeled: avalanche/Zener breakdown, recombination-generation
(two-diode) current, high-level injection, band-gap narrowing.
"""

from typing import Any

import numpy as np

from .._math import lambertw_log
from ..constants import Q, thermal_voltage
from ..materials import SILICON, Material, check_doping
from .base import Device


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

        self.na = na
        self.nd = nd
        self.material = material
        self.area = area
        self.temperature = temperature
        self.ideality = ideality
        self.series_resistance = series_resistance
        self.tau_n = tau_n
        self.tau_p = tau_p

        self.vt = thermal_voltage(temperature)  # also validates temperature
        self.ni = material.ni(temperature)

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
    @property
    def saturation_current(self) -> float:
        """Ideal-diode (long-base) saturation current [A].

        ``Is = q A ni^2 (Dn / (Ln Na) + Dp / (Lp Nd))``

        with diffusion lengths ``L = sqrt(D tau)``. Minority-carrier
        mobilities are approximated by the majority-carrier values of the
        bulk material (doping dependence of mobility is not modeled).
        """
        dn = self.material.diffusion_n(self.temperature)
        dp = self.material.diffusion_p(self.temperature)
        ln = np.sqrt(dn * self.tau_n)
        lp = np.sqrt(dp * self.tau_p)
        return float(
            Q * self.area * self.ni**2 * (dn / (ln * self.na) + dp / (lp * self.nd))
        )

    def current(self, voltage: float | np.ndarray) -> np.ndarray:
        """Diode current from the Shockley equation.

        Without series resistance:

        ``I = Is (exp(V / (n VT)) - 1)``

        With series resistance ``Rs``, the implicit equation
        ``I = Is (exp((V - I Rs) / (n VT)) - 1)`` is solved exactly:

        ``I = (n VT / Rs) W0( (Is Rs / (n VT)) exp((V + Is Rs)/(n VT)) ) - Is``

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
        n_vt = self.ideality * self.vt
        i_s = self.saturation_current
        rs = self.series_resistance
        if rs == 0.0:
            with np.errstate(over="ignore"):
                return i_s * np.expm1(v / n_vt)
        # Argument of W in log space: log(Is Rs / nVT) + (V + Is Rs) / nVT
        log_arg = np.log(i_s * rs / n_vt) + (v + i_s * rs) / n_vt
        return n_vt / rs * lambertw_log(log_arg) - i_s

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
        }
