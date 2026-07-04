"""Long-channel NMOS transistor (square-law model).

Physics implemented (see ``docs/theory.md``):

- Threshold voltage from flat-band voltage, Fermi potential, and the
  body-effect coefficient: ``Vth = Vfb + 2 phi_F + gamma sqrt(2 phi_F + Vsb)``.
- Square-law drain current in triode and saturation with optional
  channel-length modulation ``(1 + lambda Vds)`` applied in both regions
  (keeps the I-V continuous at the triode/saturation boundary).
- Optional subthreshold conduction via an EKV-style softplus interpolation
  of the overdrive voltage, giving the standard exponential subthreshold
  slope ``SS = m VT ln(10)`` and a smooth handoff at threshold.

Not modeled: PMOS (v1 is NMOS-only), velocity saturation, mobility
degradation, DIBL, short-channel Vth roll-off, or any self-consistent
charge model (this is not BSIM/EKV proper).
"""

from typing import Any

import numpy as np

from ..constants import EPS_0, EPS_OX_R, Q, thermal_voltage
from ..materials import SILICON, Material, check_doping
from .base import Device


class MOSFET(Device):
    """Long-channel enhancement-mode NMOS transistor.

    Parameters
    ----------
    na_substrate : float
        p-type substrate (body) doping [cm^-3].
    tox : float
        Gate-oxide thickness [cm] (10 nm = 1e-6 cm).
    width : float
        Channel width W [cm].
    length : float
        Channel length L [cm].
    material : Material, optional
        Substrate material (default: silicon).
    temperature : float, optional
        Operating temperature [K] (default 300).
    mobility : float or None, optional
        Effective channel electron mobility [cm^2/(V.s)]. Defaults to the
        bulk value ``material.mobility_n(T)``; real surface mobility is
        typically ~2x lower, so override for quantitative work.
    lambda_clm : float, optional
        Channel-length modulation parameter [1/V] (default 0).
    vfb : float or None, optional
        Flat-band voltage [V]. Default ``None`` computes the n+ polysilicon
        gate value ``Vfb = -(Eg/2 + phi_F)``, neglecting oxide charge.
    subthreshold : bool, optional
        Enable the softplus subthreshold interpolation (default True).
        If False, the current is zero below threshold.

    Examples
    --------
    >>> m = MOSFET(na_substrate=1e17, tox=1e-6, width=1e-3, length=1e-4)
    >>> 0.0 < m.threshold_voltage() < 1.0
    True
    """

    terminals = ("gate", "drain", "source", "body")

    def __init__(
        self,
        na_substrate: float,
        tox: float,
        width: float,
        length: float,
        material: Material = SILICON,
        temperature: float = 300.0,
        mobility: float | None = None,
        lambda_clm: float = 0.0,
        vfb: float | None = None,
        subthreshold: bool = True,
    ) -> None:
        check_doping(na_substrate, "na_substrate (body doping)")
        if tox <= 0:
            raise ValueError(f"tox must be positive, got {tox} cm")
        if width <= 0 or length <= 0:
            raise ValueError("width and length must be positive")
        if lambda_clm < 0:
            raise ValueError(f"lambda_clm must be >= 0, got {lambda_clm}")

        self.na_substrate = na_substrate
        self.tox = tox
        self.width = width
        self.length = length
        self.material = material
        self.temperature = temperature
        self.lambda_clm = lambda_clm
        self.subthreshold = subthreshold

        self.vt = thermal_voltage(temperature)  # also validates temperature
        self.mobility = (
            material.mobility_n(temperature) if mobility is None else float(mobility)
        )
        if self.mobility <= 0:
            raise ValueError("mobility must be positive")

        self._vfb_user = vfb

    # ------------------------------------------------------------------
    # Derived electrostatic quantities
    # ------------------------------------------------------------------
    @property
    def cox(self) -> float:
        """Oxide capacitance per unit area ``Cox = eps_ox / tox`` [F/cm^2]."""
        return EPS_OX_R * EPS_0 / self.tox

    @property
    def phi_f(self) -> float:
        """Bulk Fermi potential ``phi_F = VT ln(Na / ni)`` [V]."""
        return self.vt * np.log(self.na_substrate / self.material.ni(self.temperature))

    @property
    def gamma(self) -> float:
        """Body-effect coefficient ``gamma = sqrt(2 q eps_s Na) / Cox`` [V^0.5]."""
        eps_s = self.material.permittivity
        return float(np.sqrt(2.0 * Q * eps_s * self.na_substrate) / self.cox)

    @property
    def flat_band_voltage(self) -> float:
        """Flat-band voltage [V].

        User-supplied value if given, otherwise the ideal n+ poly gate on
        p-substrate work-function difference ``-(Eg/2 + phi_F)`` with zero
        oxide charge.
        """
        if self._vfb_user is not None:
            return self._vfb_user
        eg = self.material.bandgap(self.temperature)
        return -(eg / 2.0 + self.phi_f)

    def threshold_voltage(self, vsb: float = 0.0) -> float:
        """Threshold voltage including body effect.

        ``Vth = Vfb + 2 phi_F + gamma sqrt(2 phi_F + Vsb)``

        Parameters
        ----------
        vsb : float, optional
            Source-body reverse bias [V] (>= 0 for an NMOS in normal
            operation).

        Returns
        -------
        float
            Threshold voltage [V].

        Raises
        ------
        ValueError
            If ``2 phi_F + vsb`` is not positive (forward body bias beyond
            the model's validity).
        """
        arg = 2.0 * self.phi_f + vsb
        if arg <= 0:
            raise ValueError(f"2*phi_F + Vsb must be positive, got {arg:.3f} V")
        return self.flat_band_voltage + 2.0 * self.phi_f + self.gamma * np.sqrt(arg)

    def body_factor(self, vsb: float = 0.0) -> float:
        """Subthreshold slope factor ``m = 1 + gamma / (2 sqrt(2 phi_F + Vsb))``."""
        return 1.0 + self.gamma / (2.0 * np.sqrt(2.0 * self.phi_f + vsb))

    def subthreshold_swing(self, vsb: float = 0.0) -> float:
        """Subthreshold swing ``SS = m VT ln(10)`` [V/decade]."""
        return self.body_factor(vsb) * self.vt * np.log(10.0)

    # ------------------------------------------------------------------
    # Drain current
    # ------------------------------------------------------------------
    def _overdrive(self, vgs: np.ndarray, vsb: float) -> np.ndarray:
        """Effective overdrive voltage.

        Above threshold: ``Vgs - Vth``. With ``subthreshold=True`` an
        EKV-style softplus interpolation is used::

            Vov = 2 m VT ln(1 + exp((Vgs - Vth) / (2 m VT)))

        which tends to ``Vgs - Vth`` in strong inversion and yields
        ``Id ~ exp((Vgs - Vth) / (m VT))`` below threshold.
        """
        vth = self.threshold_voltage(vsb)
        if not self.subthreshold:
            return np.maximum(vgs - vth, 0.0)
        m2vt = 2.0 * self.body_factor(vsb) * self.vt
        u = (vgs - vth) / m2vt
        # Overflow-safe softplus: log(1 + e^u) = max(u, 0) + log1p(e^-|u|)
        return m2vt * (np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u))))

    def current(
        self,
        vgs: float | np.ndarray,
        vds: float | np.ndarray,
        vsb: float = 0.0,
    ) -> np.ndarray:
        """Drain current of the square-law model.

        Triode (``Vds < Vov``)::

            Id = k (Vov - Vds/2) Vds (1 + lambda Vds)

        Saturation (``Vds >= Vov``)::

            Id = (k/2) Vov^2 (1 + lambda Vds)

        with ``k = mu Cox W/L`` and ``Vov`` the (softplus-interpolated)
        overdrive from :meth:`_overdrive`. The CLM factor is applied in
        both regions so the characteristic is continuous at ``Vds = Vov``.

        Parameters
        ----------
        vgs : float or ndarray
            Gate-source voltage [V].
        vds : float or ndarray
            Drain-source voltage [V]; must be >= 0 (NMOS, normal mode).
        vsb : float, optional
            Source-body voltage [V] (scalar).

        Returns
        -------
        ndarray
            Drain current [A], broadcast over ``vgs`` and ``vds``.
        """
        vgs = np.asarray(vgs, dtype=float)
        vds = np.asarray(vds, dtype=float)
        if np.any(vds < 0):
            raise ValueError("This NMOS model requires Vds >= 0 (source/drain not swapped)")

        vov = self._overdrive(vgs, vsb)
        k = self.mobility * self.cox * self.width / self.length
        id_triode = k * (vov - vds / 2.0) * vds
        id_sat = 0.5 * k * vov**2
        i_d = np.where(vds < vov, id_triode, id_sat)
        return i_d * (1.0 + self.lambda_clm * vds)

    def iv(
        self,
        vgs: float | np.ndarray,
        vds: float | np.ndarray,
        vsb: float = 0.0,
    ) -> np.ndarray:
        """Alias for :meth:`current` (generic device interface)."""
        return self.current(vgs, vds, vsb)

    # ------------------------------------------------------------------
    @property
    def parameters(self) -> dict[str, Any]:
        """Configuration plus key derived quantities."""
        return {
            "material": self.material.name,
            "na_substrate [cm^-3]": self.na_substrate,
            "tox [cm]": self.tox,
            "width [cm]": self.width,
            "length [cm]": self.length,
            "temperature [K]": self.temperature,
            "mobility [cm^2/Vs]": self.mobility,
            "lambda_clm [1/V]": self.lambda_clm,
            "cox [F/cm^2]": self.cox,
            "phi_f [V]": self.phi_f,
            "gamma [V^0.5]": self.gamma,
            "flat_band_voltage [V]": self.flat_band_voltage,
            "threshold_voltage [V]": self.threshold_voltage(),
            "subthreshold_swing [V/dec]": self.subthreshold_swing(),
        }
