"""Long-channel MOSFET (NMOS and PMOS, square-law model).

Physics implemented (see ``docs/theory.md``):

- Threshold voltage from flat-band voltage, Fermi potential, and the
  body-effect coefficient: ``Vth = Vfb + 2 phi_F + gamma sqrt(2 phi_F + Vsb)``
  (NMOS; the PMOS expression is the mirror image).
- Square-law drain current in triode and saturation with optional
  channel-length modulation ``(1 + lambda Vds)`` applied in both regions
  (keeps the I-V continuous at the triode/saturation boundary).
- Optional velocity saturation (lateral-field mobility degradation with
  critical field ``Ec = 2 vsat / mu``), which reduces ``Vdsat`` and turns
  the saturation current from quadratic towards linear in ``Vov``.
- Optional subthreshold conduction via an EKV-style softplus interpolation
  of the overdrive voltage, giving the standard exponential subthreshold
  slope ``SS = m VT ln(10)`` and a smooth handoff at threshold.
- Optional Caughey-Thomas channel mobility evaluated at the body doping.
- Small-signal ``gm``, ``gds`` and ``gmb``.

Not modeled: vertical-field mobility degradation, DIBL, short-channel Vth
roll-off, or any self-consistent charge model (this is not BSIM/EKV proper).
"""

import warnings
from typing import Any

import numpy as np

from ..constants import EPS_0, EPS_OX_R, Q, thermal_voltage
from ..materials import SILICON, Material, check_doping
from .base import Device

_MOBILITY_MODELS = ("constant", "caughey-thomas")


class MOSFET(Device):
    """Long-channel enhancement-mode MOSFET (NMOS or PMOS).

    All bias arguments use the standard signed convention: for a PMOS,
    ``vgs``, ``vds`` are normally negative and the returned drain current
    (positive *into* the drain terminal) is negative.

    Parameters
    ----------
    body_doping : float
        Body (substrate/well) doping magnitude [cm^-3]: acceptors for NMOS,
        donors for PMOS.
    tox : float
        Gate-oxide thickness [cm] (10 nm = 1e-6 cm).
    width : float
        Channel width W [cm].
    length : float
        Channel length L [cm].
    channel : {"n", "p"}, optional
        Channel type (default ``"n"``).
    material : Material, optional
        Substrate material (default: silicon).
    temperature : float, optional
        Operating temperature [K] (default 300).
    mobility : float or None, optional
        Effective channel carrier mobility [cm^2/(V.s)]. Defaults to the
        bulk value of the channel carrier from ``mobility_model``; real
        surface mobility is typically ~2x lower, so override for
        quantitative work.
    mobility_model : {"constant", "caughey-thomas"}, optional
        Default-mobility model: lightly-doped bulk value (default) or
        Caughey-Thomas evaluated at ``body_doping``.
    lambda_clm : float, optional
        Channel-length modulation parameter [1/V] (default 0).
    vfb : float or None, optional
        Flat-band voltage [V]. Default ``None`` computes the ideal
        same-type polysilicon gate value (n+ poly for NMOS, p+ poly for
        PMOS), ``Vfb = -/+(Eg/2 + phi_F)``, neglecting oxide charge.
    vsat : float or None, optional
        Carrier saturation velocity [cm/s] (Si electrons ~1e7). ``None``
        (default) disables velocity saturation.
    subthreshold : bool, optional
        Enable the softplus subthreshold interpolation (default True).
        If False, the current is zero below threshold.
    na_substrate : float, optional
        Deprecated v1 alias of ``body_doping`` (NMOS only).

    Examples
    --------
    >>> m = MOSFET(body_doping=1e17, tox=1e-6, width=1e-3, length=1e-4)
    >>> 0.0 < m.threshold_voltage() < 1.0
    True
    >>> p = MOSFET(body_doping=1e17, tox=1e-6, width=1e-3, length=1e-4, channel="p")
    >>> p.threshold_voltage() < 0.0
    True
    """

    terminals = ("gate", "drain", "source", "body")

    def __init__(
        self,
        body_doping: float | None = None,
        tox: float | None = None,
        width: float | None = None,
        length: float | None = None,
        channel: str = "n",
        material: Material = SILICON,
        temperature: float = 300.0,
        mobility: float | None = None,
        mobility_model: str = "constant",
        lambda_clm: float = 0.0,
        vfb: float | None = None,
        vsat: float | None = None,
        subthreshold: bool = True,
        *,
        na_substrate: float | None = None,
    ) -> None:
        if na_substrate is not None:
            if body_doping is not None:
                raise TypeError("pass either body_doping or na_substrate, not both")
            warnings.warn(
                "na_substrate is deprecated since semisim 2.0; use body_doping",
                DeprecationWarning,
                stacklevel=2,
            )
            body_doping = na_substrate
        missing = [
            name
            for name, value in (
                ("body_doping", body_doping),
                ("tox", tox),
                ("width", width),
                ("length", length),
            )
            if value is None
        ]
        if missing:
            raise TypeError(f"MOSFET missing required argument(s): {', '.join(missing)}")

        channel = str(channel).strip().lower()
        if channel in ("nmos", "p", "pmos", "n"):
            channel = channel[0]
        else:
            raise ValueError(f"channel must be 'n' or 'p', got {channel!r}")
        check_doping(body_doping, "body_doping")
        if tox <= 0:
            raise ValueError(f"tox must be positive, got {tox} cm")
        if width <= 0 or length <= 0:
            raise ValueError("width and length must be positive")
        if lambda_clm < 0:
            raise ValueError(f"lambda_clm must be >= 0, got {lambda_clm}")
        if vsat is not None and vsat <= 0:
            raise ValueError(f"vsat must be positive or None, got {vsat}")
        if mobility_model not in _MOBILITY_MODELS:
            raise ValueError(
                f"mobility_model must be one of {_MOBILITY_MODELS}, got {mobility_model!r}"
            )

        self.body_doping = float(body_doping)
        self.tox = tox
        self.width = width
        self.length = length
        self.channel = channel
        self.material = material
        self.temperature = temperature
        self.lambda_clm = lambda_clm
        self.vsat = vsat
        self.subthreshold = subthreshold
        self.mobility_model = mobility_model

        self.vt = thermal_voltage(temperature)  # also validates temperature
        if mobility is None:
            doping = self.body_doping if mobility_model == "caughey-thomas" else None
            mu_fn = material.mobility_n if channel == "n" else material.mobility_p
            mobility = mu_fn(temperature, doping)
        self.mobility = float(mobility)
        if self.mobility <= 0:
            raise ValueError("mobility must be positive")

        self._vfb_user = vfb

    # ------------------------------------------------------------------
    @property
    def polarity(self) -> int:
        """``+1`` for NMOS, ``-1`` for PMOS."""
        return 1 if self.channel == "n" else -1

    @property
    def na_substrate(self) -> float:
        """Deprecated v1 alias of :attr:`body_doping`."""
        return self.body_doping

    # ------------------------------------------------------------------
    # Derived electrostatic quantities
    # ------------------------------------------------------------------
    @property
    def cox(self) -> float:
        """Oxide capacitance per unit area ``Cox = eps_ox / tox`` [F/cm^2]."""
        return EPS_OX_R * EPS_0 / self.tox

    @property
    def phi_f(self) -> float:
        """Bulk Fermi potential magnitude ``phi_F = VT ln(N_body / ni)`` [V]."""
        return self.vt * np.log(self.body_doping / self.material.ni(self.temperature))

    @property
    def gamma(self) -> float:
        """Body-effect coefficient ``gamma = sqrt(2 q eps_s N_body) / Cox`` [V^0.5]."""
        eps_s = self.material.permittivity
        return float(np.sqrt(2.0 * Q * eps_s * self.body_doping) / self.cox)

    @property
    def flat_band_voltage(self) -> float:
        """Flat-band voltage [V].

        User-supplied value if given, otherwise the ideal same-type poly
        gate value ``-(Eg/2 + phi_F)`` (NMOS) or ``+(Eg/2 + phi_F)`` (PMOS)
        with zero oxide charge.
        """
        if self._vfb_user is not None:
            return self._vfb_user
        eg = self.material.bandgap(self.temperature)
        return -self.polarity * (eg / 2.0 + self.phi_f)

    def _vth_magnitude_term(self, vsb_n: float) -> float:
        arg = 2.0 * self.phi_f + vsb_n
        if arg <= 0:
            raise ValueError(
                f"2*phi_F + |Vsb| must be positive (forward body bias too large), got {arg:.3f} V"
            )
        return 2.0 * self.phi_f + self.gamma * np.sqrt(arg)

    def threshold_voltage(self, vsb: float = 0.0) -> float:
        """Threshold voltage including body effect.

        NMOS: ``Vth = Vfb + 2 phi_F + gamma sqrt(2 phi_F + Vsb)``
        PMOS: ``Vth = Vfb - 2 phi_F - gamma sqrt(2 phi_F - Vsb)``

        Parameters
        ----------
        vsb : float, optional
            Source-body voltage [V]: >= 0 is reverse body bias for NMOS,
            <= 0 for PMOS.

        Returns
        -------
        float
            Threshold voltage [V] (negative for a typical PMOS).

        Raises
        ------
        ValueError
            If the forward body bias exceeds ``2 phi_F``.
        """
        p = self.polarity
        return self.flat_band_voltage + p * self._vth_magnitude_term(p * vsb)

    def body_factor(self, vsb: float = 0.0) -> float:
        """Subthreshold slope factor ``m = 1 + gamma / (2 sqrt(2 phi_F + |Vsb|))``."""
        return 1.0 + self.gamma / (2.0 * np.sqrt(2.0 * self.phi_f + self.polarity * vsb))

    def subthreshold_swing(self, vsb: float = 0.0) -> float:
        """Subthreshold swing ``SS = m VT ln(10)`` [V/decade]."""
        return self.body_factor(vsb) * self.vt * np.log(10.0)

    def critical_field(self) -> float:
        """Velocity-saturation critical field ``Ec = 2 vsat / mu`` [V/cm] (inf if disabled)."""
        if self.vsat is None:
            return float("inf")
        return 2.0 * self.vsat / self.mobility

    # ------------------------------------------------------------------
    # Drain current
    # ------------------------------------------------------------------
    def _overdrive(self, vgs_n: np.ndarray, vsb_n: float) -> np.ndarray:
        """Effective overdrive in NMOS-normalized coordinates.

        Above threshold: ``Vgs - Vth``. With ``subthreshold=True`` an
        EKV-style softplus interpolation is used::

            Vov = 2 m VT ln(1 + exp((Vgs - Vth) / (2 m VT)))

        which tends to ``Vgs - Vth`` in strong inversion and yields
        ``Id ~ exp((Vgs - Vth) / (m VT))`` below threshold.
        """
        vth_n = self.polarity * self.flat_band_voltage + self._vth_magnitude_term(vsb_n)
        if not self.subthreshold:
            return np.maximum(vgs_n - vth_n, 0.0)
        m2vt = 2.0 * (1.0 + self.gamma / (2.0 * np.sqrt(2.0 * self.phi_f + vsb_n))) * self.vt
        u = (vgs_n - vth_n) / m2vt
        # Overflow-safe softplus: log(1 + e^u) = max(u, 0) + log1p(e^-|u|)
        return m2vt * (np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u))))

    def _id_normalized(
        self, vgs_n: np.ndarray, vds_n: np.ndarray, vsb_n: float
    ) -> np.ndarray:
        """Drain current magnitude in NMOS-normalized coordinates (no checks)."""
        vov = self._overdrive(vgs_n, vsb_n)
        k = self.mobility * self.cox * self.width / self.length
        if self.vsat is None:
            vdsat = vov
            id_triode = k * (vov - vds_n / 2.0) * vds_n
            id_sat = 0.5 * k * vov**2
        else:
            ecl = self.critical_field() * self.length
            vdsat = vov * ecl / (vov + ecl)
            id_triode = k * (vov - vds_n / 2.0) * vds_n / (1.0 + vds_n / ecl)
            id_sat = k * (vov - vdsat / 2.0) * vdsat / (1.0 + vdsat / ecl)
        i_d = np.where(vds_n < vdsat, id_triode, id_sat)
        return i_d * (1.0 + self.lambda_clm * vds_n)

    def saturation_voltage(
        self, vgs: float | np.ndarray, vsb: float = 0.0
    ) -> np.ndarray:
        """Drain saturation voltage ``Vdsat`` [V] (signed like ``vds``).

        Square law: ``Vdsat = Vov``; with velocity saturation
        ``Vdsat = Vov Ec L / (Vov + Ec L)``.
        """
        p = self.polarity
        vov = self._overdrive(p * np.asarray(vgs, dtype=float), p * vsb)
        if self.vsat is not None:
            ecl = self.critical_field() * self.length
            vov = vov * ecl / (vov + ecl)
        return p * vov

    def current(
        self,
        vgs: float | np.ndarray,
        vds: float | np.ndarray,
        vsb: float = 0.0,
    ) -> np.ndarray:
        """Drain current (positive into the drain; negative for PMOS).

        In NMOS-normalized coordinates (PMOS voltages and current are
        sign-flipped), triode (``Vds < Vdsat``)::

            Id = k (Vov - Vds/2) Vds / (1 + Vds/(Ec L)) * (1 + lambda Vds)

        saturation (``Vds >= Vdsat``): the triode expression at
        ``Vds = Vdsat``, times ``(1 + lambda Vds)``. Here ``k = mu Cox W/L``,
        ``Vov`` is the (softplus-interpolated) overdrive and ``Ec L -> inf``
        without velocity saturation (giving the classic square law).

        Parameters
        ----------
        vgs : float or ndarray
            Gate-source voltage [V].
        vds : float or ndarray
            Drain-source voltage [V]; must be >= 0 for NMOS and <= 0 for
            PMOS (source/drain not swapped).
        vsb : float, optional
            Source-body voltage [V] (scalar).

        Returns
        -------
        ndarray
            Drain current [A], broadcast over ``vgs`` and ``vds``.
        """
        p = self.polarity
        vgs_n = p * np.asarray(vgs, dtype=float)
        vds_n = p * np.asarray(vds, dtype=float)
        if np.any(vds_n < 0):
            sign = ">= 0" if p == 1 else "<= 0"
            raise ValueError(
                f"This {self.channel.upper()}MOS model requires Vds {sign} "
                "(source/drain not swapped)"
            )
        return p * self._id_normalized(vgs_n, vds_n, p * vsb)

    def iv(
        self,
        vgs: float | np.ndarray,
        vds: float | np.ndarray,
        vsb: float = 0.0,
    ) -> np.ndarray:
        """Alias for :meth:`current` (generic device interface)."""
        return self.current(vgs, vds, vsb)

    # ------------------------------------------------------------------
    # Small-signal parameters
    # ------------------------------------------------------------------
    def small_signal(
        self,
        vgs: float | np.ndarray,
        vds: float | np.ndarray,
        vsb: float = 0.0,
        step: float = 1e-6,
    ) -> dict[str, np.ndarray]:
        """Small-signal conductances at a bias point.

        Central finite differences of the drain current (the model is
        smooth, so the error is ``O(step^2)`` ~ 1e-12 relative)::

            gm  = dId/dVgs,   gds = dId/dVds,   gmb = dId/dVbs

        Parameters
        ----------
        vgs, vds : float or ndarray
            Bias point [V].
        vsb : float, optional
            Source-body voltage [V] (scalar).
        step : float, optional
            Finite-difference step [V].

        Returns
        -------
        dict of ndarray
            ``gm``, ``gds``, ``gmb`` [S] (all >= 0 for both polarities) and
            ``ro`` = 1/gds [ohm].
        """
        p = self.polarity
        vgs_n = p * np.asarray(vgs, dtype=float)
        vds_n = p * np.asarray(vds, dtype=float)
        if np.any(vds_n < 0):
            raise ValueError("vds has the wrong sign for this channel type")
        vsb_n = p * vsb
        f = self._id_normalized
        h = step
        gm = (f(vgs_n + h, vds_n, vsb_n) - f(vgs_n - h, vds_n, vsb_n)) / (2 * h)
        gds = (f(vgs_n, vds_n + h, vsb_n) - f(vgs_n, vds_n - h, vsb_n)) / (2 * h)
        # gmb = dId/dVbs = -dId/dVsb; keep the bracket inside the valid range.
        h_b = min(h, 0.5 * (2.0 * self.phi_f + vsb_n))
        gmb = -(f(vgs_n, vds_n, vsb_n + h_b) - f(vgs_n, vds_n, vsb_n - h_b)) / (2 * h_b)
        with np.errstate(divide="ignore"):
            ro = np.where(gds > 0, 1.0 / np.where(gds > 0, gds, 1.0), np.inf)
        return {"gm": gm, "gds": gds, "gmb": gmb, "ro": ro}

    # ------------------------------------------------------------------
    @property
    def parameters(self) -> dict[str, Any]:
        """Configuration plus key derived quantities."""
        return {
            "channel": f"{self.channel.upper()}MOS",
            "material": self.material.name,
            "body_doping [cm^-3]": self.body_doping,
            "tox [cm]": self.tox,
            "width [cm]": self.width,
            "length [cm]": self.length,
            "temperature [K]": self.temperature,
            "mobility [cm^2/Vs]": self.mobility,
            "lambda_clm [1/V]": self.lambda_clm,
            "vsat [cm/s]": self.vsat,
            "cox [F/cm^2]": self.cox,
            "phi_f [V]": self.phi_f,
            "gamma [V^0.5]": self.gamma,
            "flat_band_voltage [V]": self.flat_band_voltage,
            "threshold_voltage [V]": self.threshold_voltage(),
            "subthreshold_swing [V/dec]": self.subthreshold_swing(),
        }
