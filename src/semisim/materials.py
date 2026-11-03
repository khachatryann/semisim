"""Semiconductor material models (silicon, germanium, gallium arsenide).

Provides a :class:`Material` dataclass with temperature-dependent bandgap
(Varshni equation), intrinsic carrier concentration, effective densities of
states, and mobility with power-law temperature scaling and optional
Caughey-Thomas doping dependence.

Simplifications (documented in ``docs/theory.md``):

- Boltzmann (non-degenerate) statistics; a warning is emitted for doping
  above :data:`DEGENERATE_LIMIT`.
- Mobility temperature scaling is a single power law per carrier type,
  applied uniformly to the Caughey-Thomas curve.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .constants import EPS_0, K_B_EV, T_REF, thermal_voltage

#: Doping level above which Boltzmann statistics become unreliable [cm^-3].
DEGENERATE_LIMIT = 1e19


@dataclass(frozen=True)
class Material:
    """Bulk semiconductor material parameters.

    Parameters
    ----------
    name : str
        Human-readable material name.
    eg_0k : float
        Bandgap at 0 K [eV] (Varshni ``Eg(0)``).
    varshni_alpha : float
        Varshni alpha coefficient [eV/K].
    varshni_beta : float
        Varshni beta coefficient [K].
    nc_300 : float
        Conduction-band effective density of states at 300 K [cm^-3].
    nv_300 : float
        Valence-band effective density of states at 300 K [cm^-3].
    mu_n_300 : float
        Low-field electron mobility at 300 K [cm^2/(V.s)].
    mu_p_300 : float
        Low-field hole mobility at 300 K [cm^2/(V.s)].
    eps_r : float
        Relative permittivity (dimensionless).
    mu_temp_exp_n : float, optional
        Power-law exponent for electron mobility temperature scaling.
    mu_temp_exp_p : float, optional
        Power-law exponent for hole mobility temperature scaling.
    ct_mu_min_n, ct_mu_min_p : float, optional
        Caughey-Thomas minimum (heavy-doping) mobility at 300 K
        [cm^2/(V.s)]. ``None`` disables doping dependence for that carrier.
    ct_nref_n, ct_nref_p : float, optional
        Caughey-Thomas reference doping [cm^-3].
    ct_alpha_n, ct_alpha_p : float, optional
        Caughey-Thomas exponent (dimensionless).
    """

    name: str
    eg_0k: float
    varshni_alpha: float
    varshni_beta: float
    nc_300: float
    nv_300: float
    mu_n_300: float
    mu_p_300: float
    eps_r: float
    mu_temp_exp_n: float = -2.4
    mu_temp_exp_p: float = -2.2
    ct_mu_min_n: float | None = None
    ct_nref_n: float = 1e17
    ct_alpha_n: float = 0.7
    ct_mu_min_p: float | None = None
    ct_nref_p: float = 1e17
    ct_alpha_p: float = 0.7

    def bandgap(self, temperature: float = T_REF) -> float:
        """Bandgap via the Varshni equation.

        ``Eg(T) = Eg(0) - alpha * T^2 / (T + beta)``

        Parameters
        ----------
        temperature : float
            Absolute temperature [K].

        Returns
        -------
        float
            Bandgap [eV].
        """
        _check_temperature(temperature)
        t = temperature
        return self.eg_0k - self.varshni_alpha * t**2 / (t + self.varshni_beta)

    def nc(self, temperature: float = T_REF) -> float:
        """Conduction-band effective density of states, ``Nc(T) = Nc(300) (T/300)^1.5`` [cm^-3]."""
        _check_temperature(temperature)
        return self.nc_300 * (temperature / T_REF) ** 1.5

    def nv(self, temperature: float = T_REF) -> float:
        """Valence-band effective density of states, ``Nv(T) = Nv(300) (T/300)^1.5`` [cm^-3]."""
        _check_temperature(temperature)
        return self.nv_300 * (temperature / T_REF) ** 1.5

    def ni(self, temperature: float = T_REF) -> float:
        """Intrinsic carrier concentration.

        ``ni(T) = sqrt(Nc Nv) exp(-Eg / (2 kT))``

        Parameters
        ----------
        temperature : float
            Absolute temperature [K].

        Returns
        -------
        float
            Intrinsic carrier concentration [cm^-3].

        Notes
        -----
        With the tabulated Nc/Nv values this yields ~6e9 cm^-3 for Si at
        300 K. Literature values (1.0-1.5e10 cm^-3) vary with the choice of
        effective masses; agreement within a factor of ~2 is expected.
        """
        _check_temperature(temperature)
        eg = self.bandgap(temperature)
        return float(
            np.sqrt(self.nc(temperature) * self.nv(temperature))
            * np.exp(-eg / (2.0 * K_B_EV * temperature))
        )

    def mobility_n(self, temperature: float = T_REF, doping: float | None = None) -> float:
        """Electron mobility [cm^2/(V.s)].

        Without ``doping``: lattice-limited ``mu_n(T) = mu_n(300) (T/300)^a``.
        With ``doping`` (total ionized impurity concentration [cm^-3]) the
        Caughey-Thomas model is applied before temperature scaling::

            mu = mu_min + (mu_max - mu_min) / (1 + (N / Nref)^alpha)
        """
        _check_temperature(temperature)
        mu = _caughey_thomas(
            self.mu_n_300, self.ct_mu_min_n, self.ct_nref_n, self.ct_alpha_n, doping
        )
        return mu * (temperature / T_REF) ** self.mu_temp_exp_n

    def mobility_p(self, temperature: float = T_REF, doping: float | None = None) -> float:
        """Hole mobility [cm^2/(V.s)]; see :meth:`mobility_n` for the model."""
        _check_temperature(temperature)
        mu = _caughey_thomas(
            self.mu_p_300, self.ct_mu_min_p, self.ct_nref_p, self.ct_alpha_p, doping
        )
        return mu * (temperature / T_REF) ** self.mu_temp_exp_p

    def diffusion_n(self, temperature: float = T_REF, doping: float | None = None) -> float:
        """Electron diffusion coefficient via the Einstein relation, ``Dn = mu_n kT/q`` [cm^2/s]."""
        return self.mobility_n(temperature, doping) * thermal_voltage(temperature)

    def diffusion_p(self, temperature: float = T_REF, doping: float | None = None) -> float:
        """Hole diffusion coefficient via the Einstein relation, ``Dp = mu_p kT/q`` [cm^2/s]."""
        return self.mobility_p(temperature, doping) * thermal_voltage(temperature)

    @property
    def permittivity(self) -> float:
        """Absolute permittivity ``eps_r * eps_0`` [F/cm]."""
        return self.eps_r * EPS_0


def _caughey_thomas(
    mu_max: float, mu_min: float | None, nref: float, alpha: float, doping: float | None
) -> float:
    """Caughey-Thomas doping-dependent mobility at 300 K (lattice value if disabled)."""
    if doping is None or mu_min is None:
        return mu_max
    if doping < 0:
        raise ValueError(f"doping must be >= 0, got {doping} cm^-3")
    return mu_min + (mu_max - mu_min) / (1.0 + (doping / nref) ** alpha)


def _check_temperature(temperature: float) -> None:
    """Raise ``ValueError`` for non-positive temperatures."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature} K")


def check_doping(doping: float, label: str = "doping") -> None:
    """Validate a doping concentration.

    Raises ``ValueError`` if non-positive; warns if the level is degenerate
    (Boltzmann statistics unreliable above :data:`DEGENERATE_LIMIT`).

    Parameters
    ----------
    doping : float
        Doping concentration [cm^-3].
    label : str, optional
        Name used in the error/warning message.
    """
    if doping <= 0:
        raise ValueError(f"{label} must be positive, got {doping} cm^-3")
    if doping > DEGENERATE_LIMIT:
        warnings.warn(
            f"{label} = {doping:.3g} cm^-3 exceeds {DEGENERATE_LIMIT:.0e} cm^-3; "
            "Boltzmann statistics are unreliable for degenerate doping",
            UserWarning,
            stacklevel=3,
        )


#: Silicon (Varshni: Eg(0)=1.170 eV; Nc/Nv from Sze; mobilities at 300 K;
#: Caughey-Thomas parameters from Jacoboni et al., Solid-State Electron. 20, 77, 1977).
SILICON = Material(
    name="Si",
    eg_0k=1.170,
    varshni_alpha=4.73e-4,
    varshni_beta=636.0,
    nc_300=2.8e19,
    nv_300=1.04e19,
    mu_n_300=1350.0,
    mu_p_300=480.0,
    eps_r=11.7,
    ct_mu_min_n=92.0,
    ct_nref_n=1.3e17,
    ct_alpha_n=0.91,
    ct_mu_min_p=48.0,
    ct_nref_p=2.35e17,
    ct_alpha_p=0.88,
)

#: Germanium (Varshni: Eg(0)=0.7437 eV; Nc/Nv from Sze; mobilities at 300 K).
GERMANIUM = Material(
    name="Ge",
    eg_0k=0.7437,
    varshni_alpha=4.774e-4,
    varshni_beta=235.0,
    nc_300=1.04e19,
    nv_300=6.0e18,
    mu_n_300=3900.0,
    mu_p_300=1900.0,
    eps_r=16.0,
    ct_mu_min_n=850.0,
    ct_nref_n=2.6e17,
    ct_alpha_n=0.56,
    ct_mu_min_p=300.0,
    ct_nref_p=1.0e17,
    ct_alpha_p=1.0,
)

#: Gallium arsenide (Varshni: Eg(0)=1.519 eV, Vurgaftman 2001; Nc/Nv from
#: Sze; Caughey-Thomas parameters from Sotoodeh et al., JAP 87, 2890, 2000).
GALLIUM_ARSENIDE = Material(
    name="GaAs",
    eg_0k=1.519,
    varshni_alpha=5.405e-4,
    varshni_beta=204.0,
    nc_300=4.7e17,
    nv_300=9.0e18,
    mu_n_300=8500.0,
    mu_p_300=400.0,
    eps_r=12.9,
    mu_temp_exp_n=-1.0,
    mu_temp_exp_p=-2.1,
    ct_mu_min_n=500.0,
    ct_nref_n=6.0e16,
    ct_alpha_n=0.394,
    ct_mu_min_p=20.0,
    ct_nref_p=1.48e17,
    ct_alpha_p=0.38,
)

#: Registry of built-in materials, keyed by lowercase name.
MATERIALS: dict[str, Material] = {"si": SILICON, "ge": GERMANIUM, "gaas": GALLIUM_ARSENIDE}


def get_material(name: str) -> Material:
    """Look up a built-in material by name (case-insensitive).

    Parameters
    ----------
    name : str
        ``"Si"``/``"silicon"``, ``"Ge"``/``"germanium"``, or
        ``"GaAs"``/``"gallium arsenide"``.

    Returns
    -------
    Material

    Raises
    ------
    KeyError
        If the material is not registered.
    """
    key = name.strip().lower()
    aliases = {
        "silicon": "si",
        "germanium": "ge",
        "gallium arsenide": "gaas",
        "gallium_arsenide": "gaas",
    }
    key = aliases.get(key, key)
    if key not in MATERIALS:
        raise KeyError(f"Unknown material {name!r}; available: {sorted(MATERIALS)}")
    return MATERIALS[key]
