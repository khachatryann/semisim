"""Semiconductor material models (silicon, germanium).

Provides a :class:`Material` dataclass with temperature-dependent bandgap
(Varshni equation), intrinsic carrier concentration, effective densities of
states, and power-law mobility scaling.

Simplifications (documented in ``docs/theory.md``):

- Boltzmann (non-degenerate) statistics; a warning is emitted for doping
  above :data:`DEGENERATE_LIMIT`.
- Mobility is doping-independent (no Caughey-Thomas model) and scales with
  temperature as a single power law per carrier type.
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

    def mobility_n(self, temperature: float = T_REF) -> float:
        """Electron mobility, ``mu_n(T) = mu_n(300) (T/300)^a`` [cm^2/(V.s)]."""
        _check_temperature(temperature)
        return self.mu_n_300 * (temperature / T_REF) ** self.mu_temp_exp_n

    def mobility_p(self, temperature: float = T_REF) -> float:
        """Hole mobility, ``mu_p(T) = mu_p(300) (T/300)^a`` [cm^2/(V.s)]."""
        _check_temperature(temperature)
        return self.mu_p_300 * (temperature / T_REF) ** self.mu_temp_exp_p

    def diffusion_n(self, temperature: float = T_REF) -> float:
        """Electron diffusion coefficient via the Einstein relation, ``Dn = mu_n kT/q`` [cm^2/s]."""
        return self.mobility_n(temperature) * thermal_voltage(temperature)

    def diffusion_p(self, temperature: float = T_REF) -> float:
        """Hole diffusion coefficient via the Einstein relation, ``Dp = mu_p kT/q`` [cm^2/s]."""
        return self.mobility_p(temperature) * thermal_voltage(temperature)

    @property
    def permittivity(self) -> float:
        """Absolute permittivity ``eps_r * eps_0`` [F/cm]."""
        return self.eps_r * EPS_0


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


#: Silicon (Varshni: Eg(0)=1.170 eV; Nc/Nv from Sze; mobilities at 300 K).
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
)

#: Registry of built-in materials, keyed by lowercase name.
MATERIALS: dict[str, Material] = {"si": SILICON, "ge": GERMANIUM}


def get_material(name: str) -> Material:
    """Look up a built-in material by name (case-insensitive).

    Parameters
    ----------
    name : str
        ``"Si"``/``"silicon"`` or ``"Ge"``/``"germanium"``.

    Returns
    -------
    Material

    Raises
    ------
    KeyError
        If the material is not registered.
    """
    key = name.strip().lower()
    aliases = {"silicon": "si", "germanium": "ge"}
    key = aliases.get(key, key)
    if key not in MATERIALS:
        raise KeyError(f"Unknown material {name!r}; available: {sorted(MATERIALS)}")
    return MATERIALS[key]
