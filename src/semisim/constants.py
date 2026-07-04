"""Physical constants in cm-based semiconductor units.

All of ``semisim`` uses the unit system conventional in semiconductor
device textbooks (Sze, Pierret): lengths in cm, densities in cm^-3,
mobility in cm^2/(V.s), capacitance in F/cm^2, permittivity in F/cm.

Values are CODATA 2018 exact/recommended values.
"""

import numpy as np

#: Elementary charge [C].
Q = 1.602176634e-19

#: Boltzmann constant [J/K].
K_B = 1.380649e-23

#: Boltzmann constant [eV/K].
K_B_EV = 8.617333262e-5

#: Vacuum permittivity [F/cm].
EPS_0 = 8.8541878128e-14

#: Relative permittivity of SiO2 (gate oxide).
EPS_OX_R = 3.9

#: Reference temperature [K].
T_REF = 300.0


def thermal_voltage(temperature: float) -> float:
    """Thermal voltage ``kT/q``.

    Parameters
    ----------
    temperature : float
        Absolute temperature [K]. Must be positive.

    Returns
    -------
    float
        Thermal voltage [V] (~0.02585 V at 300 K).

    Raises
    ------
    ValueError
        If ``temperature`` is not positive.
    """
    if np.any(np.asarray(temperature) <= 0):
        raise ValueError(f"Temperature must be positive, got {temperature} K")
    return K_B * temperature / Q
