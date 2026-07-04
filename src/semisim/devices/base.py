"""Abstract base class for device models.

Every device exposes named terminals, a parameter dictionary, and a
``current(**bias)`` method with a uniform call convention. A future
SPICE-style netlist solver can therefore instantiate and evaluate devices
generically (e.g. for nodal analysis) without knowing their physics.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np


class Device(ABC):
    """Abstract two-or-more-terminal device model.

    Subclasses must define :attr:`terminals`, implement :meth:`current`,
    and expose their configuration through :attr:`parameters`.
    """

    #: Ordered terminal names, e.g. ``("anode", "cathode")``.
    terminals: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def current(self, **bias: float | np.ndarray) -> np.ndarray:
        """Terminal current for the given bias point(s).

        Parameters
        ----------
        **bias : float or ndarray
            Device-specific bias voltages [V] (e.g. ``voltage=`` for a
            diode; ``vgs=, vds=, vsb=`` for a MOSFET). Array inputs
            broadcast element-wise.

        Returns
        -------
        ndarray
            Current [A], broadcast over the inputs.
        """

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Dictionary of the device's configuration and derived quantities."""

    def summary(self) -> str:
        """Human-readable multi-line summary of the device parameters."""
        lines = [f"{type(self).__name__} (terminals: {', '.join(self.terminals)})"]
        for key, value in self.parameters.items():
            if isinstance(value, float):
                lines.append(f"  {key:24s} = {value:.6g}")
            else:
                lines.append(f"  {key:24s} = {value}")
        return "\n".join(lines)
