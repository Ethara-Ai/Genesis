from typing import Any

import quadrants as qd
from pydantic import StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


@qd.data_oriented
class ElastoPlastic(Base):
    """
    The elasto-plastic material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    yield_lower : float, optional
        Lower bound for the yield clamp (ignored if using von Mises). Default is 2.5e-2.
    yield_higher : float, optional
        Upper bound for the yield clamp (ignored if using von Mises). Default is 4.5e-3.
    use_von_mises : bool, optional
        Whether to use von Mises yield criterion. Default is True.
    von_mises_yield_stress : float, optional
        Yield stress for von Mises criterion. Default is 10000.
    """

    yield_lower: NonNegativeFloat = 2.5e-2
    yield_higher: NonNegativeFloat = 4.5e-3
    use_von_mises: StrictBool = True
    von_mises_yield_stress: PositiveFloat = 10000.0

    def model_post_init(self, context: Any) -> None:
        pass

    @qd.func
    def _update_F_S_Jp_elasto_plastic(self, J, F_tmp, U, S, V, Jp):
        pass
