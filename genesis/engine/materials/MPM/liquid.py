from typing import Any

import quadrants as qd
from pydantic import StrictBool

import genesis as gs

from .base import Base


@qd.data_oriented
class Liquid(Base):
    """
    The liquid material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    viscous : bool, optional
        Whether the liquid is viscous. Simply sets mu to zero when non-viscous. Default is False.
    """

    viscous: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        pass

    @qd.func
    def _update_F_S_Jp_liquid(self, J, F_tmp, U, S, V, Jp):
        pass

    @qd.func
    def _update_stress_liquid(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        pass
