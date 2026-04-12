from typing import Any, Literal

import quadrants as qd

import genesis as gs

from genesis.typing import PositiveFloat

from .base import Base


@qd.data_oriented
class Elastic(Base):
    """
    The elastic material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 3e5.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohooken'). Default is 'corotation'.
    """

    E: PositiveFloat = 3e5
    model: Literal["corotation", "neohooken"] = "corotation"

    def model_post_init(self, context: Any) -> None:
        pass

    @qd.func
    def _update_F_S_Jp_elastic(self, J, F_tmp, U, S, V, Jp):
        pass

    @qd.func
    def _update_stress_corotation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        pass

    @qd.func
    def _update_stress_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        pass
