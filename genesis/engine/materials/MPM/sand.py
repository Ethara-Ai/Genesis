import math
from typing import Any, Literal

import quadrants as qd
from pydantic import Field

import genesis as gs
from genesis.typing import PositiveFloat, ValidFloat

from .base import Base, SamplerType


@qd.data_oriented
class Sand(Base):
    """
    The sand material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    sampler : str, optional
        Particle sampler. Default is 'random'.
    friction_angle : float, optional
        Friction angle in degrees, used to compute internal pressure-dependent plasticity. Default is 45.
    """

    sampler: SamplerType = "random"
    friction_angle: PositiveFloat = 45.0

    # Derived from friction_angle, set in model_post_init.
    alpha: ValidFloat = Field(default=0.0, exclude=True)

    def model_post_init(self, context: Any) -> None:
        pass

    @qd.func
    def _sand_projection(self, S, Jp):
        pass

    @qd.func
    def _update_F_S_Jp_sand(self, J, F_tmp, U, S, V, Jp):
        pass

    @qd.func
    def _update_stress_sand(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        pass
