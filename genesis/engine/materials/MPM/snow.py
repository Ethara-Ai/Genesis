from typing import Any

import quadrants as qd
from pydantic import StrictBool, model_validator

import genesis as gs
from genesis.typing import PositiveFloat

from .base import SamplerType
from .elasto_plastic import ElastoPlastic


@qd.data_oriented
class Snow(ElastoPlastic):
    """
    The snow material class for MPM.

    Note
    ----
    Snow is a special type of ElastoPlastic that gets harder when compressed.
    It does not support von Mises yield criterion.

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
    yield_lower : float, optional
        Lower bound of yield condition. Default is 2.5e-2.
    yield_higher : float, optional
        Upper bound of yield condition. Default is 4.5e-3.
    """

    sampler: SamplerType = "random"
    use_von_mises: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _enforce_no_von_mises(cls, data: dict) -> dict:
        pass

    def model_post_init(self, context: Any) -> None:
        pass

    @qd.func
    def _update_F_S_Jp_snow(self, J, F_tmp, U, S, V, Jp):
        pass

    @qd.func
    def _update_stress_snow(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # Hardening coefficient: material harder when compressed
        pass
