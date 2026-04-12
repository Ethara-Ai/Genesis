from typing import TYPE_CHECKING

import quadrants as qd
import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import qd_to_torch
from genesis.engine.entities.base_entity import Entity
from genesis.repr_base import RBC


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class Solver(RBC):
    def __init__(self, scene: "Scene", sim: "Simulator", options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._dt: float = options.dt
        self._substep_dt: float = options.dt / sim.substeps
        self._init_gravity = getattr(options, "gravity", None)
        self._gravity = None
        self._entities: list[Entity] = gs.List()

        self.data_manager = None

        # force fields
        self._ffs = list()

    def _add_force_field(self, force_field):
        pass

    def build(self):
        self._B = self._sim._B
        if self._init_gravity is not None:
            gravity = np.tile(np.asarray(self._init_gravity, dtype=gs.np_float), (self._B, 1))
            self._gravity = array_class.V(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    @gs.assert_built
    def set_gravity(self, gravity, envs_idx=None):
        pass

    def get_gravity(self, envs_idx=None):
        tensor = qd_to_torch(self._gravity, envs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def dump_ckpt_to_numpy(self) -> dict[str, np.ndarray]:
        pass

    def load_ckpt_from_numpy(self, arr_dict: dict[str, np.ndarray]) -> None:
        pass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        pass

    @property
    def scene(self):
        pass

    @property
    def sim(self):
        pass

    @property
    def dt(self):
        pass

    @property
    def is_built(self):
        pass

    @property
    def substep_dt(self):
        pass

    @property
    def gravity(self):
        pass

    @property
    def entities(self) -> list[Entity]:
        pass

    @property
    def n_entities(self):
        pass

    def _repr_brief(self):
        pass


@qd.kernel
def _kernel_set_gravity_field(tensor: qd.types.ndarray(), envs_idx: qd.types.ndarray(), gravity: qd.template()):
    pass


@qd.kernel
def _kernel_set_gravity_ndarray(tensor: qd.types.ndarray(), envs_idx: qd.types.ndarray(), gravity: qd.types.ndarray()):
    pass
