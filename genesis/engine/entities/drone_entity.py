import os
import xml.etree.ElementTree as ET

import torch

import genesis as gs
from genesis.utils.misc import get_assets_dir

from .rigid_entity import RigidEntity


class DroneEntity(RigidEntity):
    def _load_scene(self, morph, surface):
        super()._load_scene(morph, surface)

        # additional drone specific attributes
        properties = ET.parse(os.path.join(get_assets_dir(), morph.file)).getroot()[0].attrib
        self._KF = float(properties["kf"])
        self._KM = float(properties["km"])

        self._n_propellers = len(morph.propellers_link_name)

        propellers_link = gs.List([self.get_link(name) for name in morph.propellers_link_name])
        self._propellers_link_idx = torch.tensor(
            [link.idx for link in propellers_link], dtype=gs.tc_int, device=gs.device
        )
        try:
            self._propellers_vgeom_idxs = torch.tensor(
                [link.vgeoms[0].idx for link in propellers_link], dtype=gs.tc_int, device=gs.device
            )
            self._animate_propellers = True
        except Exception:
            gs.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")
            self._animate_propellers = False

        self._propellers_spin = torch.tensor(morph.propellers_spin, dtype=gs.tc_float, device=gs.device)
        self._model = morph.model

    def _build(self):
        super()._build()

        self._propellers_revs = torch.zeros((self._n_propellers, self.solver._B), dtype=gs.tc_float, device=gs.device)
        self._prev_prop_t = None

    def set_propellers_rpm(self, propellers_rpm):
        """
        Set the RPM (revolutions per minute) for each propeller in the drone.

        Parameters
        ----------
        propellers_rpm : array-like or torch.Tensor
            A tensor or array of shape (n_propellers,) or (n_envs, n_propellers) specifying
            the desired RPM values for each propeller. Must be non-negative.

        Raises
        ------
        RuntimeError
            If the method is called more than once per simulation step, or if the input shape
            does not match the number of propellers, or contains negative values.
        """
        pass

    def update_propeller_vgeoms(self):
        """
        Update the visual geometry of the propellers for animation based on their current rotation.

        This method is a no-op if animation is disabled due to missing visual geometry.
        """
        if self._animate_propellers:
            self.solver.update_drone_propeller_vgeoms(
                self._propellers_vgeom_idxs, self._propellers_revs, self._propellers_spin
            )

    @property
    def model(self):
        """The model type of the drone."""
        return self._model

    @property
    def KF(self):
        """The drone's thrust coefficient."""
        pass

    @property
    def KM(self):
        """The drone's moment coefficient."""
        pass

    @property
    def n_propellers(self):
        """The number of propellers on the drone."""
        pass

    @property
    def COM_link_idx(self):
        """The index of the center-of-mass (COM) link of the drone."""
        pass

    @property
    def propellers_idx(self):
        """The indices of the drone's propeller links."""
        pass

    @property
    def propellers_spin(self):
        """The spin direction for each propeller."""
        pass
