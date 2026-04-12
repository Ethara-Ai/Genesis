import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import quadrants as qd
import torch
import torch.nn.functional as F

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


class PathPlanner(ABC):
    def __init__(self, entity):
        self._entity = entity
        self._solver: "RigidSolver" = entity._solver

        self.PENETRATION_EPS = 1e-5 if gs.qd_float == qd.f32 else 0.0

        for joint in entity.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                gs.raise_exception("planning for the gs.JOINT_TYPE.FREE is not supported (yet)")
            elif joint.type == gs.JOINT_TYPE.SPHERICAL:
                gs.raise_exception("planning for the gs.JOINT_TYPE.SPHERICAL is not supported (yet)")

    @abstractmethod
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
    ): ...

    def get_link_pose(self, robot_g_link_idx, obj_g_link_idx, envs_idx):
        """
        Get the relative pose of a given robot link wrt some object link.

        Parameters
        ----------
        robot_g_link_idx: int
            Global link idx of the link of the robot.
        obj_g_link_idx: int
            Global link idx of the base link of the object.
        """
        pass

    def update_object(self, ee_link_idx, obj_link_idx, _pos, _quat, envs_idx):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------ util funcs ------------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_qposs(self, qpos_goal, qpos_start, envs_idx):
        pass

    def get_exclude_geom_pairs(self, qposs, envs_idx):
        """
        Parameters
        ----------
        qposs : list of torch.Tensor
            List of qpos tensors to ignore the collision check.
        envs_idx : torch.Tensor
            Environment indices.

        Returns
        -------
        unique_pairs : torch.Tensor
            Unique pairs of geom indices to ignore the collision check.
        """
        pass

    @qd.kernel
    def interpolate_path(
        self,
        path: qd.types.ndarray(),  # [N, B, Dof]
        sample_ind: qd.types.ndarray(),  # [B, 2]
        mask: qd.types.ndarray(),  # [B]
        tensor: qd.types.ndarray(),  # [N, B, Dof]
    ):
        pass

    def check_collision(
        self,
        path,
        ignore_geom_pairs,
        envs_idx,
        *,
        is_plan_with_obj=False,
        obj_geom_start=-1,
        obj_geom_end=-1,
        ee_link_idx=0,
        obj_link_idx=0,
        _pos=None,
        _quat=None,
    ):
        pass

    @qd.kernel
    def _kernel_check_collision(
        self,
        ignore_geom_pairs: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        is_plan_with_obj: qd.i32,
        obj_geom_start: qd.i32,
        obj_geom_end: qd.i32,
        out: qd.types.ndarray(),
        collider_state: array_class.ColliderState,
    ):
        pass

    @qd.func
    def _func_check_collision(
        self,
        collider_state: array_class.ColliderState,
        ignore_geom_pairs: qd.types.ndarray(),
        i_b: qd.i32,
        is_plan_with_obj: qd.i32 = False,
        obj_geom_start: qd.i32 = -1,
        obj_geom_end: qd.i32 = -1,
    ) -> qd.i32:
        pass

    def shortcut_path(
        self,
        path_mask,
        path,
        iterations=50,
        ignore_geom_pairs=None,
        envs_idx=None,
        is_plan_with_obj=False,
        obj_geom_start=-1,
        obj_geom_end=-1,
        ee_link_idx=0,
        obj_link_idx=0,
        _pos=None,
        _quat=None,
    ):
        """
        path_mask: torch.Tensor
            valid waypoint mask [N,B] for the obtained path
        path: torch.Tensor
            the [N,B,Dof] tensor containing batched waypoints
        iterations: int
            the number of refine iterations
        """
        pass


@qd.data_oriented
class RRT(PathPlanner):
    def __init__(self, entity):
        super().__init__(entity)
        self._is_rrt_init = False

    def _init_rrt_fields(self, goal_bias=0.05, max_nodes=2000, pos_tol=5e-3, max_step_size=0.1):
        pass

    def _reset_rrt_fields(self):
        pass

    @qd.kernel
    def _kernel_rrt_init(
        self, qpos_start: qd.types.ndarray(), qpos_goal: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        pass

    @qd.kernel
    def _kernel_rrt_step1(
        self,
        q_limit_lower: qd.types.ndarray(),
        q_limit_upper: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 1 includes:
        - generate random sample
        - find nearest neighbor
        - steer from nearest neighbor to random sample
        - add new node
        - set the steer result (to prepare for collision checking)
        """
        pass

    @qd.kernel
    def _kernel_rrt_step2(
        self,
        ignore_geom_pairs: qd.types.ndarray(),
        ignore_collision: qd.i32,
        envs_idx: qd.types.ndarray(),
        is_plan_with_obj: qd.i32,
        obj_geom_start: qd.i32,
        obj_geom_end: qd.i32,
        collider_state: array_class.ColliderState,
    ):
        """
        Step 2 includes:
        - check collision
        - if collision is detected, remove the new node
        - if collision is not detected, check if the new node is within goal configuration
        """
        pass

    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.05,
        timeout=None,
        max_nodes=2000,
        smooth_path=True,
        num_waypoints=100,
        ignore_collision=False,
        ee_link_idx=None,
        obj_entity=None,
        envs_idx=None,
    ):
        pass


@qd.data_oriented
class RRTConnect(PathPlanner):
    def __init__(self, entity):
        super().__init__(entity)
        self._is_rrt_connect_init = False

    def _init_rrt_connect_fields(self, goal_bias=0.1, max_nodes=4000, max_step_size=0.05):
        pass

    def _reset_rrt_connect_fields(self):
        pass

    @qd.kernel
    def _kernel_rrt_connect_init(
        self, qpos_start: qd.types.ndarray(), qpos_goal: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        # NOTE: run IK before this
        pass

    @qd.kernel
    def _kernel_rrt_connect_step1(
        self,
        qpos: array_class.V_ANNOTATION,
        forward_pass: qd.i32,
        q_limit_lower: qd.types.ndarray(),
        q_limit_upper: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 1 includes:
        - generate random sample
        - find nearest neighbor
        - steer from nearest neighbor to random sample
        - add new node
        - set the steer result (to prepare for collision checking)
        """
        pass

    @qd.kernel
    def _kernel_rrt_connect_step2(
        self,
        forward_pass: qd.i32,
        ignore_geom_pairs: qd.types.ndarray(),
        ignore_collision: qd.i32,
        envs_idx: qd.types.ndarray(),
        is_plan_with_obj: qd.i32,
        obj_geom_start: qd.i32,
        obj_geom_end: qd.i32,
        collider_state: array_class.ColliderState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 2 includes:
        - check collision
        - if collision is detected, remove the new node
        - if collision is not detected, check if the new node is within goal configuration
        """
        pass

    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.05,
        timeout=None,
        max_nodes=4000,
        smooth_path=True,
        num_waypoints=300,
        ignore_collision=False,
        ee_link_idx=None,
        obj_entity=None,
        envs_idx=None,
    ):
        pass


# ------------------------------------------------------------------------------------
# ------------------------------------ utils -----------------------------------------
# ------------------------------------------------------------------------------------


def align_waypoints_length(path: torch.Tensor, mask: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Aligns each waypoints length to the given num_points.

    Parameters
    ----------
    path: torch.Tensor
        path tensor in [N, B, Dof]
    mask: torch.Tensor
        the masking of path, indicating active waypoints [N, B]
    num_points: int
        the number of the desired waypoints

    Returns
    -------
        A new 2D PyTorch tensor [num_points, B, Dof]
    """
    pass


def rrt_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the RRTConnect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    pass


def rrt_connect_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the RRTConnect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    pass
