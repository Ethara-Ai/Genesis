"""
Inverse kinematics for rigid body entities.

This module contains the inverse kinematics kernel for computing joint configurations
that achieve desired end-effector poses.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.linalg as lu
import genesis.utils.array_class as array_class


# FIXME: RigidEntity is not compatible with fast cache
@qd.kernel(fastcache=False)
def kernel_rigid_entity_inverse_kinematics(
    rigid_entity: qd.template(),
    links_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    quats: qd.types.ndarray(),
    local_points: qd.types.ndarray(),
    dofs_idx: qd.types.ndarray(),
    custom_init_qpos: qd.i32,
    init_qpos: qd.types.ndarray(),
    max_samples: qd.i32,
    max_solver_iters: qd.i32,
    damping: qd.f32,
    pos_tol: qd.f32,
    rot_tol: qd.f32,
    pos_mask_: qd.types.ndarray(),
    rot_mask_: qd.types.ndarray(),
    link_pos_mask: qd.types.ndarray(),
    link_rot_mask: qd.types.ndarray(),
    max_step_size: qd.f32,
    respect_joint_limit: qd.i32,
    envs_idx: qd.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    pass
