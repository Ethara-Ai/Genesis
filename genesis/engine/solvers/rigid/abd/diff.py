"""
Backward pass functions for the rigid body solver.

This module contains functions used during the backward pass (gradient computation)
of the rigid body simulation. These functions handle:
- Copying state between next and current time steps
- Saving and loading adjoint cache for gradient computation
- Preparing and beginning backward substeps
- Gradient validity checking
- Cartesian space copying for adjoint computation
- Acceleration copying and dq integration

These functions are extracted from the main rigid_solver module to improve
code organization and maintainability.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .forward_kinematics import func_update_cartesian_space


@qd.func
def func_copy_next_to_curr(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    n_qs = rigid_global_info.qpos.shape[0]
    n_dofs = dofs_state.vel.shape[0]
    _B = dofs_state.vel.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # Prevent nan propagation
        is_valid = True
        for i_d in range(n_dofs):
            e = dofs_state.vel_next[i_d, i_b]
            is_valid &= not qd.math.isnan(e)
        for i_q in range(n_qs):
            e = rigid_global_info.qpos_next[i_q, i_b]
            is_valid &= not qd.math.isnan(e)

        if is_valid:
            for i_d in range(n_dofs):
                dofs_state.vel[i_d, i_b] = dofs_state.vel_next[i_d, i_b]

            for i_q in range(n_qs):
                rigid_global_info.qpos[i_q, i_b] = rigid_global_info.qpos_next[i_q, i_b]
        else:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_ACC_NAN


@qd.func
def func_copy_next_to_curr_grad(
    f: qd.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: qd.template(),
):
    pass


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_save_adjoint_cache(
    f: qd.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: qd.template(),
):
    func_save_adjoint_cache(f, dofs_state, rigid_global_info, rigid_adjoint_cache, static_rigid_sim_config)


@qd.func
def func_save_adjoint_cache(
    f: qd.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    n_qs = rigid_global_info.qpos.shape[0]
    _B = dofs_state.vel.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        rigid_adjoint_cache.dofs_vel[f, i_d, i_b] = dofs_state.vel[i_d, i_b]
        rigid_adjoint_cache.dofs_acc[f, i_d, i_b] = dofs_state.acc[i_d, i_b]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        rigid_adjoint_cache.qpos[f, i_q, i_b] = rigid_global_info.qpos[i_q, i_b]


@qd.func
def func_load_adjoint_cache(
    f: qd.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: qd.template(),
):
    pass


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_prepare_backward_substep(
    f: qd.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: qd.template(),
):
    # Load the current state from adjoint cache
    pass


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_begin_backward_substep(
    f: qd.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: qd.template(),
) -> qd.i32:
    pass


@qd.func
def func_is_grad_valid(
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: qd.template(),
):
    pass


@qd.func
def func_copy_cartesian_space(
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    joints_state: array_class.JointsState,
    geoms_state: array_class.GeomsState,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    static_rigid_sim_config: qd.template(),
):
    # Copy outputs of [kernel_update_cartesian_space] among [dofs, links, joints, geoms] states. This is used to restore
    # the outputs that were overwritten if we disabled mujoco compatibility for backward pass.

    # dofs state
    pass


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_copy_acc(
    f: qd.int32,
    dofs_state: array_class.DofsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: qd.template(),
):
    pass


@qd.func
def func_integrate_dq_entity(
    dq,
    i_e,
    i_b,
    respect_joint_limit,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    pass
