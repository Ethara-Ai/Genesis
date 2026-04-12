import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class


@qd.func
def func_matvec_Ap(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    i_b,
):
    """
    Compute Ap = (M + J^T * diag(D) * J) * p on the current active set, which is used for solving the adjoint u.

    Specifically, M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D, and p = search direction.
    """
    pass


@qd.kernel
def kernel_solve_adjoint_u(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    r"""
    Solve for the adjoint vector [u] from Au = g, where A = dF/dqacc (primal Hessian on the active set) and g = dL/dqacc.
    Intuitively, [u] is a sensitivity vector that translates the upstream gradient dL/dqacc into the primal space.
    This adjoint vector [u] can be used an intermediate variable to compute the downstream gradients. Since A is a
    Semi-Positive Definite (SPD) matrix, we can solve A * u = g using either Cholesky decomposition or CG solver.
    When Newton solver was used, we reuse the Cholesky decomposition of A (= L * L^T) to solve A * u = g. Otherwise,
    we use CG solver.

    Specifically, A = M + J^T * diag(D) * J, where M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D.
    """
    pass


@qd.kernel
def kernel_compute_gradients(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    r"""
    Compute gradients of the loss with respect to the input variables to this solver. Note that we use the intermediate
    adjoint vector [u] computed in [kernel_solve_adjoint_u] to compute these gradients.

    Specifically, the gradients are computed as follows:
    - dL_dM = -u * qacc^T
    - dL_djac = -[u * y^T + qacc * (D \odot (Ju))^T] (y = D \odot w, w = (Jqacc - aref))
    - dL_daref = Ju \odot D
    - dL_defc_D = -Ju \odot (Jqacc - aref)
    - dL_dforce = u
    """
    pass
