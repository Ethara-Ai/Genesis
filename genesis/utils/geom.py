import math
from typing import Literal

import numba as nb
import numpy as np
import quadrants as qd
import torch
import torch.nn.functional as F

import genesis as gs
from genesis.typing import Vec3FType

# ------------------------------------------------------------------------------------
# ------------------------------------- Quadrants ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def qd_i_cross_vec(vec):
    return qd.Vector([0.0, -vec[2], vec[1]], dt=gs.qd_float)


@qd.func
def qd_j_cross_vec(vec):
    return qd.Vector([vec[2], 0.0, -vec[0]], dt=gs.qd_float)


@qd.func
def qd_k_cross_vec(vec):
    pass


@qd.func
def qd_transform_by_quat_fast(v, quat):
    """
    Assumptions:
    - quat must be normalized
    """
    q_w, q_x, q_y, q_z = quat
    u = qd.Vector([q_x, q_y, q_z])
    t = 2.0 * u.cross(v)
    return v + q_w * t + u.cross(t)


@qd.func
def qd_xyz_to_quat(xyz):
    """
    Convert intrinsic x-y-z Euler angles to quaternion.
    """
    ai, aj, ak = 0.5 * xyz[2], -0.5 * xyz[1], 0.5 * xyz[0]
    si, sj, sk = qd.sin(ai), qd.sin(aj), qd.sin(ak)
    ci, cj, ck = qd.cos(ai), qd.cos(aj), qd.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = qd.Vector(
        [
            +cj * cc + sj * ss,
            +cj * cs - sj * sc,
            -cj * ss - sj * cc,
            +cj * sc - sj * cs,
        ],
        dt=gs.qd_float,
    )
    return quat


@qd.func
def qd_R_to_xyz(R, eps):
    """
    Convert a rotation matrix into intrinsic x-y-z Euler angles.
    """
    pass


@qd.func
def qd_rotvec_to_R(rotvec, eps):
    R = qd.Matrix.identity(gs.qd_float, 3)

    angle = rotvec.norm()
    if angle > eps:
        c = qd.cos(angle)
        s = qd.sqrt(1.0 - c**2)
        t = 1.0 - c
        x, y, z = rotvec / angle

        R = qd.Matrix(
            [
                [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
            ],
            dt=gs.qd_float,
        )

    return R


@qd.func
def qd_rotvec_to_quat(rotvec, eps):
    quat = qd.Vector.zero(gs.qd_float, 4)

    # We need to use [norm_sqr] instead of [norm] to avoid nan gradients in the backward pass. Even when theta = 0,
    # the gradient of [norm] operation is computed and used (note that the gradient becomes NaN when theta = 0). This
    # is seemd to be a bug in Quadrants autodiff @TODO: change back after the bug is fixed.
    thetasq = rotvec.norm_sqr()
    if thetasq > (eps**2):
        theta = qd.sqrt(thetasq)
        theta_half = 0.5 * theta
        c, s = qd.cos(theta_half), qd.sin(theta_half)

        quat[0] = c
        xyz = s / theta * rotvec
        for i in qd.static(range(3)):
            quat[i + 1] = xyz[i]

        # First order quaternion normalization is accurate enough yet necessary
        quat *= 0.5 * (3.0 - quat.norm_sqr())
    else:
        quat[0] = 1.0

    return quat


@qd.func
def qd_quat_to_R(quat, eps):
    """
    Converts quaternion to 3x3 rotation matrix.
    """
    R = qd.Matrix.identity(gs.qd_float, 3)

    d = quat.norm_sqr()
    if d > eps:
        s = 2.0 / d
        w, x, y, z = quat
        xs, ys, zs = x * s, y * s, z * s
        wx, wy, wz = w * xs, w * ys, w * zs
        xx, xy, xz = x * xs, x * ys, x * zs
        yy, yz, zz = y * ys, y * zs, z * zs

        R = qd.Matrix(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dt=gs.qd_float,
        )

    return R


@qd.func
def qd_quat_to_xyz(quat, eps):
    """
    Convert a quaternion into intrinsic x-y-z Euler angles.
    """
    roll = gs.qd_float(0.0)
    pitch = gs.qd_float(0.0)
    yaw = gs.qd_float(0.0)

    quat_norm_sqr = quat.norm_sqr()
    if quat_norm_sqr > eps:
        s = 2.0 / quat_norm_sqr
        q_w, q_x, q_y, q_z = quat
        q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
        q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
        q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
        q_yy, q_yz, q_zz = q_y * q_ys, q_y * q_zs, q_z * q_zs

        sinycosp = q_wz - q_xy
        cosycosp = 1.0 - (q_yy + q_zz)
        cosp = qd.sqrt(cosycosp**2 + sinycosp**2)

        pitch = qd.atan2(q_xz + q_wy, cosp)
        if cosp > eps:
            roll = qd.atan2(q_wx - q_yz, 1.0 - (q_xx + q_yy))
            yaw = qd.atan2(sinycosp, cosycosp)
        else:
            yaw = qd.atan2(q_wz + q_xy, 1.0 - (q_xx + q_zz))

    return qd.Vector([roll, pitch, yaw], dt=gs.qd_float)


@qd.func
def qd_quat_to_rotvec(quat, eps):
    q_w, q_x, q_y, q_z = quat
    rotvec = qd.Vector([q_x, q_y, q_z], dt=gs.qd_float)

    s2 = rotvec.norm()
    if s2 > eps:
        angle = 2.0 * qd.atan2(s2, qd.abs(q_w))
        inv_sinc = angle / s2
        rotvec = (-1.0 if q_w < 0.0 else 1.0) * inv_sinc * rotvec

    return rotvec


@qd.func
def qd_trans_quat_to_T(trans, quat, eps):
    T = qd.Matrix.identity(gs.qd_float, 4)
    T[:3, :3] = qd_quat_to_R(quat, eps)
    T[:3, 3] = trans
    return T


@qd.func
def qd_inv_quat(quat):
    return qd.Vector([quat[0], -quat[1], -quat[2], -quat[3]], dt=gs.qd_float)


@qd.func
def qd_quat_mul_axis(q, axis):
    return qd.Vector(
        [
            -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
            +q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
            +q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
            +q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
        ]
    )


@qd.func
def qd_quat_mul(u, v):
    vu = u.outer_product(v)
    w = vu[0, 0] - vu[1, 1] - vu[2, 2] - vu[3, 3]
    x = vu[0, 1] + vu[1, 0] + vu[2, 3] - vu[3, 2]
    y = vu[0, 2] - vu[1, 3] + vu[2, 0] + vu[3, 1]
    z = vu[0, 3] + vu[1, 2] - vu[2, 1] + vu[3, 0]
    return qd.Vector([w, x, y, z], dt=gs.qd_float)


@qd.func
def qd_transform_quat_by_quat(v, u):
    """Transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    vec = qd_quat_mul(u, v)
    return vec.normalized()


@qd.func
def qd_transform_by_quat(v, quat):
    q_w, q_x, q_y, q_z = quat
    q_xx, q_xy, q_xz, q_wx = q_x * q_x, q_x * q_y, q_x * q_z, q_x * q_w
    q_yy, q_yz, q_wy = q_y * q_y, q_y * q_z, q_y * q_w
    q_zz, q_wz = q_z * q_z, q_z * q_w
    q_ww = q_w * q_w

    return qd.Vector(
        [
            v.x * (q_xx + q_ww - q_yy - q_zz) + v.y * (2.0 * q_xy - 2.0 * q_wz) + v.z * (2.0 * q_xz + 2.0 * q_wy),
            v.x * (2.0 * q_wz + 2.0 * q_xy) + v.y * (q_ww - q_xx + q_yy - q_zz) + v.z * (-2.0 * q_wx + 2.0 * q_yz),
            v.x * (-2.0 * q_wy + 2.0 * q_xz) + v.y * (2.0 * q_wx + 2.0 * q_yz) + v.z * (q_ww - q_xx - q_yy + q_zz),
        ],
        dt=gs.qd_float,
    ) / (q_ww + q_xx + q_yy + q_zz)


@qd.func
def qd_inv_transform_by_quat(v, quat):
    return qd_transform_by_quat(v, qd_inv_quat(quat))


@qd.func
def qd_transform_by_T(pos, T):
    return T[:3, :3] @ pos + T[:3, 3]


@qd.func
def qd_inv_transform_by_T(pos, T):
    pass


@qd.func
def qd_transform_by_trans_quat(pos, trans, quat):
    return qd_transform_by_quat(pos, quat) + trans


@qd.func
def qd_inv_transform_by_trans_quat(pos, trans, quat):
    return qd_transform_by_quat(pos - trans, qd_inv_quat(quat))


@qd.func
def qd_transform_motion_by_trans_quat(m_ang, m_vel, trans, quat):
    quat_inv = qd_inv_quat(quat)
    ang = qd_transform_by_quat(m_ang, quat_inv)
    vel = qd_transform_by_quat(m_vel - trans.cross(m_ang), quat_inv)
    return ang, vel


@qd.func
def qd_inv_transform_motion_by_trans_quat(m_ang, m_vel, trans, quat):
    pass


@qd.func
def qd_transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + qd_transform_by_quat(pos, t_quat)
    new_quat = qd_transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat


@qd.func
def qd_transform_inertia_by_trans_quat(i_inertial, i_mass, trans, quat, eps):
    x, y, z = trans.x, trans.y, trans.z
    xx, xy, xz, yy, yz, zz = x * x, x * y, x * z, y * y, y * z, z * z
    hhT = qd.Matrix(
        [
            [yy + zz, -xy, -xz],
            [-xy, xx + zz, -yz],
            [-xz, -yz, xx + yy],
        ]
    )

    R = qd_quat_to_R(quat, eps)
    i = R @ i_inertial @ R.transpose() + hhT * i_mass
    trans = trans * i_mass

    return i, trans, quat, i_mass


@qd.func
def qd_normalize(v, eps):
    return v / (v.norm(eps))


@qd.func
def qd_identity_quat():
    return qd.Vector([1.0, 0.0, 0.0, 0.0], dt=gs.qd_float)


@qd.func
def qd_vec3(val):
    return qd.Vector([val, val, val], dt=gs.qd_float)


@qd.func
def qd_nowhere():
    # let's inject a bit of humor here
    return qd.Vector([-2333333, -6666666, -5201314], dt=gs.qd_float)


@qd.func
def qd_tet_vol(p0, p1, p2, p3):
    return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0


@qd.func
def inertial_mul(pos, i, mass, vel, ang):
    _ang = i @ ang + pos.cross(vel)
    _vel = mass * vel - pos.cross(ang)
    return _ang, _vel


@qd.func
def motion_cross_force(m_ang, m_vel, f_ang, f_vel):
    vel = m_ang.cross(f_vel)
    ang = m_ang.cross(f_ang) + m_vel.cross(f_vel)
    return ang, vel


@qd.func
def motion_cross_motion(s_ang, s_vel, m_ang, m_vel):
    vel = s_ang.cross(m_vel) + s_vel.cross(m_ang)
    ang = s_ang.cross(m_ang)
    return ang, vel


@qd.func
def qd_orthogonals(a):
    """
    Returns orthogonal vectors `b` and `c`, given a normal vector `a`.

    Note that `a` is assumed to be normalized.
    """
    b = qd.Vector.zero(gs.qd_float, 3)
    if qd.abs(a[1]) < 0.5:
        b[0] = -a[0] * a[1]
        b[1] = 1.0 - a[1] ** 2
        b[2] = -a[2] * a[1]
    else:
        b[0] = -a[0] * a[2]
        b[1] = -a[1] * a[2]
        b[2] = 1.0 - a[2] ** 2
    b = b.normalized()
    return b.cross(a), b


@qd.func
def imp_aref(params, neg_penetration, vel, pos):
    timeconst, dampratio, dmin, dmax, width, mid, power = params

    imp_x = qd.abs(neg_penetration) / width
    imp_a = (1.0 / mid ** (power - 1)) * imp_x**power
    imp_b = 1.0 - (1.0 / (1.0 - mid) ** (power - 1)) * (1.0 - imp_x) ** power
    imp_y = imp_a if imp_x < mid else imp_b

    imp = dmin + imp_y * (dmax - dmin)
    imp = qd.math.clamp(imp, dmin, dmax)
    imp = dmax if imp_x > 1.0 else imp

    b = 2.0 / (dmax * timeconst)
    k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

    aref = -b * vel - k * imp * pos

    return imp, aref


# ------------------------------------------------------------------------------------
# -------------------------------- torch and numpy -----------------------------------
# ------------------------------------------------------------------------------------


def inv_quat(quat):
    if isinstance(quat, torch.Tensor):
        _quat = quat.clone()
        _quat[..., 1:].neg_()
    elif isinstance(quat, np.ndarray):
        _quat = quat.copy()
        _quat[..., 1:] *= -1
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
    return _quat


def inv_T(T):
    if isinstance(T, torch.Tensor):
        T_inv = torch.zeros_like(T)
    elif isinstance(T, np.ndarray):
        T_inv = np.zeros_like(T)
    else:
        gs.raise_exception(f"the input must be torch.Tensor or np.ndarray. got: {type(T)=}")

    trans, R = T[..., :3, 3], T[..., :3, :3]
    T_inv[..., 3, 3] = 1.0
    T_inv[..., :3, 3] = -R.T @ trans
    T_inv[..., :3, :3] = R.T

    return T_inv


def normalize(x, eps: float = 1e-12):
    pass


def rot6d_to_R(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] http://arxiv.org/abs/1812.07035
    """
    pass


def R_to_rot6d(R):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        R: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] http://arxiv.org/abs/1812.07035
    """
    pass


@nb.jit(nopython=True, cache=True)
def _np_xyz_to_quat(xyz: np.ndarray, rpy: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the (qw, qx, qy, qz) Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param xyz: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert xyz.ndim >= 1
    if out is None:
        out_ = np.empty((*xyz.shape[:-1], 4), dtype=xyz.dtype)
    else:
        assert out.shape == (*xyz.shape[:-1], 4)
        out_ = out

    rpy2 = 0.5 * xyz
    roll2, pitch2, yaw2 = rpy2[..., 0], rpy2[..., 1], rpy2[..., 2]
    cosr, sinr = np.cos(roll2), np.sin(roll2)
    cosp, sinp = np.cos(pitch2), np.sin(pitch2)
    cosy, siny = np.cos(yaw2), np.sin(yaw2)
    sign = 1.0 if rpy else -1.0

    out_[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out_[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out_[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out_[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out_


@torch.jit.script
def _tc_xyz_to_quat(xyz: torch.Tensor, rpy: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty(xyz.shape[:-1] + (4,), dtype=xyz.dtype, device=xyz.device)

    roll2, pitch2, yaw2 = (0.5 * xyz).unbind(-1)
    cosr, sinr = roll2.cos(), roll2.sin()
    cosp, sinp = pitch2.cos(), pitch2.sin()
    cosy, siny = yaw2.cos(), yaw2.sin()
    sign = 1.0 if rpy else -1.0

    out[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out


def xyz_to_quat(xyz, rpy=False, degrees=False):
    if isinstance(xyz, torch.Tensor):
        if degrees:
            xyz = torch.deg2rad(xyz)
        return _tc_xyz_to_quat(xyz, rpy)
    elif isinstance(xyz, np.ndarray):
        if degrees:
            xyz = np.deg2rad(xyz)
        return _np_xyz_to_quat(xyz, rpy)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(xyz)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_R(quat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of quaternions.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3, 3)
        out_ = out

    s = 2.0 / np.sum(np.square(quat), -1)
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz = q_y * q_ys, q_y * q_zs
    q_zz = q_z * q_zs

    out_[..., 0, 0] = 1.0 - (q_yy + q_zz)
    out_[..., 0, 1] = q_xy - q_wz
    out_[..., 0, 2] = q_xz + q_wy
    out_[..., 1, 0] = q_xy + q_wz
    out_[..., 1, 1] = 1.0 - (q_xx + q_zz)
    out_[..., 1, 2] = q_yz - q_wx
    out_[..., 2, 0] = q_xz - q_wy
    out_[..., 2, 1] = q_yz + q_wx
    out_[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return out_


@torch.jit.script
def _tc_quat_to_R(quat, out: torch.Tensor | None = None):
    if out is None:
        R = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    else:
        assert out.shape == quat.shape[:-1] + (3, 3)
        R = out

    s = 2 / (quat**2).sum(dim=-1, keepdim=True)
    q_vec_s = s * quat[..., 1:]

    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    q_sx, q_sy, q_sz = q_vec_s[..., 0], q_vec_s[..., 1], q_vec_s[..., 2]
    q_wx, q_wy, q_wz = q_w * q_sx, q_w * q_sy, q_w * q_sz
    q_xx, q_xy, q_xz = q_x * q_sx, q_x * q_sy, q_x * q_sz
    q_yy, q_yz = q_y * q_sy, q_y * q_sz
    q_zz = q_z * q_sz

    R[..., 0, 0] = 1.0 - (q_yy + q_zz)
    R[..., 0, 1] = q_xy - q_wz
    R[..., 0, 2] = q_xz + q_wy
    R[..., 1, 0] = q_xy + q_wz
    R[..., 1, 1] = 1.0 - (q_xx + q_zz)
    R[..., 1, 2] = q_yz - q_wx
    R[..., 2, 0] = q_xz - q_wy
    R[..., 2, 1] = q_yz + q_wx
    R[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return R


def quat_to_R(quat, *, out=None):
    # NOTE: Ignore zero-norm quaternion for efficiency

    if all(isinstance(e, torch.Tensor) for e in (quat, out) if e is not None):
        return _tc_quat_to_R(quat, out=out)
    elif all(isinstance(e, np.ndarray) for e in (quat, out) if e is not None):
        return _np_quat_to_R(quat, out=out)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_xyz(quat, rpy=False, out=None):
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a batch of quaternions.

    The Roll, Pitch and Yaw angles are guaranteed to be within range [-pi,pi], [-pi/2,pi/2], [-pi,pi], respectively.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    pass


@torch.jit.script
def _tc_quat_to_xyz(quat, eps: float, rpy: bool = False):
    pass


def quat_to_xyz(quat, rpy=False, degrees=False):
    pass


@nb.jit(nopython=True, cache=True)
def _np_R_to_quat(R, out=None):
    assert R.ndim >= 2

    if out is None:
        out_ = np.empty((*R.shape[:-2], 4), dtype=R.dtype)
    else:
        assert out.shape == (*R.shape[:-2], 4)
        out_ = out

    for i in np.ndindex(R.shape[:-2]):
        R_i = R[i]
        quat_i = out_[i]

        if R_i[2, 2] < 0.0:
            if R_i[0, 0] > R_i[1, 1]:
                t = 1.0 + R_i[0, 0] - R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[2, 1] - R_i[1, 2]
                quat_i[1] = t
                quat_i[2] = R_i[1, 0] + R_i[0, 1]
                quat_i[3] = R_i[0, 2] + R_i[2, 0]
            else:
                t = 1.0 - R_i[0, 0] + R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[0, 2] - R_i[2, 0]
                quat_i[1] = R_i[1, 0] + R_i[0, 1]
                quat_i[2] = t
                quat_i[3] = R_i[2, 1] + R_i[1, 2]
        else:
            if R_i[0, 0] < -R_i[1, 1]:
                t = 1.0 - R_i[0, 0] - R_i[1, 1] + R_i[2, 2]
                quat_i[0] = R_i[1, 0] - R_i[0, 1]
                quat_i[1] = R_i[0, 2] + R_i[2, 0]
                quat_i[2] = R_i[2, 1] + R_i[1, 2]
                quat_i[3] = t
            else:
                t = 1.0 + R_i[0, 0] + R_i[1, 1] + R_i[2, 2]
                quat_i[0] = t
                quat_i[1] = R_i[2, 1] - R_i[1, 2]
                quat_i[2] = R_i[0, 2] - R_i[2, 0]
                quat_i[3] = R_i[1, 0] - R_i[0, 1]
        quat_i /= 2.0 * np.sqrt(t)

    return out_


@torch.jit.script
def _tc_R_to_quat(R, out=None):
    if out is None:
        quat = torch.zeros(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)
    else:
        # assert out.shape == R.shape[:-2] + (4,)
        quat = out

    r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = r11 + r22 + r33
    c0 = trace > 0.0
    c1 = (r11 > r22) & (r11 > r33)
    c2 = r22 > r33
    s = torch.where(
        c0,
        torch.sqrt(trace + 1.0) * 2,
        torch.where(
            c1,
            torch.sqrt(1.0 + r11 - r22 - r33) * 2,
            torch.where(
                c2,
                torch.sqrt(1.0 + r22 - r11 - r33) * 2,
                torch.sqrt(1.0 + r33 - r11 - r22) * 2,
            ),
        ),
    )
    return torch.where(
        c0[..., None],
        torch.stack([0.25 * s, (r32 - r23) / s, (r13 - r31) / s, (r21 - r12) / s], dim=-1),
        torch.where(
            c1[..., None],
            torch.stack([(r32 - r23) / s, 0.25 * s, (r12 + r21) / s, (r13 + r31) / s], dim=-1),
            torch.where(
                c2[..., None],
                torch.stack([(r13 - r31) / s, (r12 + r21) / s, 0.25 * s, (r23 + r32) / s], dim=-1),
                torch.stack([(r21 - r12) / s, (r13 + r31) / s, (r23 + r32) / s, 0.25 * s], dim=-1),
            ),
        ),
    )

    return quat


def R_to_quat(R, *, out=None):
    if all(isinstance(e, torch.Tensor) for e in (R, out) if e is not None):
        return _tc_R_to_quat(R, out=out)
    elif all(isinstance(e, np.ndarray) for e in (R, out) if e is not None):
        return _np_R_to_quat(R, out=out)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


def R_to_xyz(R, rpy=False, degrees=False):
    pass


def trans_R_to_T(trans=None, R=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, R) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, R) if e is not None)

    T = out
    B = () if trans is None and R is None else R.shape[:-2] if trans is None else trans.shape[:-1]
    dtype = R.dtype if trans is None else trans.dtype
    if is_torch:
        if T is None:
            device = R.device if trans is None else trans.device
            T = torch.zeros((*B, 4, 4), dtype=dtype, device=device)
    elif is_numpy:
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=dtype)
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(R)=}")
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if R is None:
        if is_torch:
            torch.diagonal(T, dim1=-2, dim2=-1).fill_(1.0)
        else:
            T[..., [0, 1, 2], [0, 1, 2]] = 1.0
    else:
        T[..., :3, :3] = R

    return T


def R_to_T(R, *, out=None):
    pass


def trans_to_T(trans, *, out=None):
    return trans_R_to_T(trans, None, out=out)


def trans_quat_to_T(trans=None, quat=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, quat) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, quat) if e is not None)

    T = out
    B = () if trans is None and quat is None else quat.shape[:-2] if trans is None else trans.shape[:-1]
    if is_torch:
        if T is None:
            T = torch.zeros((*B, 4, 4), dtype=trans.dtype, device=trans.device)
    elif is_numpy:
        dtype = np.result_type(trans, quat)
        trans, quat = trans.astype(dtype), quat.astype(dtype)
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=dtype)
    else:
        gs.raise_exception(
            f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(quat)=}"
        )
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if quat is not None:
        quat_to_R(quat, out=T[..., :3, :3])

    return T


def T_to_trans(T):
    return T[..., :3, 3]


def T_to_quat(T):
    return R_to_quat(T[..., :3, :3])


def T_to_trans_quat(T):
    return T_to_trans(T), T_to_quat(T)


@nb.jit(nopython=True, cache=True)
def _np_quat_mul(u, v, out=None):
    assert u.shape == v.shape
    u_2d = np.atleast_2d(u)
    v_2d = np.atleast_2d(v)

    w1, x1, y1, z1 = u_2d[..., 0], u_2d[..., 1], u_2d[..., 2], u_2d[..., 3]
    w2, x2, y2, z2 = v_2d[..., 0], v_2d[..., 1], v_2d[..., 2], v_2d[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    if out is None:
        out_ = np.empty(u_2d.shape, dtype=qq.dtype)
    else:
        assert out.shape == u.shape
        out_ = out

    out_[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out_[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out_[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out_[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    out_ /= np.sqrt(np.sum(np.square(np.expand_dims(out_, -2)), -1))

    return out_.reshape(u.shape)


@torch.jit.script
def _tc_quat_mul(u, v):
    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out = torch.empty(qq.shape + (4,), dtype=qq.dtype, device=qq.device)
    out[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    out /= torch.linalg.vector_norm(out, ord=2, dim=-1, keepdim=True)
    return out


def transform_quat_by_quat(v, u):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert u.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (u, v)):
        quat = _tc_quat_mul(u, v)
    elif all(isinstance(e, np.ndarray) for e in (u, v)):
        quat = _np_quat_mul(u, v, out=None)
    else:
        gs.raise_exception(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")

    return quat


@nb.jit(nopython=True, cache=True)
def _np_transform_by_quat(v, quat, out=None):
    if out is None:
        out_ = np.empty(v.shape, dtype=v.dtype)
    else:
        assert out.shape == v.shape
        out_ = out

    v_T, quat_T, out_T = v.T, quat.T, out_.T
    v_x, v_y, v_z = v_T
    q_ww, q_wx, q_wy, q_wz = quat_T[0] * quat_T
    q_xx, q_xy, q_xz = quat_T[1] * quat_T[1:]
    q_yy, q_yz = quat_T[2] * quat_T[2:]
    q_zz = quat_T[3] * quat_T[3]

    out_T[0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
    out_T[1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
    out_T[2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

    out_T /= q_ww + q_xx + q_yy + q_zz

    return out_


@torch.jit.script
def _tc_transform_by_quat(v, quat, out: torch.Tensor | None = None):
    q_w, q_x, q_y, q_z = quat[..., :1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:]
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z**2

    vs = v / (q_ww + q_xx + q_yy + q_zz)
    v_x, v_y, v_z = vs[..., :1], vs[..., 1:2], vs[..., 2:]

    if out is None:
        out = torch.empty(vs.shape, dtype=vs.dtype, device=vs.device)
    u_x, u_y, u_z = out[..., :1], out[..., 1:2], out[..., 2:]

    u_x.copy_(v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy))
    u_y.copy_(v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx))
    u_z.copy_(v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz))

    return out


def transform_by_quat(v, quat):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert v.ndim >= 1 and quat.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (v, quat)):
        return _tc_transform_by_quat(v, quat)
    elif all(isinstance(e, np.ndarray) for e in (v, quat)):
        return _np_transform_by_quat(v, quat, out=None)
    else:
        gs.raise_exception(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")


def axis_angle_to_quat(angle, axis):
    pass


def transform_by_xyz(pos, xyz):
    pass


def transform_by_trans_quat(pos, trans, quat):
    return transform_by_quat(pos, quat) + trans


def inv_transform_by_quat(pos, quat):
    return transform_by_quat(pos, inv_quat(quat))


def inv_transform_by_trans_quat(pos, trans, quat):
    return inv_transform_by_quat(pos - trans, quat)


def transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + transform_by_quat(pos, t_quat)
    new_quat = transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat


def inv_transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    t_quat_inv = inv_quat(t_quat)
    new_pos = transform_by_quat(pos - t_trans, t_quat_inv)
    new_quat = transform_quat_by_quat(quat, t_quat_inv)
    return new_pos, new_quat


def transform_by_R(pos, R):
    """
    Transforms 3D points by a 3x3 rotation matrix or a batch of matrices, supporting both NumPy arrays and PyTorch
    tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a batched
        batch of points (B, N, 3).
    R: np.ndarray | torch.Tensor
        The 3x3 rotation matrix or a batch of B rotation matrices of shape (B, 3, 3). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3 and R.shape[-2:] == (3, 3)
    assert R.ndim - pos.ndim in (0, 1)

    B = R.shape[:-2]
    N = pos.shape[-2] if pos.ndim == R.ndim else 1
    R = R.reshape((-1, 3, 3))
    pos_ = pos.reshape((math.prod(B), N, 3))

    if all(isinstance(e, torch.Tensor) for e in (pos, R) if e is not None):
        new_pos = torch.bmm(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    elif all(isinstance(e, np.ndarray) for e in (pos, R) if e is not None):
        new_pos = np.matmul(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(pos)=} and {type(R)=}")

    new_pos = new_pos.reshape(pos.shape)

    return new_pos


def transform_by_trans_R(pos, trans, R):
    assert trans.shape[:-1] == R.shape[:-2]
    if trans.ndim < pos.ndim:
        trans = trans[..., None, :]
    return transform_by_R(pos, R) + trans


def transform_by_T(pos, T):
    """
    Transforms 3D points by a 4x4 transformation matrix or a batch of matrices, supporting both NumPy arrays and
    PyTorch tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a
        batched batch of points (B, N, 3).
    T: np.ndarray | torch.Tensor
        The 4x4 transformation matrix or a batch of B transformation matrices of shape (B, 4, 4). Must be of the
        same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    return transform_by_trans_R(pos, T[..., :3, 3], T[..., :3, :3])


def inv_transform_by_T(pos, T):
    pass


def _tc_polar(A: torch.Tensor, pure_rotation: bool, side: Literal["right", "left"], eps: float):
    """Torch implementation of polar decomposition with batched support."""
    pass


@nb.jit(nopython=True, cache=True)
def _np_polar_core_single(A, pure_rotation: bool, side_int: int):
    """
    Numba-accelerated core computation for polar decomposition of a single matrix.

    Parameters
    ----------
    A : np.ndarray
        The matrix to decompose. Must be a 2D matrix (M, N).
    pure_rotation : bool
        If True, ensure the unitary matrix U has det(U) = 1 (pure rotation).
    side_int : int
        0 for "right", 1 for "left".

    Returns
    -------
    U : np.ndarray
        Unitary matrix.
    P : np.ndarray
        Positive semi-definite matrix.
    """
    pass


@nb.jit(nopython=True, cache=True)
def _np_polar_core_batched(A, pure_rotation: bool, side_int: int, U_out, P_out):
    """
    Numba-accelerated core computation for batched polar decomposition.

    Parameters
    ----------
    A : np.ndarray
        The batched matrices to decompose. Shape (*batch, M, N).
    pure_rotation : bool
        If True, ensure the unitary matrix U has det(U) = 1 (pure rotation).
    side_int : int
        0 for "right", 1 for "left".
    U_out : np.ndarray
        Output array for U, shape (*batch, M, N).
    P_out : np.ndarray
        Output array for P, shape (*batch, N, N) or (*batch, M, M) depending on side.

    Returns
    -------
    None (results written to U_out and P_out)
    """
    pass


def _np_polar(A: np.ndarray, pure_rotation: bool, side: Literal["right", "left"]):
    """Numpy implementation of polar decomposition with numba acceleration and batched support."""
    pass


def polar(A, pure_rotation: bool = True, side: Literal["right", "left"] = "right"):
    """
    Compute the polar decomposition of a matrix or batch of matrices.

    Parameters
    ----------
    A : np.ndarray | torch.Tensor
        The matrix or batch of matrices to decompose. Can be:
        - Single matrix: shape (M, N)
        - Batched: shape (*batch, M, N)
    pure_rotation : bool, optional
        If True, ensure the unitary matrix U has det(U) = 1 (pure rotation).
        If False, U may have det(U) = -1 (contains reflection). Default is True.
    side : Literal['right', 'left'], optional
        The side of the decomposition. Either 'right' or 'left'. Default is 'right'.

    Returns
    -------
    tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
        A tuple of (U, P) where:
        - U : The unitary matrix (rotation part), same shape as A (M, N) or (*batch, M, N).
        - P : The positive semi-definite matrix (scaling part). For 'right' decomposition,
          P has shape (N, N) or (*batch, N, N). For 'left' decomposition, P has shape (M, M) or (*batch, M, M).
    """
    pass


@nb.jit(nopython=True, cache=True)
def _np_slerp(q0, q1, t):
    pass


@torch.jit.script
def _tc_slerp(q0, q1, t, eps: float):
    pass


def slerp(q0, q1, t):
    """
    Perform spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q0 : numpy.array | torch.Tensor
        The start quaternion (w, x, y, z), can be batched.
    q1 : numpy.array | torch.Tensor
        The end quaternion (w, x, y, z), can be batched.
    t : numpy.array | torch.Tensor
        The interpolation parameter between 0 and 1.

    Returns
    -------
    numpy.array | torch.Tensor
        The interpolated quaternion (w, x, y, z).
    """
    pass


# ------------------------------------------------------------------------------------
# ------------------------------------- numpy ----------------------------------------
# ------------------------------------------------------------------------------------


def scale_to_T(scale):
    T = np.eye(4, dtype=scale.dtype)
    T[[0, 1, 2], [0, 1, 2]] = scale
    return T


@nb.jit(nopython=True, cache=True)
def _np_z_up_to_R(z, up=None, out=None):
    B = z.shape[:-1]
    if out is None:
        out_ = np.empty((*B, 3, 3), dtype=z.dtype)
    else:
        assert out.shape == (*B, 3, 3)
        out_ = out

    z_norm = np.sqrt(np.sum(np.square(z.reshape((-1, 3))), -1)).reshape(B)

    out_[..., 2] = z
    for i in np.ndindex(B):
        z_norm_i = z_norm[i]
        R = out_[i]
        x, y, z = R.T

        if z_norm_i > gs.EPS:
            z /= z_norm_i
        else:
            if up is None or abs(up[i][1]) < 0.5:
                z[:] = 0.0, 1.0, 0.0
            else:
                z[:] = 0.0, 0.0, 1.0

        if up is not None:
            x[:] = np.cross(up[i], z)
        else:
            if abs(z[2]) < 1.0 - gs.EPS:
                # up = (0.0, 0.0, 1.0)
                x[0] = z[1]
                x[1] = -z[0]
                x[2] = 0.0
            else:
                # up = (0.0, 1.0, 0.0)
                x[0] = z[2]
                x[1] = 0.0
                x[2] = -z[0]

        x_norm = np.linalg.norm(x)
        if x_norm > gs.EPS:
            x /= x_norm
            y[:] = np.cross(z, x)
        else:
            # This would only occurs if the user specified non-zero colinear z and up
            R[:] = np.eye(3, dtype=R.dtype)

    return out_


@torch.jit.script
def _tc_z_up_to_R(z, eps: float, up=None, out: torch.Tensor | None = None):
    if out is None:
        R = torch.empty(z.shape[:-1] + (3, 3), dtype=z.dtype, device=z.device)
    else:
        # assert out.shape == z.shape[:-1] + (3, 3)
        R = out

    # Set z as the third column of rotation matrix
    R[..., 2] = z

    # Handle batch dimension properly
    x, y, z = R[..., 0], R[..., 1], R[..., 2]

    # Normalize z vectors
    z_norm = torch.linalg.vector_norm(z, ord=2, dim=-1, keepdim=True)
    z /= z_norm.clamp(min=eps)

    # Handle zero norm cases
    zero_mask = z_norm < eps
    if up is None:
        torch.where(zero_mask, torch.tensor((0.0, 1.0, 0.0), device=z.device, dtype=z.dtype), z, out=z)
    else:
        up_mask = up[..., 1:2].abs() < 0.5
        torch.where(zero_mask & up_mask, torch.tensor((0.0, 1.0, 0.0), device=z.device, dtype=z.dtype), z, out=z)
        torch.where(zero_mask & ~up_mask, torch.tensor((0.0, 0.0, 1.0), device=z.device, dtype=z.dtype), z, out=z)

    # Compute x vectors (first column)
    if up is not None:
        x[:] = torch.cross(torch.broadcast_to(up, z.shape), z, dim=-1)
    else:
        up_mask = z[..., 2:].abs() < 1.0 - eps
        torch.where(up_mask, z[..., 1], z[..., 2], out=x[..., 0])
        torch.where(up_mask, -z[..., 0], 0.0, out=x[..., 1])
        torch.where(up_mask, 0.0, -z[..., 0], out=x[..., 2])

    # Normalize x vectors
    x_norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    x /= x_norm.clamp(min=eps)

    # Handle zero x norm cases
    zero_x_mask = x_norm < eps
    # For zero x norm, set identity matrix
    torch.where(zero_x_mask[..., None], torch.eye(3, device=z.device, dtype=z.dtype), R, out=R)
    # Continue with non-zero cases
    torch.where(~zero_x_mask, torch.cross(z, x, dim=-1), y, out=y)

    return R


def z_up_to_R(z, up=None, out=None):
    if isinstance(z, torch.Tensor):
        return _tc_z_up_to_R(z, gs.EPS, up, out)
    else:
        return _np_z_up_to_R(z, up, out)


def pos_lookat_up_to_T(pos, lookat, up):
    if all(isinstance(e, torch.Tensor) for e in (pos, lookat, up) if e is not None):
        T = torch.zeros((*pos.shape[:-1], 4, 4), dtype=pos.dtype, device=pos.device)
        T[..., 3, 3] = 1.0
        T[..., :3, 3] = pos

        z = pos - lookat
        z_up_to_R(z, up=up, out=T[..., :3, :3])
        return T
    elif all(isinstance(e, np.ndarray) for e in (pos, lookat, up) if e is not None):
        T = np.zeros((*pos.shape[:-1], 4, 4), dtype=pos.dtype)
        T[..., 3, 3] = 1.0
        T[..., :3, 3] = pos

        z = pos - lookat
        z_up_to_R(z, up=up, out=T[..., :3, :3])
        return T
    else:
        gs.raise_exception(
            f"all of the inputs must be torch.Tensor or np.ndarray. got: {type(pos)=}, {type(lookat)=}, {type(up)=}"
        )


def T_to_pos_lookat_up(T):
    pos = T[..., :3, 3]
    lookat = T[..., :3, 3] - T[..., :3, 2]
    up = T[..., :3, 1]
    return pos, lookat, up


def euler_to_quat(euler_xyz):
    return xyz_to_quat(np.asarray(euler_xyz), rpy=True, degrees=True)


@nb.jit(nopython=True, cache=True)
def _np_euler_to_R(rpy: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose last dimension gathers the 3 Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert rpy.ndim >= 1
    if out is None:
        out_ = np.empty((*rpy.shape[:-1], 3, 3), dtype=rpy.dtype)
    else:
        assert out.shape == (*rpy.shape[:-1], 3, 3)
        out_ = out

    cos_rpy, sin_rpy = np.cos(rpy), np.sin(rpy)
    cos_roll, cos_pitch, cos_yaw = cos_rpy[..., 0], cos_rpy[..., 1], cos_rpy[..., 2]
    sin_roll, sin_pitch, sin_yaw = sin_rpy[..., 0], sin_rpy[..., 1], sin_rpy[..., 2]

    out_[..., 0, 0] = cos_pitch * cos_yaw
    out_[..., 0, 1] = -cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw
    out_[..., 0, 2] = sin_roll * sin_yaw + cos_roll * sin_pitch * cos_yaw
    out_[..., 1, 0] = cos_pitch * sin_yaw
    out_[..., 1, 1] = cos_roll * cos_yaw + sin_roll * sin_pitch * sin_yaw
    out_[..., 1, 2] = -sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw
    out_[..., 2, 0] = -sin_pitch
    out_[..., 2, 1] = sin_roll * cos_pitch
    out_[..., 2, 2] = cos_roll * cos_pitch

    return out_


def euler_to_R(euler_xyz):
    return _np_euler_to_R(np.deg2rad(euler_xyz))


@nb.jit(nopython=True, cache=True)
def quat_to_rotvec(quat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the angle-axis representation of a single or a batch of quaternions (qw, qx, qy, qz).

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert quat.ndim >= 1
    B = quat.shape[:-1]
    if out is None:
        out_ = np.empty((*B, 3), dtype=quat.dtype)
    else:
        assert out.shape == (*B, 3)
        out_ = out

    # Split real (qw,) and imaginary (qx, qy, qz) quaternion parts
    q_w, q_vec = quat[..., :1], np.ascontiguousarray(quat[..., 1:])

    # Compute the angle-axis representation of the relative rotation
    s2 = np.sqrt(np.sum(np.square(q_vec.reshape((-1, 3))), -1)).reshape((*B, 1))
    angle = 2.0 * np.arctan2(s2, np.abs(q_w))
    # FIXME: Ideally, a taylor expansion should be used to handle angle ~ 0.0
    inv_sinc = angle / np.maximum(s2, gs.EPS)
    out_[:] = np.where(q_w < 0.0, -1.0, 1.0) * inv_sinc * q_vec

    if out is None:
        return out_
    return None


@nb.jit(nopython=True, cache=True)
def rotvec_to_quat(rotvec: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the quaternion representation (qw, qx, qy, qz) of a single or a batch of angle-axis vectors.

    :param rotvec: N-dimensional array whose last dimension gathers the 3 angle-axis components
                   angle * (ax, ay, az).
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    pass


@nb.jit(nopython=True, cache=True)
def _np_axis_cos_angle_to_R(axis: np.ndarray, cos_theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    pass


def axis_angle_to_R(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    pass


def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    pass


@nb.jit(nopython=True, cache=True)
def R_to_rotvec(R: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the angle-axis representation of a single or a batch of 3D rotation matrices.

    :param R: N-dimensional array whose last 2 dimensions gathers individual 3D rotation matrices.
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    # The quaternion path (Shepperd + arctan2) is preferred over direct Rodrigues formula theta from arccos of trace)
    # because the latter is catastrophically imprecise for near-identity rotations in float32 (the trace is ~3.0 and
    # arccos amplifies rounding).
    return quat_to_rotvec(_np_R_to_quat(R), out=out)


@nb.jit(nopython=True, cache=True)
def z_to_R(v_a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the "smallest" rotation transforming vector 'v_a' in 'e_z' as a rotation matrix.

    This operation is computed by rotating the world frame by moving the original z-axis to the given vector via the
    shortest path.
    """
    pass


@nb.jit(nopython=True, cache=True)
def z_to_quat(v_a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the "smallest" rotation transforming vector 'v_a' in 'e_z' as a quaternion (q_w, q_x, q_y, q_z).

    This method is (surprisingly) slower than `z_up_to_R`, w/ and w/o chaining `transform_by_(quat|R)`.
    """
    pass


def compute_camera_angle(camera_pos, camera_lookat):
    pass


def transform_inertia_by_T(inertia_tensor, T, mass):
    """
    Transform the inertia tensor to the new reference frame.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Parallel axis theorem
    translation_inertia = mass * (np.dot(t, t) * np.eye(3) - np.outer(t, t))

    return R @ inertia_tensor @ R.T + translation_inertia


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    theta : torch.Tensor
        Horizontal angles in radians.
    phi : torch.Tensor
        Vertical angles in radians.

    Returns
    -------
    vectors : torch.Tensor
        Vectors in cartesian coordinates as tensor of shape (..., 3).
    """
    cos_phi = torch.cos(phi)

    x = torch.cos(theta) * cos_phi  # forward
    y = torch.sin(theta) * cos_phi  # left
    z = torch.sin(phi)  # up

    return torch.stack([x, y, z], dim=-1)


def random_quaternion(batch_size):
    # Generate three uniform random numbers for each quaternion in the batch
    u1, u2, u3 = np.random.rand(3, batch_size)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.stack((q1, q2, q3, q4), axis=1)


def generate_grid_points_on_plane(lo: Vec3FType, hi: Vec3FType, normal: Vec3FType, nx: int, ny: int) -> np.ndarray:
    """
    Build an nx-by-ny grid of points on the plane defined by the bounds and normal.

    Parameters
    ----------
    lo: array-like[float, float, float]
        Lower bound of the plane
    hi: array-like[float, float, float]
        Upper bound of the plane
    normal: array-like[float, float, float]
        Normal of the plane
    nx: int
        Number of grid points in x direction
    ny: int
        Number of grid points in y direction

    Returns
    -------
    grid: np.ndarray, shape (ny, nx, 3)
        Grid points on the plane
    """
    pass


# ------------------------------------------------------------------------------------
# ------------------------------------- misc ----------------------------------------
# ------------------------------------------------------------------------------------


def zero_pos():
    return np.zeros(3, dtype=gs.np_float)


def identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=gs.np_float)


def tc_zero_pos():
    pass


def tc_identity_quat():
    pass


def nowhere():
    # let's inject a bit of humor here
    return np.array([2333333, 6666666, 5201314], dtype=gs.np_float)


def default_solver_params():
    """
    Default constraint solver parameters ``(timeconst, dampratio, dmin, dmax, width, mid, power)``.

    The constraint reference acceleration is computed as ``aref = -b * vel - k * imp * pos``, where:

    - ``b = 2 / (dmax * timeconst)`` controls velocity damping (energy-dissipating). Depends only on ``timeconst``.
    - ``k = 1 / (dmax² * timeconst² * dampratio²)`` controls position stiffness (energy-conserving elastic spring).

    Despite its name, ``dampratio`` does **not** control damping — it scales the spring stiffness ``k``. Smaller
    ``dampratio`` yields a stiffer spring (more elastic bounce), while larger values yield a softer, more compliant
    contact. Setting ``dampratio`` to zero is forbidden as it removes the elastic restoring force entirely. Use a
    small positive value (e.g. ``0.01``) with a large ``timeconst`` (e.g. ``1.0``) for nearly undamped elastic contact.

    The impedance parameters ``(dmin, dmax, width, mid, power)`` control the depth-dependent constraint rigidity.

    Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    """
    return np.array([0.0, 1.0, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0])


def default_friction():
    return 1.0


def default_dofs_kp(n=6):
    return np.full((n,), fill_value=100.0, dtype=gs.np_float)


def default_dofs_kv(n=6):
    return np.full((n,), fill_value=10.0, dtype=gs.np_float)


@qd.data_oriented
class SpatialHasher:
    def __init__(self, cell_size, grid_res, n_slots=None):
        self.cell_size = cell_size
        self.grid_res = grid_res

        if n_slots is None:
            self.n_slots = np.prod(grid_res)
        else:
            self.n_slots = n_slots

    def build(self, n_batch):
        self._B = n_batch
        # number of elements in each slot
        self.slot_size = qd.field(gs.qd_int, shape=(self.n_slots, self._B))
        # element index offset in each slot
        self.slot_start = qd.field(gs.qd_int, shape=(self.n_slots, self._B))
        self.cur_cnt = qd.field(gs.qd_int, shape=(self._B,))

    @qd.func
    def compute_reordered_idx(self, n, pos, active, reordered_idx):
        """
        Reordered element idx based on the given positions and active flags.

        Parameters:
            n (int)       : The number of elements in the positions and active arrays.
            pos           : The array of positions.
            active        : The array of active flags.
            reordered_idx : The array to store the computed reordered indices.

        Returns:
            None
        """

        self.slot_size.fill(0)
        self.slot_start.fill(0)
        self.cur_cnt.fill(0)

        for i_n, i_b in qd.ndrange(n, self._B):
            if active[i_n, i_b]:
                slot_idx = self.pos_to_slot(pos[i_n, i_b])
                qd.atomic_add(self.slot_size[slot_idx, i_b], 1)

        for i_n in range(self.n_slots):
            for i_b in range(self._B):
                self.slot_start[i_n, i_b] = qd.atomic_add(self.cur_cnt[i_b], self.slot_size[i_n, i_b])

        for i_n, i_b in qd.ndrange(n, self._B):
            if active[i_n, i_b]:
                slot_idx = self.pos_to_slot(pos[i_n, i_b])
                reordered_idx[i_n, i_b] = qd.atomic_add(self.slot_start[slot_idx, i_b], 1)

        # recover slot_start
        for i_s, i_b in qd.ndrange(self.n_slots, self._B):
            self.slot_start[i_s, i_b] -= self.slot_size[i_s, i_b]

    @qd.func
    def for_all_neighbors(self, i_p, pos, task_range, ret: qd.template(), task: qd.template(), i_b):
        """
        Iterates over all neighbors of a given position and performs a task on each neighbor.
        Elements are considered neighbors if they are within task_range.

        Parameters:
            i_p (int)  : Index of the querying particle.
            pos        : Template for the positions of all particles.
            task       : Template for the task to be performed on each neighbor of the querying particle.
            task_range : Range within which the task should be performed.
            ret        : Template for the return value of the task.

        Returns:
            None
        """
        base = self.pos_to_grid(pos[i_p, i_b])
        for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
            slot_idx = self.grid_to_slot(base + offset)
            for j_p in range(
                self.slot_start[slot_idx, i_b], self.slot_size[slot_idx, i_b] + self.slot_start[slot_idx, i_b]
            ):
                if i_p != j_p and (pos[i_p, i_b] - pos[j_p, i_b]).norm() < task_range:
                    task(i_p, j_p, ret, i_b)

    @qd.func
    def pos_to_grid(self, pos):
        return qd.floor(pos / self.cell_size, gs.qd_int)

    @qd.func
    def grid_to_pos(self, grid_id):
        pass

    @qd.func
    def grid_to_slot(self, grid_id):
        return (
            grid_id[0] * self.grid_res[1] * self.grid_res[2] + grid_id[1] * self.grid_res[2] + grid_id[2]
        ) % self.n_slots

    @qd.func
    def pos_to_slot(self, pos):
        return self.grid_to_slot(self.pos_to_grid(pos))


@nb.jit(nopython=True, cache=True)
def cubic_spline_1d(x, y, xv):
    """
    Evaluate a 1D cubic spline at specified points.

    Constructs a cubic spline interpolation of the input data `(x, y)` and
    evaluates it at `xv`. The spline is C² continuous and uses **not-a-knot**
    boundary conditions, producing a smooth curve that passes through all data points.

    Parameters
    ----------
    x : array_like, shape (n,)
        Strictly increasing x-coordinates of the data points.
    y : array_like, shape (n,) or (n, m)
        Corresponding y-coordinates. Multiple columns can be provided for
        simultaneous interpolation of multiple datasets.
    xv : array_like
        Points at which to evaluate the spline.

    Returns
    -------
    yv : ndarray, shape (len(xv),) or (len(xv), m)
        Interpolated values at `xv`.
    """
    assert len(x) == len(y)
    y_2d = y[:, None] if y.ndim == 1 else y
    n, m = y_2d.shape
    h = np.diff(x)

    # Band storage: only store the non-zero diagonals
    # band[0] = 2nd lower diagonal (only element at (2, 0))
    # band[1] = 1st lower diagonal
    # band[2] = main diagonal
    # band[3] = 1st upper diagonal
    # band[4] = 2nd upper diagonal (only element at (n - 3, n - 1))
    band = np.zeros((5, n), dtype=gs.np_float)
    rhs = np.zeros((n, m), dtype=gs.np_float)

    # Not-a-knot boundary conditions
    band[2, 0], band[3, 0], band[4, 0] = h[1], -(h[0] + h[1]), h[0]
    band[0, n - 1], band[1, n - 1], band[2, n - 1] = h[-1], -(h[-2] + h[-1]), h[-2]

    # Interior points
    for i in range(1, n - 1):
        band[1, i], band[2, i], band[3, i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 3 * ((y_2d[i + 1] - y_2d[i]) / h[i] - (y_2d[i] - y_2d[i - 1]) / h[i - 1])

    # Specialized Gaussian elimination with band storage
    for k in range(n - 1):
        pivot = band[2, k]
        if k + 1 < n:
            factor = band[1, k + 1] / pivot
            band[2, k + 1] -= factor * band[3, k]
            if k + 2 < n:
                band[3, k + 1] -= factor * band[4, k]
            rhs[k + 1] -= factor * rhs[k]
        if k == n - 3:
            factor = band[0, n - 1] / pivot
            band[1, n - 1] -= factor * band[3, k]
            band[2, n - 1] -= factor * band[4, k]
            rhs[n - 1] -= factor * rhs[k]

    # Back substitution
    c = np.zeros_like(rhs)
    c[n - 1] = rhs[n - 1] / band[2, n - 1]
    if n > 1:
        c[n - 2] = (rhs[n - 2] - band[3, n - 2] * c[n - 1]) / band[2, n - 2]
    for i in range(n - 3, -1, -1):
        c[i] = rhs[i] - band[3, i] * c[i + 1]
        if i + 2 < n and band[4, i] != 0:
            c[i] -= band[4, i] * c[i + 2]
        c[i] /= band[2, i]

    # Solve for b and d
    b = (y_2d[1:] - y_2d[:-1]) / h[:, None] - h[:, None] * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h[:, None])

    # Evaluate spline at xv
    xv = np.atleast_1d(xv)
    ix = np.clip(np.searchsorted(x[1:], xv), 0, n - 2)
    dx = (xv - x[ix])[:, None]
    return y_2d[ix] + b[ix] * dx + c[ix] * dx**2 + d[ix] * dx**3


@nb.jit(nopython=True, cache=True)
def orthogonals(a) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns orthogonal vectors `b` and `c`, given a normal vector `a`.

    Note that `a` is assumed to be normalized.
    """
    B = a.shape[:-1]

    a_1d = a.reshape((-1, 3))
    a_x, a_y, a_z = a_1d[:, 0], a_1d[:, 1], a_1d[:, 2]

    b1 = np.stack((-a_x * a_y, 1.0 - a_y**2, -a_z * a_y), axis=-1)
    b2 = np.stack((-a_x * a_z, -a_y * a_z, 1.0 - a_z**2), axis=-1)
    b_1d = np.where(np.abs(a_y).reshape((-1, 1)) < 0.5, b1, b2)
    b_1d /= np.sqrt(np.sum(np.square(b_1d), -1)).reshape((-1, 1))

    return np.cross(b_1d, a_1d).reshape((*B, 3)), b_1d.reshape((*B, 3))
