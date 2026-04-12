from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Type

import numpy as np
from typing_extensions import override

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.mesh import create_cylinder, create_plane
from genesis.utils.misc import tensor_to_array
from genesis.utils.raycast import Ray, RayHit, plane_raycast
from genesis.vis.keybindings import MouseButton

from ..viewer_plugin import EVENT_HANDLE_STATE, EVENT_HANDLED, RaycasterViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


MIN_PICKABLE_MASS = 1e-3  # kg — links below this threshold are skipped to avoid numerical instability


def with_lock(fun: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fun)
    pass


class MouseInteractionPlugin(RaycasterViewerPlugin):
    """
    Basic interactive viewer plugin that enables using mouse to apply spring force on rigid entities.
    """

    def __init__(
        self,
        use_force: bool = True,
        spring_const: float = 1000.0,
        color: tuple[float, float, float, float] = (0.2, 0.8, 0.8, 0.6),
    ) -> None:
        super().__init__()
        self.use_force = bool(use_force)
        self.spring_const = float(spring_const)
        self.color = tuple(color)

        self._lock: Lock = Lock()
        self._held_link: "RigidLink | None" = None
        self._held_point_local: np.ndarray | None = None  # Held point in link-local frame
        self._mouse_drag_plane: tuple[np.ndarray, float] | None = None
        self._prev_mouse_screen_pos: tuple[int, int] = (0, 0)
        self._prev_mouse_scene_pos: np.ndarray | None = None
        self._surface_normal: np.ndarray | None = None
        self._plane_rotation_angle: float = 0.0

        # Persistent debug nodes (lifecycle managed in on_draw)
        self._sphere_node: "Node | None" = None
        self._line_node: "Node | None" = None
        self._plane_node: "Node | None" = None
        self._arrow_node: "Node | None" = None

        # Shared meshes (created in build)
        self._unit_cylinder_mesh = None
        self._plane_mesh = None

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)
        self._prev_mouse_screen_pos = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

        self._unit_cylinder_mesh = create_cylinder(radius=0.005, height=1.0, color=self.color)
        plane_color = (*self.color[:3], self.color[3] * 0.5)
        plane_vmesh, _ = create_plane(plane_size=(0.3, 0.3), color_or_texture=plane_color, double_sided=True)
        self._plane_mesh = plane_vmesh

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        pass

    @with_lock
    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    @with_lock
    @override
    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> EVENT_HANDLE_STATE:
        pass

    @with_lock
    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    @with_lock
    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    @with_lock
    @override
    def update_on_sim_step(self) -> None:
        super().update_on_sim_step()

        if self._held_link:
            mouse_ray: Ray = self._screen_position_to_ray(*self._prev_mouse_screen_pos)
            assert self._mouse_drag_plane is not None
            ray_hit: RayHit = plane_raycast(*self._mouse_drag_plane, mouse_ray)

            # If ray doesn't hit the plane, skip this update
            if ray_hit is None:
                return

            self._prev_mouse_scene_pos = ray_hit.position

            if self.use_force:
                self._apply_spring_force(ray_hit.position, self.scene.sim.dt)
            else:
                assert self._held_point_local is not None
                link_quat = tensor_to_array(self._held_link.get_quat())
                offset_world = gu.transform_by_quat(self._held_point_local, link_quat)
                self._held_link.entity.set_pos(ray_hit.position - offset_world)

    @with_lock
    @override
    def on_draw(self) -> None:
        pass

    def _compute_line_T(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Compute transform for unit cylinder (height=1, centered at z=0) from start to end."""
        pass

    def _update_drag_plane(self) -> None:
        """Update the drag plane based on surface normal and rotation angle."""
        pass

    def _apply_spring_force(self, control_point: np.ndarray, dt: float) -> None:
        if not self._held_link:
            return

        # Get current link state
        link_pos = tensor_to_array(self._held_link.get_pos())
        link_quat = tensor_to_array(self._held_link.get_quat())
        lin_vel = tensor_to_array(self._held_link.get_vel())
        ang_vel = tensor_to_array(self._held_link.get_ang())

        # Compute current world position of held point
        held_point_world = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)

        # Compute inertial frame properties
        inertial_pos = tensor_to_array(self._held_link.inertial_pos)
        inertial_quat = tensor_to_array(self._held_link.inertial_quat)
        world_principal_quat = gu.transform_quat_by_quat(inertial_quat, link_quat)

        # Compute arm from COM to held point in world frame
        arm_in_principal = gu.inv_transform_by_trans_quat(self._held_point_local, inertial_pos, inertial_quat)
        arm_in_world = gu.transform_by_quat(arm_in_principal, world_principal_quat)

        # Compute inverse inertia in world frame
        R_world = gu.quat_to_R(world_principal_quat)
        inertia_world = R_world @ self._held_link.inertial_i @ R_world.T
        inv_inertia_world = np.linalg.inv(inertia_world)

        pos_err_v = control_point - held_point_world
        inv_mass = float(1.0 / self._held_link.get_mass())

        total_impulse = np.zeros(3, dtype=gs.np_float)
        total_torque_impulse = np.zeros(3, dtype=gs.np_float)

        # Approximate spring-damper in each axis
        for i in range(3):
            body_point_vel = lin_vel + np.cross(ang_vel, arm_in_world)
            vel_err_v = -body_point_vel

            direction = np.zeros(3, dtype=gs.np_float)
            direction[i % 3] = 1.0

            pos_err = np.dot(direction, pos_err_v)
            vel_err = np.dot(direction, vel_err_v)

            # Compute virtual mass (effective inertia for this constraint direction)
            arm_x_dir = np.cross(arm_in_world, direction)
            rot_mass = np.dot(arm_x_dir, inv_inertia_world @ arm_x_dir)
            virtual_mass = 1.0 / (inv_mass + rot_mass + gs.EPS)

            # Critical damping
            damping_coeff = 2.0 * np.sqrt(self.spring_const * virtual_mass)
            # Impulse: J = F*dt = k*x*dt + c*v*dt
            impulse = (self.spring_const * pos_err + damping_coeff * vel_err) * dt

            lin_vel += direction * impulse * inv_mass
            ang_vel += inv_inertia_world @ (arm_x_dir * impulse)

            total_impulse[i % 3] += impulse
            total_torque_impulse += arm_x_dir * impulse

        # Apply the new force
        self._held_link.solver.apply_links_external_force(
            total_impulse / dt, (self._held_link.idx,), ref="link_com", local=False
        )
        self._held_link.solver.apply_links_external_torque(
            total_torque_impulse / dt, (self._held_link.idx,), ref="link_com", local=False
        )
