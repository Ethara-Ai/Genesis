import os
import pickle as pkl
from itertools import chain
from typing import TYPE_CHECKING

import igl
import numpy as np
import skimage
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array, qd_to_torch, DeprecationError

if TYPE_CHECKING:
    from genesis.engine.materials.rigid import Rigid as RigidMaterial
    from genesis.engine.mesh import Mesh
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    from .rigid_entity import RigidEntity
    from .rigid_link import RigidLink


NUM_VERTS_VISUAL_GEOM_AABB = 200


class RigidGeom(RBC):
    """
    A `RigidGeom` is the basic building block of a `RigidEntity` for collision checking. It is usually constructed from a single mesh. This can be accessed via `link.geoms`.
    """

    def __init__(
        self,
        link: "RigidLink",
        idx,
        cell_start: int,
        vert_start: int,
        face_start: int,
        edge_start: int,
        verts_state_start: int,
        mesh: "Mesh",
        type: gs.GEOM_TYPE,
        friction: float,
        sol_params,
        init_pos,
        init_quat,
        needs_coup: bool,
        contype,
        conaffinity,
        center_init=None,
        data=None,
    ):
        self._link: "RigidLink" = link
        self._entity: "RigidEntity" = link.entity
        self._material: "RigidMaterial" = link.entity.material
        self._solver: "RigidSolver" = link.entity.solver
        self._mesh: "Mesh" = mesh

        self._uid = gs.UID()
        self._idx = idx
        self._type: gs.GEOM_TYPE = type
        self._friction: float = friction
        self._sol_params = sol_params
        self._needs_coup: bool = needs_coup
        self._contype = int(contype)
        self._conaffinity = int(conaffinity)
        self._is_convex: bool = mesh.is_convex
        self._cell_start: int = cell_start
        self._vert_start: int = vert_start
        self._face_start: int = face_start
        self._edge_start: int = edge_start
        self._verts_state_start: int = verts_state_start

        self._coup_softness: float = self._material.coup_softness
        self._coup_friction: float = self._material.coup_friction
        self._coup_restitution: float = self._material.coup_restitution

        self._init_pos: np.ndarray = init_pos
        self._init_quat: np.ndarray = init_quat

        # For heterogeneous simulation: which environments this geom is active in (None = all envs)
        self.active_envs_mask: torch.Tensor | None = None
        self.active_envs_idx: np.ndarray | None = None

        self._init_verts = mesh.verts
        self._init_faces = mesh.faces
        self._init_edges = mesh.get_unique_edges()
        self._init_normals = mesh.normals
        self._uvs = mesh.uvs
        self._surface = mesh.surface
        self._metadata = mesh.metadata

        if center_init is None:
            self._init_center_pos = np.repeat(
                self._init_verts.mean(0, keepdims=True), repeats=self._init_verts.shape[0], axis=0
            )
        else:
            self._init_center_pos = np.array(center_init)
        self._data = np.zeros([7])
        if data is not None:
            self._data[: len(data)] = data

        # verts and faces for sdf genertaion
        if "sdf_mesh" in self._metadata:
            self._sdf_verts = np.ascontiguousarray(self._metadata["sdf_mesh"].vertices)
            self._sdf_faces = np.ascontiguousarray(self._metadata["sdf_mesh"].faces)
        else:
            self._sdf_verts = np.array(self._init_verts)
            self._sdf_faces = np.array(self._init_faces)

        if len(self._sdf_faces) > 50000:
            mesh_descr = f"({mesh.metadata['mesh_path']})" if "mesh_path" in mesh.metadata else ""
            gs.logger.warning(
                f"Beware that SDF pre-processing of mesh {mesh_descr} having more than 50000 vertices may take a very "
                "long time (>10min) and require large RAM allocation (>20Gb). Please either enable convexify or "
                "decimation. (see FileMorph options)"
            )

        # Compute adjacency graph
        tmesh = trimesh.Trimesh(vertices=self._init_verts, faces=self._init_faces, process=False)
        all_vert_neighbors_list = tmesh.vertex_neighbors
        assert self.n_verts == len(all_vert_neighbors_list)
        self.vert_neighbors = np.array(tuple(chain.from_iterable(all_vert_neighbors_list)), dtype=gs.np_int)
        self.vert_n_neighbors = np.array(tuple(map(len, all_vert_neighbors_list)), dtype=gs.np_int)
        self.vert_neighbor_start = np.array((0, *np.cumsum(self.vert_n_neighbors)[:-1]), dtype=gs.np_int)

        # NOTE: sdf size is from the center of the lower voxel cell to the center of the upper voxel cell
        # add padding. Adjust the cell size to keep resolution within bounds.
        padding_ratio = 0.2
        lower = self._init_verts.min(axis=0)
        upper = self._init_verts.max(axis=0)
        grid_size = (upper - lower).max() * padding_ratio + (upper - lower)
        self._sdf_cell_size = gs.EPS + np.clip(
            self._material.sdf_cell_size,
            grid_size.max() / (self._material.sdf_max_res - 1),
            grid_size.min() / max(self._material.sdf_min_res - 1, 2),
        )
        self._sdf_res = np.ceil(grid_size / self._sdf_cell_size).astype(gs.np_int) + 1
        self._sdf_grad_delta = 0.0 if self.type == gs.GEOM_TYPE.TERRAIN else self._sdf_cell_size * 1e-2
        self._is_preprocessed = False

    def _build(self):
        pass

    def _preprocess(self):
        # compute file name via hashing for caching
        pass

    def _compute_sd(self, query_points):
        pass

    def _compute_closest_verts(self, query_points):
        pass

    def _compute_sd_grad(self, query_points, delta=5e-4):
        ######## sdf gradient via finite differencing ########
        pass

    def get_trimesh(self):
        """
        Get the geom's trimesh object.
        """
        return self._mesh.trimesh

    def get_sdf_trimesh(self, color=[1.0, 1.0, 0.6, 1.0]):
        """
        Reconstruct trimesh object from sdf.
        """
        if self.sdf_val.min() >= 0:
            gs.logger.warning("SDF is positive everywhere. Returning empty mesh.")
            return trimesh.Trimesh(vertices=[], faces=[])
        else:
            vertices, faces, _, _ = skimage.measure.marching_cubes(self.sdf_val, level=0)
            vertices = (np.linalg.inv(self.T_mesh_to_sdf) @ np.hstack([vertices, np.ones([len(vertices), 1])]).T)[:3].T
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(np.array([color]), [len(vertices), 1]))
            return mesh

    def visualize_sdf(
        self,
        pos=None,
        T=None,
        color=(1.0, 1.0, 0.3, 1.0),
        show_axis=False,
        axis_color=(1.0, 0.0, 0.0, 1.0),
        axis_length=0.3,
        show_boundary=False,
        boundary_color=(0.0, 1.0, 0.0, 0.2),
    ):
        """
        Visualizes the signed distance field (SDF) of the rigid geometry in the viewer.
        """
        pass

    def set_friction(self, friction):
        """
        Set the friction coefficient of this geometry.
        """
        pass

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None):
        """
        Get the position of the geom in world frame.
        """
        tensor = qd_to_torch(self._solver.geoms_state.pos, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Get the quaternion of the geom in world frame.
        """
        tensor = qd_to_torch(self._solver.geoms_state.quat, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_verts(self):
        """
        Get the vertices of the geom in world frame.
        """
        self._solver.update_verts_for_geoms(self._idx)

        verts_idx = slice(self.verts_state_start, self.verts_state_end)
        if self.is_fixed and not self._entity._batch_fixed_verts:
            tensor = qd_to_torch(self._solver.fixed_verts_state.pos, verts_idx, copy=True)
        else:
            tensor = qd_to_torch(self._solver.free_verts_state.pos, None, verts_idx, transpose=True, copy=True)
            if self._solver.n_envs == 0:
                tensor = tensor[0]
        return tensor

    @gs.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the geom in world frame.
        """
        verts = self.get_verts()
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    def set_sol_params(self, sol_params):
        """
        Set the solver parameters of this geometry.
        """
        pass

    @property
    def sol_params(self):
        """
        Get the solver parameters of this geometry.
        """
        pass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Get the unique ID of the geom.
        """
        pass

    @property
    def idx(self) -> int:
        """
        Get the global index of the geom in RigidSolver.
        """
        pass

    @property
    def type(self) -> gs.GEOM_TYPE:
        """
        Get the type of the geom.
        """
        pass

    @property
    def friction(self):
        """
        Get the friction coefficient of the geom.
        """
        pass

    @property
    def data(self):
        """
        Get the additional data of the geom.
        """
        pass

    @property
    def metadata(self):
        """
        Get the metadata of the geom.
        """
        pass

    @property
    def link(self) -> "RigidLink":
        """
        Get the link that the geom belongs to.
        """
        pass

    @property
    def entity(self) -> "RigidEntity":
        """
        Get the entity that the geom belongs to.
        """
        pass

    @property
    def solver(self) -> "RigidSolver":
        """
        Get the solver that the geom belongs to.s
        """
        pass

    @property
    def is_convex(self) -> bool:
        """
        Get whether the geom is convex.
        """
        pass

    @property
    def mesh(self) -> "Mesh":
        pass

    @property
    def needs_coup(self) -> bool:
        """
        Get whether the geom needs coupling with other non-rigid entities.
        """
        pass

    @property
    def contype(self) -> int:
        """
        Get the contact type of the geometry for collision pair filtering.

        The two geoms are deemed "compatible" (i.e. collisions between them is allowed) if the 'contype' of one geom
        and the 'conaffinity' of the other geom have a common bit set to 1, i.e.
        `(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity) == True`. This is a powerful
        mechanism borrowed from Open Dynamics Engine.
        """
        pass

    @property
    def conaffinity(self) -> int:
        """
        Get the contact affinity of the geometry for collision pair filtering.

        See `contype` documentation for details.
        """
        pass

    @property
    def coup_softness(self) -> float:
        """
        Get the softness coefficient of the geom for coupling.
        """
        pass

    @property
    def coup_friction(self) -> float:
        """
        Get the friction coefficient of the geom for coupling.
        """
        pass

    @property
    def coup_restitution(self) -> float:
        """
        Get the restitution coefficient of the geom for coupling.
        """
        pass

    @property
    def init_pos(self) -> np.ndarray:
        """
        Get the initial position of the geom.
        """
        pass

    @property
    def init_quat(self) -> np.ndarray:
        """
        Get the initial quaternion of the geom.
        """
        pass

    @property
    def init_verts(self):
        """
        Get the initial vertices of the geom.
        """
        pass

    @property
    def init_faces(self):
        """
        Get the initial faces of the geom.
        """
        pass

    @property
    def init_edges(self):
        """
        Get the initial edges of the geom.
        """
        pass

    @property
    def init_normals(self):
        """
        Get the initial normals of the geom.
        """
        pass

    @property
    def init_center_pos(self):
        """
        Get the initial center position of the geom.
        """
        pass

    @property
    def uvs(self):
        """
        Get the UV coordinates of the geom.
        """
        pass

    @property
    def surface(self):
        """
        Get the surface object of the geom.
        """
        pass

    @property
    def gsd_path(self):
        """
        Get the path to the preprocessed `.gsd` file.
        """
        pass

    @property
    def sdf_res(self):
        """
        Get the resolution of the geom's signed distance field (SDF).
        """
        pass

    @property
    def sdf_val(self):
        """
        Get the signed distance field (SDF) of the geom.
        """
        pass

    @property
    def sdf_val_flattened(self):
        """
        Get the flattened signed distance field (SDF) of the geom.
        """
        pass

    @property
    def sdf_grad(self):
        """
        Get the gradient of the geom's signed distance field (SDF).
        """
        pass

    @property
    def sdf_grad_flattened(self):
        """
        Get the flattened gradient of the geom's signed distance field (SDF).
        """
        pass

    @property
    def sdf_max(self):
        """
        Get the maximum value of the geom's signed distance field (SDF).
        """
        pass

    @property
    def sdf_cell_size(self):
        """
        Get the cell size of the geom's signed distance field (SDF).
        """
        pass

    @property
    def sdf_grad_delta(self):
        """
        Get the delta value for computing the gradient of the geom's signed distance field (SDF).
        """
        pass

    @property
    def sdf_closest_vert(self):
        """
        Get the closest vertex of each cell of the geom's signed distance field (SDF).
        """
        pass

    @property
    def sdf_closest_vert_flattened(self):
        """
        Get the flattened closest vertex of each cell of the geom's signed distance field (SDF).
        """
        pass

    @property
    def T_mesh_to_sdf(self):
        """
        Get the transformation matrix of the geom's mesh frame w.r.t its signed distance field (SDF) frame.
        """
        pass

    @property
    def n_cells(self):
        """
        Number of cells in the geom's signed distance field (SDF).
        """
        pass

    @property
    def n_verts(self) -> int:
        """
        Number of vertices of the geom.
        """
        pass

    @property
    def n_faces(self) -> int:
        """
        Number of faces of the geom.
        """
        pass

    @property
    def n_edges(self):
        """
        Number of edges of the geom.
        """
        pass

    @property
    def cell_start(self):
        """
        Get the starting index of the cells of the signed distance field (SDF) in the rigid solver.
        """
        pass

    @property
    def vert_start(self):
        """
        Get the starting index of the geom's vertices in the rigid solver.
        """
        pass

    @property
    def face_start(self):
        """
        Get the starting index of the geom's faces in the rigid solver.
        """
        pass

    @property
    def edge_start(self):
        """
        Get the starting index of the geom's edges in the rigid solver.
        """
        pass

    @property
    def verts_state_start(self):
        """
        Get the starting index of the geom's vertices in the rigid solver.
        """
        pass

    @property
    def cell_end(self):
        """
        Get the ending index of the cells of the signed distance field (SDF) in the rigid solver.
        """
        pass

    @property
    def vert_end(self):
        """
        Get the ending index of the geom's vertices in the rigid solver.
        """
        pass

    @property
    def verts_state_end(self):
        """
        Get the ending index of the geom's vertices in the rigid solver.
        """
        pass

    @property
    def face_end(self):
        """
        Get the ending index of the geom's faces in the rigid solver.
        """
        pass

    @property
    def edge_end(self):
        """
        Get the ending index of the geom's edges in the rigid solver.
        """
        pass

    @property
    def is_built(self):
        """
        Whether the rigid entity the geom belongs to is built.
        """
        pass

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")

    @property
    def is_fixed(self) -> bool:
        """
        Whether this geom is fixed in the world.
        """
        pass

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        pass


class RigidVisGeom(RBC):
    """
    A `RigidVisGeom` is a counterpart of `RigidGeom`, but for visualization purposes. This can be accessed via `link.vis_geoms`.
    """

    def __init__(
        self,
        link,
        idx,
        vvert_start,
        vface_start,
        vmesh,
        init_pos,
        init_quat,
    ):
        self._link = link
        self._entity = link.entity
        self._material = link.entity.material
        self._solver = link.entity.solver
        self._vmesh = vmesh

        # Lazy-initialize low-res geometry because it is usually unused and may be slow to compute
        self._init_pos_tc = torch.from_numpy(init_pos).to(device=gs.device, dtype=gs.tc_float)
        self._init_quat_tc = torch.from_numpy(init_quat).to(device=gs.device, dtype=gs.tc_float)
        self._aabb_verts: torch.Tensor | None = None

        self._uid = gs.UID()
        self._idx = idx

        self._vvert_start = vvert_start
        self._vface_start = vface_start

        self._init_pos: np.ndarray = init_pos
        self._init_quat: np.ndarray = init_quat

        # For heterogeneous simulation: which environments this vgeom is active in (None = all envs)
        self.active_envs_mask: torch.Tensor | None = None
        self.active_envs_idx: np.ndarray | None = None

        self._init_vverts = vmesh.verts
        self._init_vfaces = vmesh.faces
        self._init_vnormals = vmesh.normals
        self._uvs = vmesh.uvs
        self._surface = vmesh.surface
        self._metadata = vmesh.metadata
        self._color = vmesh._color

    def _build(self):
        pass

    def get_trimesh(self):
        """
        Get trimesh object.
        """
        return self._vmesh.trimesh

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None):
        """
        Get the position of the geom in world frame.
        """
        tensor = qd_to_torch(self._solver.vgeoms_state.pos, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Get the quaternion of the geom in world frame.
        """
        tensor = qd_to_torch(self._solver.vgeoms_state.quat, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the geom in world frame.

        This method computes the bounding box of the geometry after aggressive decimation of its convex hull. This is
        usually sufficiently accurate (<1mm), while significantly improving runtime speed and reducing memory footprint.
        """
        if self._aabb_verts is None:
            # Aggressiveness has been tuned to give sub-millimeter accuracy on Franka robot in random configurations
            aabb_mesh = self.vmesh.copy()
            aabb_mesh.convexify()
            aabb_mesh.decimate(decimate_face_num=NUM_VERTS_VISUAL_GEOM_AABB, decimate_aggressiveness=3)
            self._aabb_verts = torch.from_numpy(aabb_mesh.verts).to(dtype=gs.tc_float, device=gs.device)

        pos, quat = gu.transform_pos_quat_by_trans_quat(
            self._init_pos_tc, self._init_quat_tc, self.link.get_pos(envs_idx), self.link.get_quat(envs_idx)
        )
        vverts_pos = pos[..., None, :] + gu.transform_by_quat(self._aabb_verts, quat[..., None, :])
        return torch.stack((vverts_pos.min(dim=-2).values, vverts_pos.max(dim=-2).values), dim=-2)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Get the unique ID of the vgeom.
        """
        pass

    @property
    def idx(self):
        """
        Get the global index of the vgeom in RigidSolver.
        """
        pass

    @property
    def link(self):
        """
        Get the link that the vgeom belongs to.
        """
        pass

    @property
    def entity(self):
        """
        Get the entity that the vgeom belongs to.
        """
        pass

    @property
    def vmesh(self):
        pass

    @property
    def solver(self):
        """
        Get the solver that the vgeom belongs to.
        """
        pass

    @property
    def metadata(self):
        """
        Get the metadata of the vgeom.
        """
        pass

    @property
    def init_pos(self):
        """
        Get the initial position of the vgeom.
        """
        pass

    @property
    def init_quat(self):
        """
        Get the initial quaternion of the vgeom.
        """
        pass

    @property
    def init_vverts(self):
        """
        Get the initial vertices of the vgeom.
        """
        pass

    @property
    def init_vfaces(self):
        """
        Get the initial faces of the vgeom.
        """
        pass

    @property
    def init_vnormals(self):
        """
        Get the initial normals of the vgeom.
        """
        pass

    @property
    def uvs(self):
        """
        Get the UV coordinates of the vgeom.
        """
        pass

    @property
    def surface(self):
        """
        Get the surface object of the vgeom.
        """
        pass

    @property
    def n_vverts(self):
        """
        Number of vertices of the vgeom.
        """
        pass

    @property
    def n_vfaces(self):
        """
        Number of faces of the vgeom.
        """
        pass

    @property
    def vvert_start(self):
        """
        Get the starting index of the vgeom's vertices in the rigid solver.
        """
        pass

    @property
    def vface_start(self):
        """
        Get the starting index of the vgeom's faces in the rigid solver.
        """
        pass

    @property
    def vvert_end(self):
        """
        Get the ending index of the vgeom's vertices in the rigid solver.
        """
        pass

    @property
    def vface_end(self):
        """
        Get the ending index of the vgeom's faces in the rigid solver.
        """
        pass

    @property
    def is_built(self):
        """
        Whether the rigid entity the vgeom belongs to is built.
        """
        pass

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")

    @property
    def is_fixed(self) -> bool:
        """
        Whether this vgeom is fixed in the world.
        """
        pass

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        pass
