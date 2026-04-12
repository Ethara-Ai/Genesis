import math

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.constants import IMAGE_TYPE
from genesis.utils.misc import qd_to_torch

from .camera import Camera
from .rasterizer_context import SegmentationColorMap

# Optional imports for platform-specific functionality
try:
    from gs_madrona.renderer_gs import MadronaBatchRendererAdapter

    _MADRONA_AVAILABLE = True
except ImportError:
    MadronaBatchRendererAdapter = None
    _MADRONA_AVAILABLE = False


def _transform_camera_quat(quat):
    # quat for Madrona needs to be transformed to y-forward
    w, x, y, z = torch.unbind(quat, dim=-1)
    return torch.stack([x + w, x - w, y - z, y + z], dim=-1) / math.sqrt(2.0)


def _make_tensor(data, *, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype, device=gs.device)


class GenesisGeomRetriever:
    def __init__(self, rigid_solver, seg_level):
        self.rigid_solver = rigid_solver
        self.seg_color_map = SegmentationColorMap(to_torch=True)
        self.seg_level = seg_level
        self.geom_idxc = None

        self.default_geom_group = 2
        self.default_enabled_geom_groups = np.array([self.default_geom_group], dtype=np.int32)

    def build(self):
        self.n_vgeoms = self.rigid_solver.n_vgeoms
        self.geom_idxc = []
        vgeoms = self.rigid_solver.vgeoms
        for vgeom in vgeoms:
            seg_key = self.get_seg_key(vgeom)
            seg_idxc = self.seg_color_map.seg_key_to_idxc(seg_key)
            self.geom_idxc.append(seg_idxc)
        self.geom_idxc = torch.tensor(self.geom_idxc, dtype=torch.int32, device=gs.device)
        self.seg_color_map.generate_seg_colors()

    def get_seg_key(self, vgeom):
        if self.seg_level == "geom":
            return (vgeom.entity.idx, vgeom.link.idx, vgeom.idx)
        elif self.seg_level == "link":
            return (vgeom.entity.idx, vgeom.link.idx)
        elif self.seg_level == "entity":
            return vgeom.entity.idx
        else:
            gs.raise_exception(f"Unsupported segmentation level: {self.seg_level}")

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_meshes_static(self):
        pass

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_property_torch(self, num_worlds):
        pass

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_state_torch(self):
        pass


class Light:
    def __init__(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._pos = pos
        self._dir = tuple(dir / np.linalg.norm(dir))
        self._color = color
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff
        self._attenuation = attenuation

    @property
    def pos(self):
        pass

    @property
    def dir(self):
        pass

    @property
    def color(self):
        pass

    @property
    def intensity(self):
        pass

    @property
    def directional(self):
        pass

    @property
    def castshadow(self):
        pass

    @property
    def cutoffRad(self):
        pass

    @property
    def cutoffDeg(self):
        pass

    @property
    def attenuation(self):
        pass


class BatchRenderer(RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer, renderer_options, vis_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._use_rasterizer = renderer_options.use_rasterizer
        self._renderer = None
        self._geom_retriever = GenesisGeomRetriever(self._visualizer.scene.rigid_solver, vis_options.segmentation_level)
        self._data_cache = {}
        self._t = -1

    def add_light(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._lights.append(Light(pos, dir, color, intensity, directional, castshadow, cutoff, attenuation))

    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if not _MADRONA_AVAILABLE:
            gs.raise_exception("Madrona batch renderer is only supported on Linux x86-64.")

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRenderer requires CUDA backend.")
        gpu_id = gs.device.index if gs.device.index is not None else 0

        # Extract the complete list of non-debug cameras
        self._cameras = gs.List(
            [camera for camera in self._visualizer._cameras if not isinstance(camera, Camera) or not camera.debug]
        )
        if not self._cameras:
            gs.raise_exception("Please add at least one camera when using BatchRender.")

        # Build the geometry retriever
        self._geom_retriever.build()

        # Make sure that all cameras have identical resolution
        try:
            ((camera_width, camera_height),) = set(camera.res for camera in self._cameras)
        except ValueError as e:
            gs.raise_exception_from("All cameras must have the exact same resolution when using BatchRender.", e)

        self._renderer = MadronaBatchRendererAdapter(
            geom_retriever=self._geom_retriever,
            gpu_id=gs.device.index if gs.device.index is not None else 0,
            num_worlds=max(self._visualizer.scene.n_envs, 1),
            num_lights=len(self._lights),
            cam_fovs_tensor=_make_tensor([camera.fov for camera in self._cameras]),
            cam_znears_tensor=_make_tensor([camera.near for camera in self._cameras]),
            cam_zfars_tensor=_make_tensor([camera.far for camera in self.cameras]),
            cam_proj_types_tensor=_make_tensor(
                [camera.model == "fisheye" for camera in self._cameras], dtype=torch.uint32
            ),
            batch_render_view_width=camera_width,
            batch_render_view_height=camera_height,
            add_cam_debug_geo=False,
            use_rasterizer=self._use_rasterizer,
        )
        self._renderer.init(
            cam_pos_tensor=torch.stack([torch.atleast_2d(camera.get_pos()) for camera in self._cameras], dim=1),
            cam_rot_tensor=_transform_camera_quat(
                torch.stack([torch.atleast_2d(camera.get_quat()) for camera in self._cameras], dim=1)
            ),
            lights_pos_tensor=_make_tensor([light.pos for light in self._lights]).reshape((-1, 3)),
            lights_dir_tensor=_make_tensor([light.dir for light in self._lights]).reshape((-1, 3)),
            lights_rgb_tensor=_make_tensor([light.color for light in self._lights]).reshape((-1, 3)),
            lights_directional_tensor=_make_tensor([light.directional for light in self._lights], dtype=torch.bool),
            lights_castshadow_tensor=_make_tensor([light.castshadow for light in self._lights], dtype=torch.bool),
            lights_cutoff_tensor=_make_tensor([light.cutoffRad for light in self._lights]),
            lights_attenuation_tensor=_make_tensor([light.attenuation for light in self._lights]),
            lights_intensity_tensor=_make_tensor([light.intensity for light in self._lights]),
        )

    def update_scene(self, force_render: bool = False):
        self._visualizer._context.update(force_render)

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False, force_render=False):
        """
        Render all cameras in the batch.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render the rgb image.
        depth : bool, optional
            Whether to render the depth image.
        segmentation : bool, optional
            Whether to render the segmentation image.
        normal : bool, optional
            Whether to render the normal image.
        antialiasing : bool, optional
            Whether to apply anti-aliasing.
        force_render : bool, optional
            Whether to force render the scene.

        Returns
        -------
        rgb_arr : tuple of arrays
            The sequence of rgb images associated with each camera.
        depth_arr : tuple of arrays
            The sequence of depth images associated with each camera.
        segmentation_arr : tuple of arrays
            The sequence of segmentation images associated with each camera.
        normal_arr : tuple of arrays
            The sequence of normal images associated with each camera.
        """

        # Clear cache if requested or necessary
        if force_render or self._t < self._visualizer.scene.t:
            self._data_cache.clear()

        # Fetch available cached data
        request = (rgb, depth, segmentation, normal)
        cache_key = (antialiasing,)
        cached = [self._data_cache.get((img_type, cache_key), None) for img_type in IMAGE_TYPE]

        # Force disabling rendering whenever cached data is already available
        needed = tuple(req and arr is None for req, arr in zip(request, cached))

        # Early return if everything requested is already cached
        if not any(needed):
            return tuple(arr if req else None for req, arr in zip(request, cached))

        # Update scene
        self.update_scene(force_render)

        # Render only what is needed (flags still passed to renderer)
        cameras_pos = torch.stack([torch.atleast_2d(camera.get_pos()) for camera in self._cameras], dim=1)
        cameras_quat = torch.stack([torch.atleast_2d(camera.get_quat()) for camera in self._cameras], dim=1)
        cameras_quat = _transform_camera_quat(cameras_quat)
        render_flags = np.array(
            (
                *(
                    needed[img_type]
                    for img_type in (IMAGE_TYPE.RGB, IMAGE_TYPE.DEPTH, IMAGE_TYPE.NORMAL, IMAGE_TYPE.SEGMENTATION)
                ),
                antialiasing,
            ),
            dtype=np.uint32,
        )
        rendered = list(self._renderer.render(cameras_pos, cameras_quat, render_flags))

        # convert seg geom idx to seg_idxc
        if needed[IMAGE_TYPE.SEGMENTATION]:
            seg_geoms = rendered[IMAGE_TYPE.SEGMENTATION]
            mask = seg_geoms != -1
            seg_geoms[mask] = self._geom_retriever.geom_idxc[seg_geoms[mask]]
            seg_geoms[~mask] = 0

        # Post-processing:
        # * Remove alpha channel from RGBA
        # * Squeeze env and channel dims if necessary
        # * Split along camera dim
        for img_type, data in enumerate(rendered):
            if needed[img_type]:
                data = data.swapaxes(0, 1)
                if self._visualizer.scene.n_envs == 0:
                    data = data.squeeze(1)
                rendered[img_type] = tuple(data[..., :3].squeeze(-1))

        # Convert center distance depth to plane distance
        if not self._use_rasterizer and needed[IMAGE_TYPE.DEPTH]:
            rendered[IMAGE_TYPE.DEPTH] = tuple(
                camera.distance_center_to_plane(depth_data)
                for camera, depth_data in zip(self._cameras, rendered[IMAGE_TYPE.DEPTH])
            )

        # Update cache
        self._t = self._visualizer.scene.t
        for img_type, data in enumerate(rendered):
            if needed[img_type]:
                self._data_cache[(img_type, cache_key)] = rendered[img_type]

        # Return in the required order, or None if not requested
        return tuple(self._data_cache[(img_type, cache_key)] if needed[img_type] else None for img_type in IMAGE_TYPE)

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self._geom_retriever.seg_color_map.colorize_seg_idxc_arr(seg_idxc_arr)

    def destroy(self):
        self._lights.clear()
        self._data_cache.clear()
        if self._renderer is not None:
            del self._renderer.madrona
            self._renderer = None

    def reset(self):
        self._t = -1

    @property
    def lights(self):
        pass

    @property
    def cameras(self):
        pass

    @property
    def seg_idxc_map(self):
        pass
