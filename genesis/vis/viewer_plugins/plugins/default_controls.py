import os
from typing import TYPE_CHECKING

import genesis as gs
from genesis.vis.keybindings import Key, Keybind

from ..viewer_plugin import ViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class DefaultControlsPlugin(ViewerPlugin):
    """
    Default keyboard controls for the Genesis viewer.

    This plugin handles the standard viewer keyboard shortcuts for recording, changing render modes, etc.
    """

    def __init__(self):
        super().__init__()

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        self.viewer.register_keybinds(
            Keybind("record_video", Key.R, callback=self._toggle_record_video, allow_overload=True),
            Keybind("save_image", Key.S, callback=self._save_image, allow_overload=True),
            Keybind("reset_camera", Key.Z, callback=self._reset_camera, allow_overload=True),
            Keybind("camera_rotation", Key.A, callback=self._toggle_cam_rotation, allow_overload=True),
            Keybind("shadow", Key.H, callback=self._toggle_shadow, allow_overload=True),
            Keybind("face_normals", Key.F, callback=self._toggle_face_normals, allow_overload=True),
            Keybind("vertex_normals", Key.V, callback=self._toggle_vertex_normals, allow_overload=True),
            Keybind("world_frame", Key.W, callback=self._toggle_world_frame, allow_overload=True),
            Keybind("link_frame", Key.L, callback=self._toggle_link_frame, allow_overload=True),
            Keybind("wireframe", Key.D, callback=self._toggle_wireframe, allow_overload=True),
            Keybind("camera_frustum", Key.C, callback=self._toggle_camera_frustum, allow_overload=True),
            Keybind("reload_shader", Key.P, callback=self._reload_shader, allow_overload=True),
            Keybind("fullscreen_mode", Key.F11, callback=self._toggle_fullscreen, allow_overload=True),
        )

    def _toggle_cam_rotation(self):
        pass

    def _toggle_fullscreen(self):
        pass

    def _toggle_shadow(self):
        pass

    def _toggle_world_frame(self):
        pass

    def _toggle_link_frame(self):
        pass

    def _toggle_camera_frustum(self):
        pass

    def _toggle_face_normals(self):
        pass

    def _toggle_vertex_normals(self):
        pass

    def _toggle_record_video(self):
        pass

    def _save_image(self):
        pass

    def _toggle_wireframe(self):
        pass

    def _reset_camera(self):
        pass

    def _reload_shader(self):
        pass
