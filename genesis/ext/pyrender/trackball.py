"""Trackball class for 3D manipulation of viewpoints."""

import numpy as np

import trimesh.transformations as transformations


EPSILON = np.finfo(np.float32).eps


class Trackball(object):
    """A trackball class for creating camera transforms from mouse movements."""

    STATE_ROTATE = 0
    STATE_PAN = 1
    STATE_ROLL = 2
    STATE_ZOOM = 3

    def __init__(self, pose, size, scale, target=np.array([0.0, 0.0, 0.0])):
        """Initialize a trackball with an initial camera-to-world pose
        and the given parameters.

        Parameters
        ----------
        pose : [4,4]
            An initial camera-to-world pose for the trackball.

        size : (float, float)
            The width and height of the camera image in pixels.

        scale : float
            The diagonal of the scene's bounding box --
            used for ensuring translation motions are sufficiently
            fast for differently-sized scenes.

        target : (3,) float
            The center of the scene in world coordinates.
            The trackball will revolve around this point.
        """
        self._size = np.asarray(size, dtype=np.float32)
        self._scale = float(scale)

        self._pose = pose.copy()
        self._n_pose = pose.copy()

        self._target = target
        self._n_target = target

        self._state = Trackball.STATE_ROTATE

        self._pdown = np.array([0.0, 0.0], dtype=np.float32)

    @property
    def pose(self):
        """autolab_core.RigidTransform : The current camera-to-world pose."""
        pass

    def set_state(self, state):
        """Set the state of the trackball in order to change the effect of
        dragging motions.

        Parameters
        ----------
        state : int
            One of Trackball.STATE_ROTATE, Trackball.STATE_PAN,
            Trackball.STATE_ROLL, and Trackball.STATE_ZOOM.
        """
        self._state = state

    def resize(self, size):
        """Resize the window.

        Parameters
        ----------
        size : (float, float)
            The new width and height of the camera image in pixels.
        """
        self._size = np.array(size)

    def down(self, point):
        """Record an initial mouse press at a given point.

        Parameters
        ----------
        point : (2,) int
            The x and y pixel coordinates of the mouse press.
        """
        pass

    def drag(self, point):
        """Update the trackball during a drag.

        Parameters
        ----------
        point : (2,) int
            The current x and y pixel coordinates of the mouse during a drag.
            This will compute a movement for the trackball with the relative
            motion between this point and the one marked by down().
        """
        pass

    def scroll(self, clicks):
        """Zoom using a mouse scroll wheel motion.

        Parameters
        ----------
        clicks : int
            The number of clicks. Positive numbers indicate forward wheel
            movement.
        """
        pass

    def rotate(self, azimuth, axis=None):
        """Rotate the trackball about the "Up" axis by azimuth radians.

        Parameters
        ----------
        azimuth : float
            The number of radians to rotate.
        """
        target = self._target

        y_axis = self._n_pose[:3, 1]
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        self._n_pose = x_rot_mat @ self._n_pose

        y_axis = self._pose[:3, 1]
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        self._pose = x_rot_mat @ self._pose

    def set_camera_pose(self, pose):
        """Set the camera pose.

        Parameters
        ----------
        pose : [4,4]
            The camera pose.
        """
        self._n_pose = pose
