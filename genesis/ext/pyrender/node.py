"""Nodes, conforming to the glTF 2.0 standards as specified in
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-node

Author: Matthew Matl
"""

import numpy as np

import trimesh.transformations as transformations

from .camera import Camera
from .mesh import Mesh
from .light import Light


class Node(object):
    """A node in the node hierarchy.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    camera : :class:`Camera`, optional
        The camera in this node.
    children : list of :class:`Node`
        The children of this node.
    skin : int, optional
        The index of the skin referenced by this node.
    matrix : (4,4) float, optional
        A floating-point 4x4 transformation matrix.
    mesh : :class:`Mesh`, optional
        The mesh in this node.
    rotation : (4,) float, optional
        The node's unit quaternion in the order (x, y, z, w), where
        w is the scalar.
    scale : (3,) float, optional
        The node's non-uniform scale, given as the scaling factors along the x,
        y, and z axes.
    translation : (3,) float, optional
        The node's translation along the x, y, and z axes.
    weights : (n,) float
        The weights of the instantiated Morph Target. Number of elements must
        match number of Morph Targets of used mesh.
    light : :class:`Light`, optional
        The light in this node.
    """

    def __init__(
        self,
        name=None,
        camera=None,
        children=None,
        skin=None,
        matrix=None,
        mesh=None,
        rotation=None,
        scale=None,
        translation=None,
        weights=None,
        light=None,
    ):
        # Set defaults
        if children is None:
            children = []

        self._matrix = None
        self._scale = None
        self._rotation = None
        self._translation = None
        if matrix is None:
            if rotation is None:
                rotation = np.array([0.0, 0.0, 0.0, 1.0])
            if translation is None:
                translation = np.zeros(3)
            if scale is None:
                scale = np.ones(3)
            self.rotation = rotation
            self.translation = translation
            self.scale = scale
        else:
            self.matrix = matrix

        self.name = name
        self.camera = camera
        self.children = children
        self.skin = skin
        self.mesh = mesh
        self.weights = weights
        self.light = light

    @property
    def name(self):
        """str : The user-defined name of this object."""
        pass

    @name.setter
    def name(self, value):
        pass

    @property
    def camera(self):
        """:class:`Camera` : The camera in this node."""
        pass

    @camera.setter
    def camera(self, value):
        pass

    @property
    def children(self):
        """list of :class:`Node` : The children of this node."""
        pass

    @children.setter
    def children(self, value):
        pass

    @property
    def skin(self):
        """int : The skin index for this node."""
        pass

    @skin.setter
    def skin(self, value):
        pass

    @property
    def mesh(self):
        """:class:`Mesh` : The mesh in this node."""
        pass

    @mesh.setter
    def mesh(self, value):
        pass

    @property
    def light(self):
        """:class:`Light` : The light in this node."""
        pass

    @light.setter
    def light(self, value):
        pass

    @property
    def rotation(self):
        """(4,) float : The xyzw quaternion for this node."""
        pass

    @rotation.setter
    def rotation(self, value):
        pass

    @property
    def translation(self):
        """(3,) float : The translation for this node."""
        pass

    @translation.setter
    def translation(self, value):
        pass

    @property
    def scale(self):
        """(3,) float : The scale for this node."""
        pass

    @scale.setter
    def scale(self, value):
        pass

    @property
    def matrix(self):
        """(4,4) float : The homogenous transform matrix for this node.

        Note that this matrix's elements are not settable,
        it's just a copy of the internal matrix. You can set the whole
        matrix, but not an individual element.
        """
        if self._matrix is None:
            self._matrix = self._m_from_tqs(self.translation, self.rotation, self.scale)
        return self._matrix.copy()

    @matrix.setter
    def matrix(self, value):
        value = np.array(value, copy=True)
        if value.shape != (4, 4):
            raise ValueError("Matrix must be a 4x4 numpy ndarray")
        if not np.allclose(value[3, :], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError("Bottom row of matrix must be [0,0,0,1]")
        self._matrix = value
        self._rotation = Node._q_from_m(value)
        self._scale = Node._s_from_m(value)
        self._translation = Node._t_from_m(value)

    @staticmethod
    def _t_from_m(m):
        return m[:3, 3]

    @staticmethod
    def _r_from_m(m):
        U = m[:3, :3]
        norms = np.linalg.norm(U.T, axis=1)
        return U / norms

    @staticmethod
    def _q_from_m(m):
        M = np.eye(4)
        M[:3, :3] = Node._r_from_m(m)
        q_wxyz = transformations.quaternion_from_matrix(M)
        return np.roll(q_wxyz, -1)

    @staticmethod
    def _s_from_m(m):
        return np.linalg.norm(m[:3, :3].T, axis=1)

    @staticmethod
    def _r_from_q(q):
        q_wxyz = np.roll(q, 1)
        return transformations.quaternion_matrix(q_wxyz)[:3, :3]

    @staticmethod
    def _m_from_tqs(t, q, s):
        S = np.eye(4)
        S[:3, :3] = np.diag(s)

        R = np.eye(4)
        R[:3, :3] = Node._r_from_q(q)

        T = np.eye(4)
        T[:3, 3] = t

        return T.dot(R.dot(S))
