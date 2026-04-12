"""Scenes, conforming to the glTF 2.0 standards as specified in
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-scene

Author: Matthew Matl
"""

import networkx as nx
import numpy as np

from .camera import Camera
from .light import DirectionalLight, Light, PointLight, SpotLight
from .mesh import Mesh
from .node import Node
from .utils import format_color_vector


CORNER_INDICES = np.array(
    [
        [0, 1, 2],
        [3, 1, 2],
        [3, 4, 2],
        [0, 4, 2],
        [0, 1, 5],
        [3, 1, 5],
        [3, 4, 5],
        [0, 4, 5],
    ]
)


class Scene(object):
    """A hierarchical scene graph.

    Parameters
    ----------
    nodes : list of :class:`Node`
        The set of all nodes in the scene.
    bg_color : (4,) float, optional
        Background color of scene.
    ambient_light : (3,) float, optional
        Color of ambient light. Defaults to no ambient light.
    name : str, optional
        The user-defined name of this object.
    """

    def __init__(self, nodes=None, bg_color=None, ambient_light=None, n_envs=None, name=None):
        if bg_color is None:
            bg_color = np.ones(4)
        else:
            bg_color = format_color_vector(bg_color, 4)

        if ambient_light is None:
            ambient_light = np.zeros(3)

        if n_envs is None:
            n_envs = 1

        if nodes is None:
            nodes = set()
        self._nodes = set()  # Will be added at the end of this function

        self.bg_color = bg_color
        self.ambient_light = ambient_light
        self.n_envs = n_envs
        self.name = name

        self._name_to_nodes = {}
        self._obj_to_nodes = {}
        self._obj_name_to_nodes = {}
        self._mesh_nodes = set()
        self._point_light_nodes = set()
        self._spot_light_nodes = set()
        self._directional_light_nodes = set()
        self._camera_nodes = set()
        self._main_camera_node = None
        self._bounds = None

        self._meshes_updated = False

        # Transform tree
        self._digraph = nx.DiGraph()
        self._digraph.add_node("world")
        self._path_cache = {}

        # Find root nodes and add them
        if len(nodes) > 0:
            node_parent_map = {n: None for n in nodes}
            for node in nodes:
                for child in node.children:
                    if node_parent_map[child] is not None:
                        raise ValueError("Nodes may not have more than one parent")
                    node_parent_map[child] = node
            for node in node_parent_map:
                if node_parent_map[node] is None:
                    self.add_node(node)

    @property
    def name(self):
        """str : The user-defined name of this object."""
        pass

    @name.setter
    def name(self, value):
        pass

    @property
    def nodes(self):
        """set of :class:`Node` : Set of nodes in the scene."""
        pass

    @property
    def bg_color(self):
        """(3,) float : The scene background color."""
        pass

    @bg_color.setter
    def bg_color(self, value):
        pass

    @property
    def ambient_light(self):
        """(3,) float : The ambient light in the scene."""
        pass

    @ambient_light.setter
    def ambient_light(self, value):
        pass

    def reset_meshes_updated(self):
        self._meshes_updated = False

    @property
    def meshes_updated(self):
        pass

    @property
    def meshes(self):
        """set of :class:`Mesh` : The meshes in the scene."""
        pass

    @property
    def mesh_nodes(self):
        """set of :class:`Node` : The nodes containing meshes."""
        pass

    @property
    def lights(self):
        """set of :class:`Light` : The lights in the scene."""
        pass

    @property
    def light_nodes(self):
        """set of :class:`Node` : The nodes containing lights."""
        pass

    @property
    def point_lights(self):
        """set of :class:`PointLight` : The point lights in the scene."""
        pass

    @property
    def point_light_nodes(self):
        """set of :class:`Node` : The nodes containing point lights."""
        pass

    @property
    def spot_lights(self):
        """set of :class:`SpotLight` : The spot lights in the scene."""
        pass

    @property
    def spot_light_nodes(self):
        """set of :class:`Node` : The nodes containing spot lights."""
        pass

    @property
    def directional_lights(self):
        """set of :class:`DirectionalLight` : The directional lights in
        the scene.
        """
        pass

    @property
    def directional_light_nodes(self):
        """set of :class:`Node` : The nodes containing directional lights."""
        pass

    @property
    def cameras(self):
        """set of :class:`Camera` : The cameras in the scene."""
        pass

    @property
    def camera_nodes(self):
        """set of :class:`Node` : The nodes containing cameras in the scene."""
        pass

    @property
    def main_camera_node(self):
        """set of :class:`Node` : The node containing the main camera in the
        scene.
        """
        pass

    @main_camera_node.setter
    def main_camera_node(self, value):
        pass

    @property
    def bounds(self):
        """(2,3) float : The axis-aligned bounds of the scene."""
        pass

    @property
    def centroid(self):
        """(3,) float : The centroid of the scene's axis-aligned bounding box
        (AABB).
        """
        pass

    @property
    def extents(self):
        """(3,) float : The lengths of the axes of the scene's AABB."""
        pass

    @property
    def scale(self):
        """(3,) float : The length of the diagonal of the scene's AABB."""
        pass

    def add(self, obj, name=None, pose=None, parent_node=None, parent_name=None):
        """Add an object (mesh, light, or camera) to the scene.

        Parameters
        ----------
        obj : :class:`Mesh`, :class:`Light`, or :class:`Camera`
            The object to add to the scene.
        name : str
            A name for the new node to be created.
        pose : (4,4) float
            The local pose of this node relative to its parent node.
        parent_node : :class:`Node`
            The parent of this Node. If None, the new node is a root node.
        parent_name : str
            The name of the parent node, can be specified instead of
            `parent_node`.

        Returns
        -------
        node : :class:`Node`
            The newly-created and inserted node.
        """
        if isinstance(obj, Mesh):
            node = Node(name=name, matrix=pose, mesh=obj)
        elif isinstance(obj, Light):
            node = Node(name=name, matrix=pose, light=obj)
        elif isinstance(obj, Camera):
            node = Node(name=name, matrix=pose, camera=obj)
        elif isinstance(obj, Node):
            node = obj
        else:
            raise TypeError("Unrecognized object type")

        if parent_node is None and parent_name is not None:
            try:
                (parent_node,) = self.get_nodes(name=parent_name)
            except ValueError:
                parent_nodes = self.get_nodes(name=parent_name)
                if len(parent_nodes) == 0:
                    raise ValueError(f"No parent node with name '{parent_name}' found")
                elif len(parent_nodes) > 1:
                    raise ValueError(f"More than one parent node with name '{parent_name}' found")

        self.add_node(node, parent_node=parent_node)

        return node

    def get_nodes(self, node=None, name=None, obj=None, obj_name=None):
        """Search for existing nodes. Only nodes matching all specified
        parameters is returned, or None if no such node exists.

        Parameters
        ----------
        node : :class:`Node`, optional
            If present, returns this node if it is in the scene.
        name : str
            A name for the Node.
        obj : :class:`Mesh`, :class:`Light`, or :class:`Camera`
            An object that is attached to the node.
        obj_name : str
            The name of an object that is attached to the node.

        Returns
        -------
        nodes : set of :class:`.Node`
            The nodes that match all query terms.
        """
        if node is not None:
            if node in self.nodes:
                return set([node])
            else:
                return set()
        nodes = set(self.nodes)
        if name is not None:
            matches = set()
            if name in self._name_to_nodes:
                matches = self._name_to_nodes[name]
            nodes = nodes & matches
        if obj is not None:
            matches = set()
            if obj in self._obj_to_nodes:
                matches = self._obj_to_nodes[obj]
            nodes = nodes & matches
        if obj_name is not None:
            matches = set()
            if obj_name in self._obj_name_to_nodes:
                matches = self._obj_name_to_nodes[obj_name]
            nodes = nodes & matches

        return nodes

    def add_node(self, node, parent_node=None):
        """Add a Node to the scene.

        Parameters
        ----------
        node : :class:`Node`
            The node to be added.
        parent_node : :class:`Node`
            The parent of this Node. If None, the new node is a root node.
        """
        if node in self.nodes:
            raise ValueError("Node already in scene")
        self.nodes.add(node)

        # Add node to sets
        if node.name is not None:
            if node.name not in self._name_to_nodes:
                self._name_to_nodes[node.name] = set()
            self._name_to_nodes[node.name].add(node)
        for obj in [node.mesh, node.camera, node.light]:
            if obj is not None:
                if obj not in self._obj_to_nodes:
                    self._obj_to_nodes[obj] = set()
                self._obj_to_nodes[obj].add(node)
                if obj.name is not None:
                    if obj.name not in self._obj_name_to_nodes:
                        self._obj_name_to_nodes[obj.name] = set()
                    self._obj_name_to_nodes[obj.name].add(node)
        if node.mesh is not None:
            self._meshes_updated = True
            self._mesh_nodes.add(node)
        if node.light is not None:
            if isinstance(node.light, PointLight):
                self._point_light_nodes.add(node)
            if isinstance(node.light, SpotLight):
                self._spot_light_nodes.add(node)
            if isinstance(node.light, DirectionalLight):
                self._directional_light_nodes.add(node)
        if node.camera is not None:
            self._camera_nodes.add(node)
            if self._main_camera_node is None:
                self._main_camera_node = node

        if parent_node is None:
            parent_node = "world"
        elif parent_node not in self.nodes:
            raise ValueError("Parent node must already be in scene")
        elif node not in parent_node.children:
            parent_node.children.append(node)

        # Create node in graph
        self._digraph.add_node(node)
        self._digraph.add_edge(node, parent_node)

        # Iterate over children
        for child in node.children:
            self.add_node(child, node)

        self._path_cache = {}
        self._bounds = None

    def has_node(self, node):
        """Check if a node is already in the scene.

        Parameters
        ----------
        node : :class:`Node`
            The node to be checked.

        Returns
        -------
        has_node : bool
            True if the node is already in the scene and false otherwise.
        """
        return node in self.nodes

    def remove_node(self, node):
        """Remove a node and all its children from the scene.

        Parameters
        ----------
        node : :class:`Node`
            The node to be removed.
        """
        # Disconnect self from parent who is staying in the graph
        parent = list(self._digraph.neighbors(node))[0]
        self._remove_node(node)
        if isinstance(parent, Node):
            parent.children.remove(node)
        self._path_cache = {}
        self._bounds = None

    def get_pose(self, node):
        """Get the world-frame pose of a node in the scene.

        Parameters
        ----------
        node : :class:`Node`
            The node to find the pose of.

        Returns
        -------
        pose : (4,4) float
            The transform matrix for this node.
        """
        if node not in self.nodes:
            raise ValueError("Node must already be in scene")
        if node in self._path_cache:
            path = self._path_cache[node]
        else:
            # Get path from from_frame to to_frame
            path = nx.shortest_path(self._digraph, node, "world")
            self._path_cache[node] = path

        # Traverse from from_node to to_node
        pose = path[0].matrix
        for parent_node in path[1:-1]:
            pose = np.dot(parent_node.matrix, pose)

        return pose

    def set_pose(self, node, pose):
        """Set the local-frame pose of a node in the scene.

        Parameters
        ----------
        node : :class:`Node`
            The node to set the pose of.
        pose : (4,4) float
            The pose to set the node to.
        """
        node._matrix = pose
        if node.mesh is not None:
            self._bounds = None

    def reorder_vertices(self, node, vertices):
        if node.mesh is None or len(node.mesh.primitives) != 1:
            raise ValueError("Node must have one primitive")
        primitive = node.mesh.primitives[0]
        return vertices[primitive.vertex_mapping] if primitive.vertex_mapping is not None else vertices

    def get_buffer_id(self, node, buffer_name):
        if node.mesh is None or len(node.mesh.primitives) != 1:
            raise ValueError("Node must have one primitive")
        primitive = node.mesh.primitives[0]
        return primitive.get_buffer_id(buffer_name)

    def clear(self):
        """Clear out all nodes to form an empty scene."""
        self._nodes = set()

        self._name_to_nodes = {}
        self._obj_to_nodes = {}
        self._obj_name_to_nodes = {}
        self._mesh_nodes = set()
        self._point_light_nodes = set()
        self._spot_light_nodes = set()
        self._directional_light_nodes = set()
        self._camera_nodes = set()
        self._main_camera_node = None
        self._bounds = None

        # Transform tree
        self._digraph = nx.DiGraph()
        self._digraph.add_node("world")
        self._path_cache = {}

    def _remove_node(self, node):
        """Remove a node and all its children from the scene.

        Parameters
        ----------
        node : :class:`Node`
            The node to be removed.
        """

        # Remove self from nodes
        self.nodes.remove(node)

        # Remove children
        for child in node.children:
            self._remove_node(child)

        # Remove self from the graph
        self._digraph.remove_node(node)

        # Remove from maps
        if node.name in self._name_to_nodes:
            self._name_to_nodes[node.name].remove(node)
            if len(self._name_to_nodes[node.name]) == 0:
                self._name_to_nodes.pop(node.name)
        for obj in [node.mesh, node.camera, node.light]:
            if obj is None:
                continue
            self._obj_to_nodes[obj].remove(node)
            if len(self._obj_to_nodes[obj]) == 0:
                self._obj_to_nodes.pop(obj)
            if obj.name is not None:
                self._obj_name_to_nodes[obj.name].remove(node)
                if len(self._obj_name_to_nodes[obj.name]) == 0:
                    self._obj_name_to_nodes.pop(obj.name)
        if node.mesh is not None:
            self._meshes_updated = True
            self._mesh_nodes.remove(node)
        if node.light is not None:
            if isinstance(node.light, PointLight):
                self._point_light_nodes.remove(node)
            if isinstance(node.light, SpotLight):
                self._spot_light_nodes.remove(node)
            if isinstance(node.light, DirectionalLight):
                self._directional_light_nodes.remove(node)
        if node.camera is not None:
            self._camera_nodes.remove(node)
            if self._main_camera_node == node:
                if len(self._camera_nodes) > 0:
                    self._main_camera_node = next(iter(self._camera_nodes))
                else:
                    self._main_camera_node = None

    @staticmethod
    def from_trimesh_scene(trimesh_scene, bg_color=None, ambient_light=None):
        """Create a :class:`.Scene` from a :class:`trimesh.scene.scene.Scene`.

        Parameters
        ----------
        trimesh_scene : :class:`trimesh.scene.scene.Scene`
            Scene with :class:~`trimesh.base.Trimesh` objects.
        bg_color : (4,) float
            Background color for the created scene.
        ambient_light : (3,) float or None
            Ambient light in the scene.

        Returns
        -------
        scene_pr : :class:`Scene`
            A scene containing the same geometry as the trimesh scene.
        """
        pass

    def sorted_mesh_nodes(self):
        cam_pos = self.get_pose(self.main_camera_node)
        cam_loc = cam_pos[..., :3, 3]
        batched_pos = len(cam_pos.shape) == 3
        solid_nodes = []
        trans_nodes = []
        for node in self.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        # TODO BETTER SORTING METHOD
        if batched_pos:
            # FIXME normally sorting should be done PER scene when having a batched rasterizer render
            trans_nodes.sort(key=lambda n: -np.linalg.norm(self.get_pose(n)[:3, 3] - cam_loc[0]))
            solid_nodes.sort(key=lambda n: -np.linalg.norm(self.get_pose(n)[:3, 3] - cam_loc[0]))
        else:
            trans_nodes.sort(key=lambda n: -np.linalg.norm(self.get_pose(n)[:3, 3] - cam_loc))
            solid_nodes.sort(key=lambda n: -np.linalg.norm(self.get_pose(n)[:3, 3] - cam_loc))

        return solid_nodes + trans_nodes
