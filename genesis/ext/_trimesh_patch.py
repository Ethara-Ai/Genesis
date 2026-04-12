import os
import re
from collections import defaultdict, deque

import numpy as np

try:
    # `pip install pillow`
    # optional: used for textured meshes
    from PIL import Image
except BaseException as E:
    # if someone tries to use Image re-raise
    # the import error so they can debug easily
    from ..exceptions import ExceptionWrapper

    Image = ExceptionWrapper(E)

import trimesh
from trimesh import util
from trimesh.constants import log, tol
from trimesh.visual.color import to_float
from trimesh.visual.material import SimpleMaterial
from trimesh.visual.texture import TextureVisuals, unmerge_faces

import genesis as gs


def load_obj(
    file_obj,
    resolver=None,
    group_material=True,
    skip_materials=False,
    maintain_order=False,
    **kwargs,
):
    """
    Load a Wavefront OBJ file into kwargs for a trimesh.Scene
    object.

    Parameters
    --------------
    file_obj : file like object
      Contains OBJ data
    resolver : trimesh.visual.resolvers.Resolver
      Allow assets such as referenced textures and
      material files to be loaded
    group_material : bool
      Group faces that share the same material
      into the same mesh.
    skip_materials : bool
      Don't load any materials.
    maintain_order : bool or None
      Do not reorder faces or vertices which may result
      in visual artifacts.

    Returns
    -------------
    kwargs : dict
      Keyword arguments which can be loaded by
      trimesh.exchange.load.load_kwargs into a trimesh.Scene
    """
    pass


def parse_mtl(mtl, resolver=None):
    """
    Parse a loaded MTL file.

    Parameters
    -------------
    mtl : str or bytes
      Data from an MTL file
    resolver : trimesh.Resolver
      Fetch assets by name from file system, web, or other

    Returns
    ------------
    mtllibs : list of dict
      Each dict has keys: newmtl, map_Kd, Kd
    """
    pass


def _parse_faces_vectorized(array, columns, sample_line):
    """
    Parse loaded homogeneous (tri/quad) face data in a
    vectorized manner.

    Parameters
    ------------
    array : (n,) int
      Indices in order
    columns : int
      Number of columns in the file
    sample_line : str
      A single line so we can assess the ordering

    Returns
    --------------
    faces : (n, d) int
      Faces in space
    faces_tex : (n, d) int or None
      Texture for each vertex in face
    faces_norm : (n, d) int or None
      Normal index for each vertex in face
    """
    pass


def _parse_faces_fallback(lines):
    """
    Use a slow but more flexible looping method to process
    face lines as a fallback option to faster vectorized methods.

    Parameters
    -------------
    lines : (n,) str
      List of lines with face information

    Returns
    -------------
    faces : (m, 3) int
      Clean numpy array of face triangles
    """
    pass


def _parse_vertices(text):
    """
    Parse raw OBJ text into vertices, vertex normals,
    vertex colors, and vertex textures.

    Parameters
    -------------
    text : str
      Full text of an OBJ file

    Returns
    -------------
    v : (n, 3) float
      Vertices in space
    vn : (m, 3) float or None
      Vertex normals
    vt : (p, 2) float or None
      Vertex texture coordinates
    vc : (n, 3) float or None
      Per-vertex color
    """
    pass


def _group_by_material(face_tuples):
    """
    For chunks of faces split by material group
    the chunks that share the same material.

    Parameters
    ------------
    face_tuples : (n,) list of (material, obj, chunk)
      The data containing faces

    Returns
    ------------
    grouped : (m,) list of (material, obj, chunk)
      Grouped by material
    """
    pass


def _preprocess_faces(text):
    """
    Pre-Process Face Text

    Rather than looking at each line in a loop we're
    going to split lines by directives which indicate
    a new mesh, specifically 'usemtl' and 'o' keys
    search for materials, objects, faces, or groups

    Parameters
    ------------
    text : str
      Raw file

    Returns
    ------------
    triple : (n, 3) tuple
      Tuples of (material, object, data-chunk)
    """
    pass


def export_obj(
    mesh,
    include_normals=None,
    include_color=True,
    include_texture=True,
    return_texture=False,
    write_texture=True,
    resolver=None,
    digits=8,
    mtl_name=None,
    header="https://github.com/mikedh/trimesh",
):
    """
    Export a mesh as a Wavefront OBJ file.
    TODO: scenes with textured meshes

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to be exported
    include_normals : Optional[bool]
      Include vertex normals in export. If None
      will only be included if vertex normals are in cache.
    include_color : bool
      Include vertex color in export
    include_texture : bool
      Include `vt` texture in file text
    return_texture : bool
      If True, return a dict with texture files
    write_texture : bool
      If True and a writable resolver is passed
      write the referenced texture files with resolver
    resolver : None or trimesh.resolvers.Resolver
      Resolver which can write referenced text objects
    digits : int
      Number of digits to include for floating point
    mtl_name : None or str
      If passed, the file name of the MTL file.
    header : str or None
      Header string for top of file or None for no header.

    Returns
    -----------
    export : str
      OBJ format output
    texture : dict
      Contains files that need to be saved in the same
      directory as the exported mesh: {file name : bytes}
    """
    pass


trimesh.exchange.obj.load_obj = load_obj
trimesh.exchange.load.mesh_loaders.update({"obj": load_obj})
