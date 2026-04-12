import hashlib
import os
import platform
import sys
import time
from itertools import combinations

import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch

import genesis as gs

from .misc import get_gel_cache_dir


def load_hmesh(fpath: str):
    pass


def trimesh_to_gelmesh(tmesh):
    pass


def get_gel_path(positions, nodes, sampling):
    pass


def skeletonization(mesh, sampling=True, verbose=False):
    pass


def reduce_graph(G, straight_thresh=10):
    pass


def check_graph(G):
    pass


def compute_graph_attribute(G, G_pos):
    # edge attributes
    pass


def norm_vec(vec):
    pass


def gel_graph_to_nx_graph(gel_graph, use_largest_cc=True):
    pass


def graph_to_tree(G):
    pass


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        # Importing mpl_toolkits is very slow and not used very often. Let's delay import.
        pass


def plot_nxgraph(
    G,
    pos=None,
    plot_arrow=True,
    use_tick_labels=False,
    show=True,
    figax=None,
    node_color=None,
    node_size=100,
    plot_node_num=True,
):
    # Importing matplotlib is very slow and not used very often. Let's delay import.
    pass
