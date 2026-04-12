from numba import *
from numba import types
from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    typeof_impl,
    as_numba_type,
    unbox,
    NativeValue,
)
from numba.core import cgutils
from contextlib import ExitStack
import OpenGL.GL as GL
from OpenGL._bytes import as_8_bit
from OpenGL.GL import GLint, GLuint, GLvoidp, GLvoid, GLfloat, GLsizei, GLboolean, GLenum, GLsizeiptr, GLintptr

import genesis as gs


class GLWrapper:
    def __init__(self):
        self.gl_funcs = {}
        self._wrapper_type = None
        self._wrapper_instance = None

    def load_func(self, func_name, *signature):
        pass

    @property
    def wrapper_type(self):
        pass

    @property
    def wrapper_instance(self):
        pass

    def build_wrapper(self):
        pass
