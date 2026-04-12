"""Font texture loader and processor.

Author: Matthew Matl
"""

import freetype
import numpy as np
import os

import OpenGL
from OpenGL.GL import *

from .constants import TextAlign, FLOAT_SZ
from .texture import Texture
from .sampler import Sampler


class FontCache(object):
    """A cache for fonts."""

    def __init__(self, font_dir=None):
        self._font_cache = {}
        self.font_dir = font_dir
        if self.font_dir is None:
            base_dir, _ = os.path.split(os.path.realpath(__file__))
            self.font_dir = os.path.join(base_dir, "fonts")

    def get_font(self, font_name, font_pt):
        # If it's a file, load it directly, else, try to load from font dir.
        pass

    def clear(self):
        for key in self._font_cache:
            try:
                self._font_cache[key].delete()
            except OpenGL.error.GLError:
                pass
        self._font_cache.clear()


class Character(object):
    """A single character, with its texture and attributes."""

    def __init__(self, texture, size, bearing, advance):
        self.texture = texture
        self.size = size
        self.bearing = bearing
        self.advance = advance


class Font(object):
    """A font object.

    Parameters
    ----------
    font_file : str
        The file to load the font from.
    font_pt : int
        The height of the font in pixels.
    """

    def __init__(self, font_file, font_pt=40):
        self.font_file = font_file
        self.font_pt = int(font_pt)
        self._face = freetype.Face(font_file)
        self._face.set_pixel_sizes(0, font_pt)
        self._character_map = {}

        for i in range(0, 128):
            face = self._face

            if "PYTEST_VERSION" in os.environ:
                # Bit-exact, ugly but deterministic
                face.load_char(
                    chr(i),
                    freetype.FT_LOAD_RENDER
                    | freetype.FT_LOAD_MONOCHROME
                    | freetype.FT_LOAD_NO_HINTING
                    | freetype.FT_LOAD_NO_AUTOHINT,
                )

                # FreeType mono bitmaps are 1 bit per pixel, packed
                bitmap = face.glyph.bitmap
                buf = np.asarray(bitmap.buffer, dtype=np.uint8)
                bits = np.unpackbits(buf).reshape(bitmap.rows, bitmap.pitch * 8)
                src = bits[:, : bitmap.width].astype(np.float32)
                if src.size == 1:
                    src = np.zeros((0, 0), dtype=np.float32)

                sampler = Sampler(
                    magFilter=GL_NEAREST,
                    minFilter=GL_NEAREST,
                    wrapS=GL_CLAMP_TO_EDGE,
                    wrapT=GL_CLAMP_TO_EDGE,
                )
            else:
                # Normal, pretty rendering
                face.load_char(chr(i))

                bitmap = face.glyph.bitmap
                src = np.asarray(bitmap.buffer, dtype=np.float32) / 255.0
                src = src.reshape((bitmap.rows, bitmap.width))

                sampler = Sampler(
                    magFilter=GL_LINEAR,
                    minFilter=GL_LINEAR,
                    wrapS=GL_CLAMP_TO_EDGE,
                    wrapT=GL_CLAMP_TO_EDGE,
                )

            tex = Texture(
                sampler=sampler,
                source=src,
                source_channels="R",
            )

            character = Character(
                texture=tex,
                size=np.array([bitmap.width, bitmap.rows], dtype=np.int32),
                bearing=np.array([face.glyph.bitmap_left, face.glyph.bitmap_top], dtype=np.int32),
                advance=face.glyph.advance.x,
            )

            self._character_map[chr(i)] = character

        self._vbo = None
        self._vao = None

    @property
    def font_file(self):
        """str : The file the font was loaded from."""
        pass

    @font_file.setter
    def font_file(self, value):
        pass

    @property
    def font_pt(self):
        """int : The height of the font in pixels."""
        pass

    @font_pt.setter
    def font_pt(self, value):
        pass

    def _add_to_context(self):
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * FLOAT_SZ, ctypes.c_void_p(0))
        glBindVertexArray(0)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        for c in self._character_map:
            ch = self._character_map[c]
            if not ch.texture._in_context():
                ch.texture._add_to_context()

    def _remove_from_context(self):
        for c in self._character_map:
            ch = self._character_map[c]
            ch.texture.delete()
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None
            self._vbo = None

    def _in_context(self):
        return self._vao is not None

    def _bind(self):
        glBindVertexArray(self._vao)

    def _unbind(self):
        glBindVertexArray(0)

    def delete(self):
        self._unbind()
        self._remove_from_context()

    def render_string(self, text, x, y, scale=1.0, align=TextAlign.BOTTOM_LEFT):
        """Render a string to the current view buffer.

        Note
        ----
        Assumes correct shader program already bound w/ uniforms set.

        Parameters
        ----------
        text : str
            The text to render.
        x : int
            Horizontal pixel location of text.
        y : int
            Vertical pixel location of text.
        scale : int
            Scaling factor for text.
        align : int
            One of the TextAlign options which specifies where the ``x``
            and ``y`` parameters lie on the text. For example,
            :attr:`.TextAlign.BOTTOM_LEFT` means that ``x`` and ``y`` indicate
            the position of the bottom-left corner of the textbox.
        """
        pass
