import os
from functools import cached_property
from typing import Annotated, Sequence, Literal, Iterable, cast

import numpy as np
from PIL import Image
from pydantic import model_validator, computed_field, BeforeValidator, Field

import genesis as gs
import genesis.utils.mesh as mu
from genesis.typing import LaxUnitIntervalArrayType, LaxFArrayType, UnitIntervalArrayType, NDArrayType

from .options import Options


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".hdr", ".exr")
HDR_EXTENSIONS = (".hdr", ".exr")


class Texture(Options):
    """
    Base class for Genesis's texture objects.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    def __init__(self, **data):
        super().__init__(**data)

    def check_dim(self, dim: int) -> "Texture | None":
        raise NotImplementedError

    def check_simplify(self) -> "Texture":
        raise NotImplementedError

    def apply_cutoff(self, cutoff: float) -> None:
        raise NotImplementedError

    @cached_property
    def is_black(self) -> bool:
        raise NotImplementedError

    @cached_property
    def requires_uv(self) -> bool:
        raise NotImplementedError


class ColorTexture(Texture):
    """
    A texture that consists of a single color.

    Parameters
    ----------
    color : list of float
        A list of color values, stored as tuple, supporting any number of channels within the range [0.0, 1.0].
        Default is (1.0, 1.0, 1.0).
    """

    color: LaxFArrayType = (1.0, 1.0, 1.0)

    def check_dim(self, dim: int) -> Texture | None:
        if len(self.color) > dim:
            self.color, res = self.color[:dim], self.color[dim]
            return ColorTexture(color=res)
        return None

    def check_simplify(self) -> "ColorTexture":
        return self

    def apply_cutoff(self, cutoff: float) -> None:
        if cutoff is None:
            return
        self.color = tuple(1.0 if c >= cutoff else 0.0 for c in self.color)

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        pass

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        pass


class ImageTexture(Texture):
    """
    A texture with a texture map (image).

    Parameters
    ----------
    image_path : str, optional
        Path to the image file.
    image_array : np.ndarray, optional
        Image array.
    image_color : float or list of float, optional
        The factor that will be multiplied with the base color, stored as tuple. Default is None.
    encoding : str, optional
        The encoding way of the image. Possible values are ['srgb', 'linear']. Default is 'srgb'.

        - 'srgb': Encoding of some RGB images.
        - 'linear': All generic images, such as opacity, roughness and normal, should be encoded with 'linear'.
    """

    image_path: str | None = None
    image_array: NDArrayType | None
    image_color: UnitIntervalArrayType
    encoding: Annotated[Literal["srgb", "linear"], BeforeValidator(lambda e: str(e).lower())] = "srgb"

    def __init__(
        self,
        *,
        image_path: str | None = None,
        image_array: np.ndarray | None = None,
        image_color: LaxUnitIntervalArrayType | float | None = None,
        encoding: str = "srgb",
        **data,
    ) -> None:
        super().__init__(
            image_path=image_path,
            image_array=image_array,
            image_color=image_color,
            encoding=encoding,
            **data,
        )

    @model_validator(mode="before")
    @classmethod
    def _validate_and_load(cls, data: dict) -> dict:
        pass

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        pass

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        pass

    @computed_field
    @property
    def channel(self) -> int:
        pass

    @computed_field
    @cached_property
    def mean_color(self) -> NDArrayType:
        pass

    def check_dim(self, dim: int) -> Texture | None:
        if self.image_array is not None:
            if self.channel > dim:
                self.image_array, res_array = self.image_array[:, :, :dim], self.image_array[:, :, dim]
                self.image_color, res_color = self.image_color[:dim], self.image_color[dim:]
                return ImageTexture(image_array=res_array, image_color=res_color, encoding="linear").check_simplify()
        return None

    def check_simplify(self) -> "ImageTexture | ColorTexture":
        if self.image_array is None:
            return self
        max_color = np.max(self.image_array, axis=(0, 1))
        min_color = np.min(self.image_array, axis=(0, 1))
        if np.all(min_color == max_color):
            return ColorTexture(color=max_color.reshape(-1) / 255.0 * self.image_color)
        return self

    def apply_cutoff(self, cutoff):
        if cutoff is None or self.image_array is None:  # Cutoff does not apply on image file.
            return
        self.image_array = np.where(self.image_array >= 255.0 * cutoff, 255, 0).astype(np.uint8)


class BatchTexture(Texture):
    """
    A batch of textures for batch rendering.

    Parameters
    ----------
    textures : List[Optional[Texture]]
        List of textures.
    """

    textures: Annotated[list[Texture | None], BeforeValidator(list)] = Field(default_factory=list)

    @staticmethod
    def from_images(
        image_paths: Sequence[str] | None = None,
        image_folder: str | None = None,
        image_arrays: Sequence[np.ndarray] | None = None,
        image_colors: Sequence[float] | Sequence[Sequence[float] | None] | None = None,
        encoding: Literal["srgb", "linear"] = "srgb",
    ) -> "BatchTexture":
        """
        Create a batch texture from images.

        Parameters
        ----------
        image_paths : List[str], optional
            List of paths to the image files.
        image_folder : str, optional
            Path to the image folder.
        image_arrays : List[np.ndarray], optional
            List of image arrays.
        image_colors : List[Union[float, List[float]]], optional
            List of color factors that will be multiplied with the base color, stored as tuple. Default is None.
        encoding : str, optional
            The encoding way of the image. Possible values are ['srgb', 'linear']. Default is 'srgb'.

            - 'srgb': Encoding of some RGB images.
            - 'linear': All generic images, such as opacity, roughness and normal, should be encoded with 'linear'.
        """
        pass

    @staticmethod
    def from_colors(
        colors: Sequence[Sequence[float]],
    ) -> "BatchTexture":
        """
        Create a batch texture from colors.

        Parameters
        ----------
        colors : List[List[float]]
            List of colors.
        """
        pass

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        pass

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        pass

    def check_dim(self, dim: int) -> "BatchTexture":
        return BatchTexture(
            textures=[texture.check_dim(dim) if texture is not None else None for texture in self.textures]
        ).check_simplify()

    def check_simplify(self) -> "BatchTexture":
        self.textures = [texture.check_simplify() if texture is not None else None for texture in self.textures]
        return self

    def apply_cutoff(self, cutoff: float) -> None:
        for texture in self.textures:
            if texture is not None:
                texture.apply_cutoff(cutoff)

    def merge(self, other: "BatchTexture") -> None:
        self.textures.extend(other.textures)
