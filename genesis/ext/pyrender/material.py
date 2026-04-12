"""Material properties, conforming to the glTF 2.0 standards as specified in
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-material
and
https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness

Author: Matthew Matl
"""

from abc import ABCMeta

import numpy as np

from .constants import TexFlags
from .utils import format_color_vector, format_texture_source
from .texture import Texture


class Material(metaclass=ABCMeta):
    """Base for standard glTF 2.0 materials.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float or :class:`Texture`, optional
        A tangent space normal map. The texture contains RGB components in
        linear space. Each texel represents the XYZ components of a normal
        vector in tangent space. Red [0 to 255] maps to X [-1 to 1]. Green
        [0 to 255] maps to Y [-1 to 1]. Blue [128 to 255] maps to Z
        [1/255 to 1]. The normal vectors use OpenGL conventions where +X is
        right and +Y is up. +Z points toward the viewer.
    occlusionTexture : (n,n,1) float or :class:`Texture`, optional
        The occlusion map texture. The occlusion values are sampled from the R
        channel. Higher values indicate areas that should receive full indirect
        lighting and lower values indicate no indirect lighting. These values
        are linear. If other channels are present (GBA), they are ignored for
        occlusion calculations.
    emissiveTexture : (n,n,3) float or :class:`Texture`, optional
        The emissive map controls the color and intensity of the light being
        emitted by the material. This texture contains RGB components in sRGB
        color space. If a fourth component (A) is present, it is ignored.
    emissiveFactor : (3,) float, optional
        The RGB components of the emissive color of the material. These values
        are linear. If an emissiveTexture is specified, this value is
        multiplied with the texel values.
    alphaMode : str, optional
        The material's alpha rendering mode enumeration specifying the
        interpretation of the alpha value of the main factor and texture.
        Allowed Values:

        - `"OPAQUE"` The alpha value is ignored and the rendered output is
          fully opaque.
        - `"MASK"` The rendered output is either fully opaque or fully
          transparent depending on the alpha value and the specified alpha
          cutoff value.
        - `"BLEND"` The alpha value is used to composite the source and
          destination areas. The rendered output is combined with the
          background using the normal painting operation (i.e. the Porter
          and Duff over operator).

    alphaCutoff : float, optional
        Specifies the cutoff threshold when in MASK mode. If the alpha value is
        greater than or equal to this value then it is rendered as fully
        opaque, otherwise, it is rendered as fully transparent.
        A value greater than 1.0 will render the entire material as fully
        transparent. This value is ignored for other modes.
    doubleSided : bool, optional
        Specifies whether the material is double sided. When this value is
        false, back-face culling is enabled. When this value is true,
        back-face culling is disabled and double sided lighting is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal
        per vertex and face indexing.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    """

    def __init__(
        self,
        name=None,
        normalTexture=None,
        occlusionTexture=None,
        emissiveTexture=None,
        emissiveFactor=None,
        alphaMode=None,
        alphaCutoff=None,
        doubleSided=False,
        smooth=True,
        wireframe=False,
    ):
        # Set defaults
        if alphaMode is None:
            alphaMode = "OPAQUE"

        if alphaCutoff is None:
            alphaCutoff = 0.5

        if emissiveFactor is None:
            emissiveFactor = np.zeros((3,), dtype=np.float32)

        self.name = name
        self.normalTexture = normalTexture
        self.occlusionTexture = occlusionTexture
        self.emissiveTexture = emissiveTexture
        self.emissiveFactor = emissiveFactor
        self.alphaMode = alphaMode
        self.alphaCutoff = alphaCutoff
        self.doubleSided = doubleSided
        self.smooth = smooth
        self.wireframe = wireframe

        self._tex_flags = None

    @property
    def name(self):
        """str : The user-defined name of this object."""
        pass

    @name.setter
    def name(self, value):
        pass

    @property
    def normalTexture(self):
        """(n,n,3) float or :class:`Texture` : The tangent-space normal map."""
        pass

    @normalTexture.setter
    def normalTexture(self, value):
        # TODO TMP
        pass

    @property
    def occlusionTexture(self):
        """(n,n,1) float or :class:`Texture` : The ambient occlusion map."""
        pass

    @occlusionTexture.setter
    def occlusionTexture(self, value):
        pass

    @property
    def emissiveTexture(self):
        """(n,n,3) float or :class:`Texture` : The emission map."""
        pass

    @emissiveTexture.setter
    def emissiveTexture(self, value):
        pass

    @property
    def emissiveFactor(self):
        """(3,) float : Base multiplier for emission colors."""
        pass

    @emissiveFactor.setter
    def emissiveFactor(self, value):
        pass

    @property
    def alphaMode(self):
        """str : The mode for blending."""
        pass

    @alphaMode.setter
    def alphaMode(self, value):
        pass

    @property
    def alphaCutoff(self):
        """float : The cutoff threshold in MASK mode."""
        pass

    @alphaCutoff.setter
    def alphaCutoff(self, value):
        pass

    @property
    def doubleSided(self):
        """bool : Whether the material is double-sided."""
        pass

    @doubleSided.setter
    def doubleSided(self, value):
        pass

    @property
    def smooth(self):
        """bool : Whether to render the mesh smoothly by
        interpolating vertex normals.
        """
        pass

    @smooth.setter
    def smooth(self, value):
        pass

    @property
    def wireframe(self):
        """bool : Whether to render the mesh in wireframe mode."""
        pass

    @wireframe.setter
    def wireframe(self, value):
        pass

    @property
    def is_transparent(self):
        """bool : If True, the object is partially transparent."""
        pass

    @property
    def tex_flags(self):
        """int : Texture availability flags."""
        pass

    @property
    def textures(self):
        """list of :class:`Texture` : The textures associated with this
        material.
        """
        pass

    def _compute_transparency(self):
        pass

    def _compute_tex_flags(self):
        pass

    def _compute_textures(self):
        pass

    def _format_texture(self, texture, target_channels="RGB"):
        """Format a texture as a float32 np array."""
        pass


class MetallicRoughnessMaterial(Material):
    """A material based on the metallic-roughness material model from
    Physically-Based Rendering (PBR) methodology.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float or :class:`Texture`, optional
        A tangent space normal map. The texture contains RGB components in
        linear space. Each texel represents the XYZ components of a normal
        vector in tangent space. Red [0 to 255] maps to X [-1 to 1]. Green
        [0 to 255] maps to Y [-1 to 1]. Blue [128 to 255] maps to Z
        [1/255 to 1]. The normal vectors use OpenGL conventions where +X is
        right and +Y is up. +Z points toward the viewer.
    occlusionTexture : (n,n,1) float or :class:`Texture`, optional
        The occlusion map texture. The occlusion values are sampled from the R
        channel. Higher values indicate areas that should receive full indirect
        lighting and lower values indicate no indirect lighting. These values
        are linear. If other channels are present (GBA), they are ignored for
        occlusion calculations.
    emissiveTexture : (n,n,3) float or :class:`Texture`, optional
        The emissive map controls the color and intensity of the light being
        emitted by the material. This texture contains RGB components in sRGB
        color space. If a fourth component (A) is present, it is ignored.
    emissiveFactor : (3,) float, optional
        The RGB components of the emissive color of the material. These values
        are linear. If an emissiveTexture is specified, this value is
        multiplied with the texel values.
    alphaMode : str, optional
        The material's alpha rendering mode enumeration specifying the
        interpretation of the alpha value of the main factor and texture.
        Allowed Values:

        - `"OPAQUE"` The alpha value is ignored and the rendered output is
          fully opaque.
        - `"MASK"` The rendered output is either fully opaque or fully
          transparent depending on the alpha value and the specified alpha
          cutoff value.
        - `"BLEND"` The alpha value is used to composite the source and
          destination areas. The rendered output is combined with the
          background using the normal painting operation (i.e. the Porter
          and Duff over operator).

    alphaCutoff : float, optional
        Specifies the cutoff threshold when in MASK mode. If the alpha value is
        greater than or equal to this value then it is rendered as fully
        opaque, otherwise, it is rendered as fully transparent.
        A value greater than 1.0 will render the entire material as fully
        transparent. This value is ignored for other modes.
    doubleSided : bool, optional
        Specifies whether the material is double sided. When this value is
        false, back-face culling is enabled. When this value is true,
        back-face culling is disabled and double sided lighting is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal
        per vertex and face indexing.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    baseColorFactor : (4,) float, optional
        The RGBA components of the base color of the material. The fourth
        component (A) is the alpha coverage of the material. The alphaMode
        property specifies how alpha is interpreted. These values are linear.
        If a baseColorTexture is specified, this value is multiplied with the
        texel values.
    baseColorTexture : (n,n,4) float or :class:`Texture`, optional
        The base color texture. This texture contains RGB(A) components in sRGB
        color space. The first three components (RGB) specify the base color of
        the material. If the fourth component (A) is present, it represents the
        alpha coverage of the material. Otherwise, an alpha of 1.0 is assumed.
        The alphaMode property specifies how alpha is interpreted.
        The stored texels must not be premultiplied.
    metallicFactor : float
        The metalness of the material. A value of 1.0 means the material is a
        metal. A value of 0.0 means the material is a dielectric. Values in
        between are for blending between metals and dielectrics such as dirty
        metallic surfaces. This value is linear. If a metallicRoughnessTexture
        is specified, this value is multiplied with the metallic texel values.
    roughnessFactor : float
        The roughness of the material. A value of 1.0 means the material is
        completely rough. A value of 0.0 means the material is completely
        smooth. This value is linear. If a metallicRoughnessTexture is
        specified, this value is multiplied with the roughness texel values.
    metallicRoughnessTexture : (n,n,2) float or :class:`Texture`, optional
        The metallic-roughness texture. The metalness values are sampled from
        the B channel. The roughness values are sampled from the G channel.
        These values are linear. If other channels are present (R or A), they
        are ignored for metallic-roughness calculations.
    """

    def __init__(
        self,
        name=None,
        normalTexture=None,
        occlusionTexture=None,
        emissiveTexture=None,
        emissiveFactor=None,
        alphaMode=None,
        alphaCutoff=None,
        doubleSided=False,
        smooth=True,
        wireframe=False,
        baseColorFactor=None,
        baseColorTexture=None,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        metallicRoughnessTexture=None,
    ):
        super().__init__(
            name=name,
            normalTexture=normalTexture,
            occlusionTexture=occlusionTexture,
            emissiveTexture=emissiveTexture,
            emissiveFactor=emissiveFactor,
            alphaMode=alphaMode,
            alphaCutoff=alphaCutoff,
            doubleSided=doubleSided,
            smooth=smooth,
            wireframe=wireframe,
        )

        # Set defaults
        if baseColorFactor is None:
            baseColorFactor = np.ones((4,), dtype=np.float32)

        self.baseColorFactor = baseColorFactor
        self.baseColorTexture = baseColorTexture
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor
        self.metallicRoughnessTexture = metallicRoughnessTexture

    @property
    def baseColorFactor(self):
        """(4,) float or :class:`Texture` : The RGBA base color multiplier."""
        pass

    @baseColorFactor.setter
    def baseColorFactor(self, value):
        pass

    @property
    def baseColorTexture(self):
        """(n,n,4) float or :class:`Texture` : The diffuse texture."""
        pass

    @baseColorTexture.setter
    def baseColorTexture(self, value):
        pass

    @property
    def metallicFactor(self):
        """float : The metalness of the material."""
        pass

    @metallicFactor.setter
    def metallicFactor(self, value):
        pass

    @property
    def roughnessFactor(self):
        """float : The roughness of the material."""
        pass

    @roughnessFactor.setter
    def roughnessFactor(self, value):
        pass

    @property
    def metallicRoughnessTexture(self):
        """(n,n,2) float or :class:`Texture` : The metallic-roughness texture."""
        pass

    @metallicRoughnessTexture.setter
    def metallicRoughnessTexture(self, value):
        pass

    def _compute_tex_flags(self):
        pass

    def _compute_transparency(self):
        pass

    def _compute_textures(self):
        pass


class SpecularGlossinessMaterial(Material):
    """A material based on the specular-glossiness material model from
    Physically-Based Rendering (PBR) methodology.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float or :class:`Texture`, optional
        A tangent space normal map. The texture contains RGB components in
        linear space. Each texel represents the XYZ components of a normal
        vector in tangent space. Red [0 to 255] maps to X [-1 to 1]. Green
        [0 to 255] maps to Y [-1 to 1]. Blue [128 to 255] maps to Z
        [1/255 to 1]. The normal vectors use OpenGL conventions where +X is
        right and +Y is up. +Z points toward the viewer.
    occlusionTexture : (n,n,1) float or :class:`Texture`, optional
        The occlusion map texture. The occlusion values are sampled from the R
        channel. Higher values indicate areas that should receive full indirect
        lighting and lower values indicate no indirect lighting. These values
        are linear. If other channels are present (GBA), they are ignored for
        occlusion calculations.
    emissiveTexture : (n,n,3) float or :class:`Texture`, optional
        The emissive map controls the color and intensity of the light being
        emitted by the material. This texture contains RGB components in sRGB
        color space. If a fourth component (A) is present, it is ignored.
    emissiveFactor : (3,) float, optional
        The RGB components of the emissive color of the material. These values
        are linear. If an emissiveTexture is specified, this value is
        multiplied with the texel values.
    alphaMode : str, optional
        The material's alpha rendering mode enumeration specifying the
        interpretation of the alpha value of the main factor and texture.
        Allowed Values:

        - `"OPAQUE"` The alpha value is ignored and the rendered output is
          fully opaque.
        - `"MASK"` The rendered output is either fully opaque or fully
          transparent depending on the alpha value and the specified alpha
          cutoff value.
        - `"BLEND"` The alpha value is used to composite the source and
          destination areas. The rendered output is combined with the
          background using the normal painting operation (i.e. the Porter
          and Duff over operator).

    alphaCutoff : float, optional
        Specifies the cutoff threshold when in MASK mode. If the alpha value is
        greater than or equal to this value then it is rendered as fully
        opaque, otherwise, it is rendered as fully transparent.
        A value greater than 1.0 will render the entire material as fully
        transparent. This value is ignored for other modes.
    doubleSided : bool, optional
        Specifies whether the material is double sided. When this value is
        false, back-face culling is enabled. When this value is true,
        back-face culling is disabled and double sided lighting is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal
        per vertex and face indexing.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    diffuseFactor : (4,) float
        The RGBA components of the reflected diffuse color of the material.
        Metals have a diffuse value of [0.0, 0.0, 0.0]. The fourth component
        (A) is the opacity of the material. The values are linear.
    diffuseTexture : (n,n,4) float or :class:`Texture`, optional
        The diffuse texture. This texture contains RGB(A) components of the
        reflected diffuse color of the material in sRGB color space. If the
        fourth component (A) is present, it represents the alpha coverage of
        the material. Otherwise, an alpha of 1.0 is assumed.
        The alphaMode property specifies how alpha is interpreted.
        The stored texels must not be premultiplied.
    specularFactor : (3,) float
        The specular RGB color of the material. This value is linear.
    glossinessFactor : float
        The glossiness or smoothness of the material. A value of 1.0 means the
        material has full glossiness or is perfectly smooth. A value of 0.0
        means the material has no glossiness or is perfectly rough. This value
        is linear.
    specularGlossinessTexture : (n,n,4) or :class:`Texture`, optional
        The specular-glossiness texture is a RGBA texture, containing the
        specular color (RGB) in sRGB space and the glossiness value (A) in
        linear space.
    """

    def __init__(
        self,
        name=None,
        normalTexture=None,
        occlusionTexture=None,
        emissiveTexture=None,
        emissiveFactor=None,
        alphaMode=None,
        alphaCutoff=None,
        doubleSided=False,
        smooth=True,
        wireframe=False,
        diffuseFactor=None,
        diffuseTexture=None,
        specularFactor=None,
        glossinessFactor=1.0,
        specularGlossinessTexture=None,
    ):
        super().__init__(
            name=name,
            normalTexture=normalTexture,
            occlusionTexture=occlusionTexture,
            emissiveTexture=emissiveTexture,
            emissiveFactor=emissiveFactor,
            alphaMode=alphaMode,
            alphaCutoff=alphaCutoff,
            doubleSided=doubleSided,
            smooth=smooth,
            wireframe=wireframe,
        )

        # Set defaults
        if diffuseFactor is None:
            diffuseFactor = np.ones((4,), dtype=np.float32)
        if specularFactor is None:
            specularFactor = np.ones((3,), dtype=np.float32)

        self.diffuseFactor = diffuseFactor
        self.diffuseTexture = diffuseTexture
        self.specularFactor = specularFactor
        self.glossinessFactor = glossinessFactor
        self.specularGlossinessTexture = specularGlossinessTexture

    @property
    def diffuseFactor(self):
        """(4,) float : The diffuse base color."""
        pass

    @diffuseFactor.setter
    def diffuseFactor(self, value):
        pass

    @property
    def diffuseTexture(self):
        """(n,n,4) float or :class:`Texture` : The diffuse map."""
        pass

    @diffuseTexture.setter
    def diffuseTexture(self, value):
        pass

    @property
    def specularFactor(self):
        """(3,) float : The specular color of the material."""
        pass

    @specularFactor.setter
    def specularFactor(self, value):
        pass

    @property
    def glossinessFactor(self):
        """float : The glossiness of the material."""
        pass

    @glossinessFactor.setter
    def glossinessFactor(self, value):
        pass

    @property
    def specularGlossinessTexture(self):
        """(n,n,4) or :class:`Texture` : The specular-glossiness texture."""
        pass

    @specularGlossinessTexture.setter
    def specularGlossinessTexture(self, value):
        pass

    def _compute_tex_flags(self):
        pass

    def _compute_transparency(self):
        pass

    def _compute_textures(self):
        pass
