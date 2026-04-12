import numpy as np
from PIL import Image
from pxr import Usd, UsdShade

import genesis as gs
from genesis.utils import mesh as mu


CS_ENCODE = {
    "raw": "linear",
    "sRGB": "srgb",
    "auto": None,
    "": None,
}


def get_input_attribute_value(shader: UsdShade.Shader, input_name, input_type=None):
    pass


def parse_component(shader: UsdShade.Shader, component_name: str, component_encode: str):
    pass


def get_shader(prim: Usd.Prim, output_name: str) -> UsdShade.Shader:
    pass


def parse_preview_surface(prim: Usd.Prim, output_name):
    pass


def parse_material_preview_surface(material: UsdShade.Material) -> tuple[dict, str]:
    """Find the preview surface for a material."""
    surface_outputs = material.GetSurfaceOutputs()
    candidates_surfaces = []
    material_dict, uv_name = {}, "st"
    for surface_output in surface_outputs:
        if not surface_output.HasConnectedSource():
            continue
        surface_output_connectable, surface_output_name, _ = surface_output.GetConnectedSource()
        surface_output_connect = surface_output_connectable.GetPrim()
        surface_shader = get_shader(surface_output_connect, "surface")
        surface_shader_implement = surface_shader.GetImplementationSource()
        surface_shader_id = surface_shader.GetShaderId()
        if surface_shader_implement == "id" and surface_shader_id == "UsdPreviewSurface":
            material_dict, uv_name = parse_preview_surface(surface_output_connect, surface_output_name)
            break
        candidates_surfaces.append((surface_shader.GetPath(), surface_shader_id, surface_shader_implement))

    if not material_dict:
        candidates_str = "\n".join(
            f"\tShader at {shader_path} with implement {shader_impl} and ID {shader_id}."
            for shader_path, shader_id, shader_impl in candidates_surfaces
        )
        gs.logger.debug(f"Material require baking:\n{candidates_str}")
    return material_dict, uv_name
