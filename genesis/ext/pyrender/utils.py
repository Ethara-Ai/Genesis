import numpy as np
from PIL import Image


def format_color_vector(value, length):
    """Format a color vector."""
    if isinstance(value, int):
        value = value / 255.0
    if isinstance(value, float):
        value = np.repeat(value, length)
    if isinstance(value, list) or isinstance(value, tuple):
        value = np.array(value)
    if isinstance(value, np.ndarray):
        value = value.squeeze()
        if np.issubdtype(value.dtype, np.integer):
            value = (value / 255.0).astype(np.float32)
        if value.ndim != 1:
            raise ValueError("Format vector takes only 1-D vectors")
        if length > value.shape[0]:
            value = np.hstack((value, np.ones(length - value.shape[0])))
        elif length < value.shape[0]:
            value = value[:length]
    else:
        raise ValueError("Invalid vector data type")

    return np.ascontiguousarray(value.squeeze(), dtype=np.float32)


def format_color_array(value, shape):
    """Format an array of colors."""
    pass


def format_texture_source(texture, target_channels="RGB"):
    """Format a texture as a float32 np array."""
    pass
