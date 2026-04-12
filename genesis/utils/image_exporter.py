import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, Executor
from functools import partial

import torch
import numpy as np

import genesis as gs
from genesis.constants import IMAGE_TYPE
from genesis.utils.misc import tensor_to_array


def as_grayscale_image(
    data: np.ndarray, clip_max: float | None = None, enable_log_scale: bool = False, black_to_white: bool = False
) -> np.ndarray:
    """Convert a batched 2D array of numeric dtype as 8 bits single channel (grayscale) image array for visualization.

    Internally, this method clips non-finite values, optionally applies log scaling (i.e. `log(1.0 + data)`), then
    normalizes values between 0.0 and 1.0, to finally convert to grayscale.

    Parameters
    ----------
    data : ndarray [(N x) H x W]
        The data to normalize as a batched 2D array with any numeric dtype.
    clip_max : float, optional
        The maximum valid value if any. Default to None.
    enable_log_scale: bool, optional
        Wether to apply log scaling before normalization. Default to False.
    black_to_white: bool, optional
        Whether the color is transitioning from black to white as value increases or conversely. Default to False.
    """
    # Cast data to float32
    data_float = data.astype(np.float32)

    # Clip data, with special handling for non-finite values only if necessary for efficiency
    valid_mask = np.isfinite(data_float)
    if np.all(valid_mask):
        data_min = np.min(data_float, axis=(-2, -1), keepdims=True)
        data_max = np.max(data_float, axis=(-2, -1), keepdims=True)
    else:
        data_min = np.min(data_float, axis=(-2, -1), keepdims=True, initial=float("+inf"), where=valid_mask)
        data_max = np.max(data_float, axis=(-2, -1), keepdims=True, initial=float("-inf"), where=valid_mask)
    data_min = np.maximum(data_min, 0.0)
    if clip_max is not None:
        data_max = np.minimum(data_max, clip_max)
    data_float = np.clip(data_float, data_min, data_max)

    # Apply log scaling if requested
    if enable_log_scale:
        data_float = np.log(1.0 + data_float)

    # Normalize values between 0.0 and 1.0
    data_delta = data_max - data_min
    data_normalized = data_float - data_min if black_to_white else data_max - data_float
    np.divide(data_normalized, data_delta, where=data_delta > gs.EPS, out=data_normalized)

    # Discretize as unsigned int8
    return (data_normalized * 255.0).astype(np.uint8)


class FrameImageExporter:
    """
    This class enables exporting images from multiple cameras and environments in batch and in parallel, unlike
    `Camera.(start|stop)_recording` API, which only allows for exporting images from a single camera and environment.
    """

    def __init__(self, export_dir: str, depth_clip_max: float = 100.0, enable_depth_log_scale: bool = False):
        self.depth_clip_max = depth_clip_max
        self.enable_depth_log_scale = enable_depth_log_scale
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_frame_all_cameras(
        self,
        i_step: int,
        cameras_idx: Iterable | None = None,
        rgb: Sequence[np.ndarray] | None = None,
        depth: Sequence[np.ndarray] | None = None,
        segmentation: Sequence[np.ndarray] | None = None,
        normal: Sequence[np.ndarray] | None = None,
    ):
        """
        Export multiple frames from different cameras and environments in parrallel as PNG files.

        Note
        ----
        All specified sequences of images must have the same length.

        Parameters
        ----------
        i_step : int
            The current step index.
        cameras_idx: Iterable, optional
            Sequence of indices of cameras to export. If None, all cameras are exported.
        rgb: Sequence[ndarray[np.floating]], optional
            RGB image is a sequence of arrays of shape ([n_envs,] H, W, 3).
        depth: Sequence[ndarray[np.floating]], optional
            Depth image is a sequence of arrays of shape ([n_envs,] H, W).
        segmentation: Sequence[ndarray[np.integer]], optional
            Segmentation image is a sequence of arrays of shape ([n_envs,] H, W).
        normal: Sequence[ndarray[np.floating]], optional
            Normal image is a sequence of arrays of shape ([n_envs,] H, W, 3).
        """
        pass

    def export_frame_single_camera(
        self,
        i_step,
        i_cam,
        rgb=None,
        depth=None,
        segmentation=None,
        normal=None,
        *,
        compress_level: int | None = None,
        executor: Executor | None = None,
    ):
        """
        Export multiple frames from a single camera but different environments in parrallel as PNG files.

        Parameters
        ----------
        i_step: int
            The current step index.
        i_cam: int
            The index of the camera.
        rgb: ndarray[np.floating], optional
            RGB image array of shape ([n_envs,] H, W, 3).
        depth: ndarray[np.floating], optional
            Depth image array of shape ([n_envs,] H, W).
        segmentation: ndarray[np.integer], optional
            Segmentation image array of shape ([n_envs,] H, W).
        normal: ndarray[np.floating], optional
            Normal image array of shape ([n_envs,] H, W, 3).
        compress_level: int, optional
            Compression level when exporting images as PNG. Default to 3.
        executor: Executor, optional
            Executor to which I/O bounded jobs (saving to PNG) will be submitted. A local executor will be instantiated
            if none is provided.
        """
        pass
