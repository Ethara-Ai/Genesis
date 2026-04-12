import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING, ForwardRef, get_args, get_origin

import torch

import genesis as gs
from genesis.options.sensors.options import SensorOptions
from genesis.utils.ring_buffer import TensorRingBuffer

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext

    from .base_sensor import Sensor, SharedSensorMetadata


class SensorManager:
    # Maps sensor options class -> sensor class for runtime dispatch.
    SENSOR_TYPES_MAP: dict[type[SensorOptions], type["Sensor"]] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[type["Sensor"], SharedSensorMetadata | None] = {}
        self._ground_truth_cache: dict[type[torch.dtype], torch.Tensor] = {}
        self._cache: dict[type[torch.dtype], torch.Tensor] = {}
        self._buffered_data: dict[type[torch.dtype], TensorRingBuffer] = {}
        self._cache_slices_by_type: dict[type["Sensor"], slice] = {}
        self._should_update_cache_by_type: dict[type["Sensor"], bool] = {}
        self._is_last_cache_cloned: dict[tuple[bool, type[torch.dtype]], bool] = {}
        self._cloned_cache: dict[tuple[bool, type[torch.dtype]], torch.Tensor] = {}

    def create_sensor(self, sensor_options: "SensorOptions") -> "Sensor":
        pass

    @staticmethod
    def _resolve_sensor_cls(options_cls: type) -> type["Sensor"]:
        """Resolve the sensor class for the given options class, triggering lazy discovery if needed."""
        pass

    def build(self):
        max_buffer_len = 0
        cache_size_per_dtype = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()

            for is_ground_truth in (False, True):
                key = (is_ground_truth, dtype)
                self._is_last_cache_cloned[key] = False
                self._cloned_cache[key] = torch.tensor([], dtype=dtype, device=gs.device)

            cache_size_per_dtype.setdefault(dtype, 0)
            cls_cache_start_idx = cache_size_per_dtype[dtype]

            update_ground_truth_only = True
            for sensor in sensors:
                update_ground_truth_only &= sensor._options.update_ground_truth_only
                sensor._cache_idx = cache_size_per_dtype[dtype]
                cache_size_per_dtype[dtype] += sensor._cache_size
                max_buffer_len = max(max_buffer_len, sensor._delay_ts + 1)
            self._should_update_cache_by_type[sensor_cls] = not update_ground_truth_only

            cls_cache_end_idx = cache_size_per_dtype[dtype]
            self._cache_slices_by_type[sensor_cls] = slice(cls_cache_start_idx, cls_cache_end_idx)

        for dtype in cache_size_per_dtype.keys():
            cache_shape = (self._sim._B, cache_size_per_dtype[dtype])
            # Ground truth cache is stored transposed (cols, B) so that per-class row slices are C-contiguous,
            # which is required for kernel writes. The cache and ring buffer stay (B, cols) since they only
            # receive data via .copy_() / torch.lerp which handle non-contiguous targets.
            gt_cache_shape = (cache_size_per_dtype[dtype], self._sim._B)
            self._ground_truth_cache[dtype] = torch.zeros(gt_cache_shape, dtype=dtype, device=gs.device)
            self._cache[dtype] = torch.zeros(cache_shape, dtype=dtype, device=gs.device)
            self._buffered_data[dtype] = TensorRingBuffer(max_buffer_len, cache_shape, dtype=dtype)

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()
            for sensor in sensors:
                sensor.build()
                sensor._is_built = True

    def destroy(self):
        for sensors_metadata in self._sensors_metadata.values():
            if sensors_metadata is not None:
                sensors_metadata.destroy()
        self._sensors_metadata.clear()
        self._sensors_by_type.clear()

    def reset(self, envs_idx=None):
        if not self._sensors_by_type:
            return

        envs_idx = self._sim._scene._sanitize_envs_idx(envs_idx)

        for dtype in self._buffered_data.keys():
            self._ground_truth_cache[dtype][:, envs_idx] = 0.0
            self._cache[dtype][envs_idx] = 0.0
            self._buffered_data[dtype].buffer[:, envs_idx] = 0.0
            for is_ground_truth in (False, True):
                key = (is_ground_truth, dtype)
                self._is_last_cache_cloned[key] = False
                self._cloned_cache[key] = torch.tensor([], dtype=dtype, device=gs.device)

        for sensor_cls in self._sensors_by_type.keys():
            dtype = sensor_cls._get_cache_dtype()
            cache_slice = self._cache_slices_by_type[sensor_cls]
            sensor_cls.reset(self._sensors_metadata[sensor_cls], self._ground_truth_cache[dtype][cache_slice], envs_idx)

    def step(self):
        for buffered_data in self._buffered_data.values():
            buffered_data.rotate()

        for sensor_cls in self._sensors_by_type.keys():
            dtype = sensor_cls._get_cache_dtype()
            cache_slice = self._cache_slices_by_type[sensor_cls]
            gt_slice = self._ground_truth_cache[dtype][cache_slice]
            sensor_cls._update_shared_ground_truth_cache(self._sensors_metadata[sensor_cls], gt_slice)
            if self._should_update_cache_by_type[sensor_cls]:
                sensor_cls._update_shared_cache(
                    self._sensors_metadata[sensor_cls],
                    gt_slice.T,
                    self._cache[dtype][:, cache_slice],
                    self._buffered_data[dtype][:, cache_slice],
                )
            for is_ground_truth in (False, True):
                key = (is_ground_truth, dtype)
                self._is_last_cache_cloned[key] = False
                self._cloned_cache[key] = torch.tensor([], dtype=dtype, device=gs.device)

    def draw_debug(self, context: "RasterizerContext"):
        for sensor in self.sensors:
            if sensor._options.draw_debug:
                sensor._draw_debug(context)

    def get_cloned_from_cache(self, sensor: "Sensor", is_ground_truth: bool = False) -> torch.Tensor:
        dtype = sensor._get_cache_dtype()
        key = (is_ground_truth, dtype)
        if not self._is_last_cache_cloned[key]:
            self._is_last_cache_cloned[key] = True
            if is_ground_truth:
                self._cloned_cache[key] = self._ground_truth_cache[dtype].T.contiguous()
            else:
                self._cloned_cache[key] = self._cache[dtype].clone()
        return self._cloned_cache[key][:, sensor._cache_idx : sensor._cache_idx + sensor._cache_size]

    @property
    def sensors(self):
        pass
