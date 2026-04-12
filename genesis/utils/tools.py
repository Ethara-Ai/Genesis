import inspect
import os
import time

import numpy as np
import quadrants as qd
from PIL import Image

import genesis as gs


def animate(imgs, filename=None, fps=60):
    """
    Create a video from a list of images.

    Args:
        imgs (list): List of input images.
        filename (str, optional): Name of the output video file. If not provided, the name will be default to the name of the caller file, with a timestamp and '.mp4' extension.
    """
    assert isinstance(imgs, list)
    if len(imgs) == 0:
        gs.logger.warning("No image to save.")
        return

    if filename is None:
        caller_file = inspect.stack()[-1].filename
        # caller file + timestamp + .mp4
        filename = os.path.splitext(os.path.basename(caller_file))[0] + f"_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)

    gs.logger.info(f'Saving video to ~<"{filename}">~...')
    from moviepy import ImageSequenceClip

    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(
        filename,
        fps=fps,
        logger=None,
        codec="libx264",
        preset="ultrafast",
        # ffmpeg_params=["-crf", "0"],
    )
    gs.logger.info("Video saved.")


def save_img_arr(arr, filename="img.png"):
    pass


class Timer:
    def __init__(self, skip=False, level=0, qd_sync=False):
        self.accu_log = dict()
        self.skip = skip
        self.level = level
        self.qd_sync = qd_sync
        self.msg_width = 0
        self.reset()

    def reset(self):
        self.just_reset = True
        if self.level == 0 and not self.skip:
            try:
                column, _lines = os.get_terminal_size()
            except OSError:
                column = 80
            print("─" * column)
        if self.qd_sync and not self.skip:
            qd.sync()
        self.prev_time = self.init_time = time.perf_counter()

    def _stamp(self, msg="", _ratio=1.0):
        pass

    def stamp(self, msg="", _ratio=1.0):
        pass


timers = dict()


def create_timer(name=None, new=False, level=0, qd_sync=False, skip_first_call=False):
    pass


class Rate:
    def __init__(self, rate):
        self.rate = rate
        self.last_time = time.perf_counter()

    def sleep(self):
        current_time = time.perf_counter()
        sleep_duration = 1.0 / self.rate - (current_time - self.last_time)
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.last_time = time.perf_counter()


class FPSTracker:
    def __init__(
        self, n_envs, alpha=0.95, minimum_interval_seconds: float | None = 0.05, outlier_threshold: float = 1.5
    ):
        self.last_time = None
        self.n_envs = n_envs
        self.dt_ema = None
        self.alpha = alpha
        self.minimum_interval_seconds = minimum_interval_seconds
        self.outlier_threshold = outlier_threshold
        self.steps_since_last_print: int = 0
        self.total_fps = 0.0

    def step(self, current_time: float | None = None) -> float | None:
        if not current_time:
            current_time = time.perf_counter()

        if self.last_time:
            dt = current_time - self.last_time
        else:
            self.last_time = current_time
            return None

        self.steps_since_last_print += 1

        # Skip if update is too soon
        if self.minimum_interval_seconds and current_time - self.last_time < self.minimum_interval_seconds:
            return None

        # Outlier rejection
        if self.dt_ema is not None:
            if dt > self.dt_ema * self.outlier_threshold or dt * self.outlier_threshold < self.dt_ema:
                self.dt_ema = dt

        # EMA update
        if self.dt_ema:
            self.dt_ema = self.alpha * self.dt_ema + (1 - self.alpha) * dt
        else:
            self.dt_ema = dt

        fps = 1 / self.dt_ema * self.steps_since_last_print
        if self.n_envs > 0:
            self.total_fps = fps * self.n_envs
            gs.logger.info(
                f"Running at ~<{self.total_fps:,.2f}>~ FPS (~<{fps:.2f}>~ FPS per env, ~<{self.n_envs}>~ envs)."
            )
        else:
            self.total_fps = fps
            gs.logger.info(f"Running at ~<{fps:.2f}>~ FPS.")
        self.last_time = current_time
        self.steps_since_last_print = 0
        return self.total_fps
