# Copyright (c) Meta Platforms, Inc. and affiliates.

""" Video Render Controller Class Module """

from __future__ import annotations
import time
import logging
from typing import List
from pathlib import Path
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import cv2
from OpenGL import GL
from tqdm import tqdm

from animated_drawings.controller.controller import Controller
from animated_drawings.model.scene import Scene
from animated_drawings.model.animated_drawing_client import AnimatedDrawing
from animated_drawings.view.view import View
from animated_drawings.config import ControllerConfig

NoneType = type(None)  # for type checking below


class ImagesRenderController(Controller):
    """ Images Render Controller is used to non-interactively generate a set of images """

    def __init__(self, cfg: ControllerConfig, scene: Scene, view: View) -> None:
        super().__init__(cfg, scene)

        self.view: View = view

        self.scene: Scene = scene

        self.frames_left_to_render: int  # when this becomes zero, stop rendering
        self.delta_t: float              # amount of time to progress scene between renders
        self._set_frames_left_to_render_and_delta_t()

        self.render_start_time: float  # track when we started to render frames (for performance stats)
        self.frames_rendered: int = 0  # track how many frames we've rendered

        self.image_width: int
        self.image_height: int
        self.image_width, self.image_height = self.view.get_framebuffer_size()

        if not Path(self.cfg.output_image_dir).is_dir():
            Path(self.cfg.output_image_dir).mkdir(parents=True)

        self.frame_data = np.empty([self.image_height, self.image_width, 4], dtype='uint8')  # 4 for RGBA
        self.black_background = np.zeros([self.image_height, self.image_width, 3], dtype='uint8')

        self.progress_bar = tqdm(total=self.frames_left_to_render)

    def _set_frames_left_to_render_and_delta_t(self) -> None:
        """
        Based upon the animated drawings within the scene, computes maximum number of frames in a BVH.
        Checks that all frame times within BVHs are equal, logs a warning if not.
        Uses results to determine number of frames and frame time for output video.
        """

        max_frames = 0
        frame_time: List[float] = []
        for child in self.scene.get_children():
            if not isinstance(child, AnimatedDrawing):
                continue
            max_frames = max(max_frames, child.motion_source.get_max_frame_count())
            frame_time.append(child.motion_source.get_frame_time())

        if not all(x == frame_time[0] for x in frame_time):
            msg = f'frame time of BVH files don\'t match. Using first value: {frame_time[0]}'
            logging.warning(msg)

        self.frames_left_to_render = max_frames
        self.delta_t = frame_time[0]

    def _prep_for_run_loop(self) -> None:
        self.run_loop_start_time = time.time()

    def _is_run_over(self) -> bool:
        return self.frames_left_to_render == 0

    def _start_run_loop_iteration(self) -> None:
        self.view.clear_window()

    def _update(self) -> None:
        self.scene.update_transforms()

    def _render(self) -> None:
        self.view.render(self.scene)

    def _tick(self) -> None:
        self.scene.progress_time(self.delta_t)

    def _handle_user_input(self) -> None:
        """ ignore all user input when rendering video file """

    def _finish_run_loop_iteration(self) -> None:
        # get pixel values from the frame buffer, send them to the video writer
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadPixels(0, 0, self.image_width, self.image_height, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, self.frame_data)

        if self.frames_rendered != 0 and not self.frames_rendered % 15:  # empty on first frame, only save image for every 15th frame
            output_image = self.black_background.copy()
            output_image[np.where(self.frame_data[:, :, 3] != 0)] = self.frame_data[np.where(self.frame_data[:, :, 3] != 0)][:, :3]
            output_image = output_image[::-1, :, :]
            cv2.imwrite(self.cfg.output_image_dir + f'/{self.frames_rendered}.png', output_image)

        # update our counts and progress_bar
        self.frames_left_to_render -= 1
        self.frames_rendered += 1
        self.progress_bar.update(1)

    def _cleanup_after_run_loop(self) -> None:
        logging.info(f'Rendered {self.frames_rendered} frames in {time.time()-self.run_loop_start_time} seconds.')
        self.view.cleanup()

        _time = time.time()
        logging.info(f'Wrote images in {time.time()-_time} seconds.')
