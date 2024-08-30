# Copyright (c) Meta Platforms, Inc. and affiliates.

""" Video Render Controller Class Module """

from __future__ import annotations
import time
import logging
from typing import List
from tqdm import tqdm

from animated_drawings.controller.controller import Controller
from animated_drawings.model.scene import Scene
from animated_drawings.model.animated_drawing import AnimatedDrawing
from animated_drawings.view.export_view import ExportView
from animated_drawings.config import ControllerConfig

NoneType = type(None)  # for type checking below


class ExportController(Controller):
    """ Video Render Controller is used to non-interactively generate a video file """

    def __init__(self, cfg: ControllerConfig, scene: Scene, view: ExportView) -> None:
        super().__init__(cfg, scene)

        self.view: ExportView = view

        self.scene: Scene = scene

        self.frames_left_to_render: int  # when this becomes zero, stop rendering
        self.delta_t: float              # amount of time to progress scene between renders
        self._set_frames_left_to_render_and_delta_t()

        self.render_start_time: float  # track when we started to render frames (for performance stats)
        self.frames_rendered: int = 0  # track how many frames we've rendered

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
            max_frames = max(max_frames, child.retargeter.motion_source.get_max_frame_count())
            frame_time.append(child.retargeter.motion_source.get_frame_time())

        if not all(x == frame_time[0] for x in frame_time):
            msg = f'frame time of BVH files don\'t match. Using first value: {frame_time[0]}'
            logging.warning(msg)

        self.frames_left_to_render = max_frames
        self.delta_t = frame_time[0]

    def _prep_for_run_loop(self) -> None:
        self.run_loop_start_time = time.time()

        self.view.record_clip_level_data(self.scene, 1/self.delta_t)

    def _is_run_over(self) -> bool:
        return self.frames_left_to_render == 0

    def _start_run_loop_iteration(self) -> None:
        self.view.clear_window()

    def _update(self) -> None:
        self.scene.update_transforms()

    def _render(self) -> None:
        self.view.record_frame_level_data(self.scene)

    def _tick(self) -> None:
        self.scene.progress_time(self.delta_t)

    def _handle_user_input(self) -> None:
        pass
        """ ignore all user input when rendering video file """

    def _finish_run_loop_iteration(self) -> None:

        # update our counts and progress_bar
        self.frames_left_to_render -= 1
        self.frames_rendered += 1
        self.progress_bar.update(1)

    def _cleanup_after_run_loop(self) -> None:
        logging.info(f'Rendered {self.frames_rendered} frames in {time.time()-self.run_loop_start_time} seconds.')
        self.view.cleanup()
