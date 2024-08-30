# Copyright (c) Meta Platforms, Inc. and affiliates.

""" Interactive Controller Class Module """

import time
from typing import Optional
import glfw
import math

from animated_drawings.controller.controller import Controller
from animated_drawings.model.scene import Scene
from animated_drawings.view.window_view import WindowView
from animated_drawings.config import ControllerConfig
from animated_drawings.model.quaternions import Quaternions
from animated_drawings.model.vectors import Vectors
import numpy as np


class InteractiveController(Controller):
    """ Interactive Controller Class """

    def __init__(self, cfg: ControllerConfig, scene: Scene, view: WindowView) -> None:
        super().__init__(cfg, scene)

        self.view: WindowView = view
        self.prev_time: float = 0.0  # tracks real-world time passing between run loops

        # TODO: camera movement is really slow depending upon whether scene is progressing in time or not.
        # Figure out a better solution than this
        self.controller_prev_time: float = 0.0  # time between run loops for controller input

        self.pause: bool = False     # tracks whether time progresses in the scene

        self.look_pos_speed = 3

        glfw.init()

        """ keyboard controls """
        # a dictionary mapping glfw key codes to [controller function, user friendly text description of the function]
        self.controls = {
            glfw.KEY_Q:      [lambda: glfw.set_window_should_close(self.view.win, True), f'{chr(glfw.KEY_Q)}: Close window'],
            glfw.KEY_ESCAPE: [lambda: glfw.set_window_should_close(self.view.win, True),  'ESCAPE: Close window'],

            glfw.KEY_W: [lambda: self.view.camera.offset(-self.look_pos_speed * (max(0.05, time.time()-self.controller_prev_time)) * self.view.camera.get_world_transform()[:-1, 2]), f'{chr(glfw.KEY_W)}: Move camera forward'],
            glfw.KEY_S: [lambda: self.view.camera.offset( self.look_pos_speed * (max(0.05, time.time()-self.controller_prev_time)) * self.view.camera.get_world_transform()[:-1, 2]), f'{chr(glfw.KEY_S)}: Move camera backwards'],
            glfw.KEY_A: [lambda: self.view.camera.offset(-self.look_pos_speed * (max(0.05, time.time()-self.controller_prev_time)) * self.view.camera.get_world_transform()[:-1, 0]), f'{chr(glfw.KEY_A)}: Move camera left'],
            glfw.KEY_D: [lambda: self.view.camera.offset( self.look_pos_speed * (max(0.05, time.time()-self.controller_prev_time)) * self.view.camera.get_world_transform()[:-1, 0]), f'{chr(glfw.KEY_D)}: Move camera right'],
            glfw.KEY_E: [lambda: self.view.camera.offset(-self.look_pos_speed * (max(0.05, time.time()-self.controller_prev_time)) * self.view.camera.get_world_transform()[:-1, 1]), f'{chr(glfw.KEY_E)}: Move camera up'],
            glfw.KEY_R: [lambda: self.view.camera.offset( self.look_pos_speed * (max(0.05, time.time()-self.controller_prev_time)) * self.view.camera.get_world_transform()[:-1, 1]), f'{chr(glfw.KEY_R)}: Move camera down'],
            glfw.KEY_T: [self._print_camera_position, f'{chr(glfw.KEY_T)}: Print camera position'],

            glfw.KEY_Z: [lambda: self.view.camera.rotation_offset(Quaternions.from_angle_axis(np.array([ 0.1], np.float32), Vectors([0.0, 1.0, 0.0]))), f'{chr(glfw.KEY_Z)}: Turn camera left'],
            glfw.KEY_C: [lambda: self.view.camera.rotation_offset(Quaternions.from_angle_axis(np.array([-0.1], np.float32), Vectors([0.0, 1.0, 0.0]))), f'{chr(glfw.KEY_C)}: Turn camera right'],

            glfw.KEY_Y: [self._toggle_ad_rig_visibility, f'{chr(glfw.KEY_Y)}: Show/hide character rig'],
            glfw.KEY_U: [self._toggle_ad_reflection_visibility, f'{chr(glfw.KEY_U)}: Show/hide character reflection'],
            glfw.KEY_I: [self._toggle_ad_txtr_visibility, f'{chr(glfw.KEY_I)}: Show/hide character txtr'],
            glfw.KEY_O: [self._toggle_ad_color_visibility, f'{chr(glfw.KEY_O)}: Show/hide character joint color'],
            glfw.KEY_P: [self._toggle_ad_meshlines_visibility, f'{chr(glfw.KEY_P)}: Show/hide character mesh lines'],
            glfw.KEY_H: [self._toggle_ad_bvh_visibility, f'{chr(glfw.KEY_H)}: Show/hide BVH'],
            glfw.KEY_J: [self._toggle_ad_shadows, f'{chr(glfw.KEY_J)}: Show/hide shadows'],
            glfw.KEY_K: [self._toggle_floor_visibility, f'{chr(glfw.KEY_K)}: Show/hide floor'],
            glfw.KEY_N: [self._toggle_background_image_visibility, f'{chr(glfw.KEY_N)}: Show/hide background image'],

            glfw.KEY_SPACE: [self._toggle_time_progress,                      'SPACE: Start/stop time progression'],
            glfw.KEY_RIGHT: [lambda: self._tick( self.cfg.keyboard_timestep), 'RIGHT: Step forward in time'],
            glfw.KEY_LEFT:  [lambda: self._tick(-self.cfg.keyboard_timestep), 'LEFT: Step backward in time'],

            glfw.KEY_L: [self._toggle_mouse_controls_active, 'L: Toggle mouse control activation'],
        }
        glfw.set_key_callback(self.view.win, self._on_key)
        print(self.get_controller_instructions())

        """ mouse controls """
        # TODO: This isn't computing existing viewing angles for first frame,
        # resulting in bug where camera moves dramatically first time mouse is touched
        fwd_x, fwd_y, fwd_z = self.view.camera._rotate_m[:-1, 2]
        self.look_v_angle = 1 - math.atan2(fwd_y, 1-fwd_y**2)
        self.look_h_angle = math.atan2(fwd_x, fwd_z)

        self.mouse_controls_active = False
        self.look_rotate_speed = 0.05
        self._toggle_mouse_controls_active()
        glfw.set_cursor_pos_callback(self.view.win, self._cursor_position_callback)

    def get_controller_instructions(self):
        header = 'KEYBOARD CONTROLS:\n=========================='
        tail = '=========================='
        return "\n".join([header, *[x[1] for x in self.controls.values()], tail])

    def _on_key(self, _win, key: int, _scancode, action, _mods) -> None:  # noqa: C901

        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key in self.controls.keys():
            self.controls[key][0]()

    def _cursor_position_callback(self, _win, xpos: float, ypos: float) -> None:
        if not self.mouse_controls_active:
            return

        xpos, ypos = glfw.get_cursor_pos(self.view.win)
        win_center = [x/2 for x in glfw.get_window_size(self.view.win)]

        self.look_v_angle += self.look_rotate_speed * (time.time()-self.controller_prev_time) * (ypos - win_center[1])
        self.look_v_angle = min(max(self.look_v_angle, -np.pi/2+0.001), -0.001+np.pi/2)
        print(self.look_v_angle)

        self.look_h_angle += self.look_rotate_speed * (time.time()-self.controller_prev_time) * (win_center[0] - xpos)

        self.view.camera.look_at_spherical_coordinates(self.look_v_angle, self.look_h_angle)

        glfw.set_cursor_pos(self.view.win, *(win_center))

    def _toggle_mouse_controls_active(self) -> None:
        self.mouse_controls_active = not self.mouse_controls_active

        if self.mouse_controls_active:
            """ mouse controls active """
            glfw.set_input_mode(self.view.win, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        else:
            """ mouse controls inactive"""
            glfw.set_input_mode(self.view.win, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def _toggle_ad_bvh_visibility(self) -> None:
        self.view.cfg.draw_bvh = not self.view.cfg.draw_bvh

    def _toggle_ad_shadows(self) -> None:
        self.view.cfg.draw_shadows = not self.view.cfg.draw_shadows

        if not self.view.cfg.draw_shadows:
            self.view.clear_shadow_buffer()

    def _toggle_floor_visibility(self) -> None:
        self.view.cfg.draw_floor = not self.view.cfg.draw_floor

    def _toggle_background_image_visibility(self) -> None:
        # if no valid background image, this should do nothing
        if not self.view.cfg.background_image_path:
            return
        self.view.cfg.draw_background_image = not self.view.cfg.draw_background_image

    def _toggle_ad_meshlines_visibility(self) -> None:
        self.view.cfg.draw_ad_mesh_lines = not self.view.cfg.draw_ad_mesh_lines

    def _toggle_ad_txtr_visibility(self) -> None:
        self.view.cfg.draw_ad_txtr = not self.view.cfg.draw_ad_txtr

    def _toggle_ad_color_visibility(self) -> None:
        self.view.cfg.draw_ad_color = not self.view.cfg.draw_ad_color

    def _toggle_ad_rig_visibility(self) -> None:
        self.view.cfg.draw_ad_rig = not self.view.cfg.draw_ad_rig

    def _toggle_ad_reflection_visibility(self) -> None:
        self.view.cfg.draw_character_reflection = not self.view.cfg.draw_character_reflection

    def _toggle_time_progress(self) -> None:
        self.pause = not self.pause
        self.prev_time = time.time()

    def _print_camera_position(self) -> None:
        print(f'camera positions: {self.view.camera.get_world_position()}')

    def _is_run_over(self) -> None:
        return glfw.window_should_close(self.view.win)

    def _prep_for_run_loop(self) -> None:
        self.prev_time = time.time()
        self.controller_prev_time = time.time()

    def _start_run_loop_iteration(self) -> None:
        self.view.clear_window()

    def _tick(self, delta_t: Optional[float] = None) -> None:
        # if passed a specific value to progress time by, do so
        if delta_t:
            self.scene.progress_time(delta_t)
        # otherwise, if scene is paused, do nothing
        elif self.pause:
            pass
        # otherwise, calculate real time passed since last call and progress scene by that amount
        else:
            cur_time = time.time()
            self.scene.progress_time(cur_time - self.prev_time)
            self.prev_time = cur_time

    def _update(self) -> None:
        self.scene.update_transforms()

    def _handle_user_input(self) -> None:
        # # Uncomment for rotation visualization  TODO: find a better place to call this
        # from animated_drawings.model.animated_drawing import AnimatedDrawing
        # target = [x for x in self.scene.get_children() if isinstance(x, AnimatedDrawing)][0]
        # target_pos = target.get_world_position()
        # self.view.camera.rotate_about_target_pos(target_pos)

        glfw.poll_events()

    def _render(self) -> None:
        self.view.render(self.scene)

    def _finish_run_loop_iteration(self) -> None:
        self.view.swap_buffers()
        self.controller_prev_time = time.time()

    def _cleanup_after_run_loop(self) -> None:
        self.view.cleanup()
