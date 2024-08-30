# Copyright (c) Meta Platforms, Inc. and affiliates.

from animated_drawings.model.transform import Transform
from animated_drawings.model.vectors import Vectors
from typing import Union, List
import logging
import numpy as np
import numpy.typing as npt


class Camera(Transform):

    def __init__(
        self,
        pos: Union[Vectors, List[Union[float, int]]] = Vectors([0.0, 0.0, 0.0]),
        fwd: Union[Vectors, List[Union[float, int]]] = Vectors([0.0, 0.0, 1.0])
    ):

        if not isinstance(pos, Vectors):
            pos = Vectors(pos)

        super().__init__(offset=pos)

        if not isinstance(fwd, Vectors):
            fwd = Vectors(fwd)
        self.look_at(fwd)

        self.update_transforms()

        self.projection_matrix: npt.NDArray[np.float32]

    def set_projection_matrix_perspective(self, buffer_w: int, buffer_h: int) -> None:
        fov = 35.0
        near = 0.1
        aspect = buffer_w / buffer_h
        top = near * np.tan(fov * np.pi / 360)
        right = top * aspect
        far = 50.0
        bottom = -top
        left = -right

        M_0_0 =       (2 * near) / (right - left)
        M_0_2 =   (left + right) / (left - right)
        M_1_1 =       (2 * near) / (top - bottom)
        M_1_2 =   (bottom + top) / (bottom-top)
        M_2_2 =     (far + near) / (near - far)
        M_2_3 = (2 * far * near) / (near - far)
        M_3_2 = -1

        M: npt.NDArray[np.float32] = np.zeros([4, 4], dtype=np.float32)
        M[0, 0] = M_0_0
        M[0, 2] = M_0_2
        M[1, 1] = M_1_1
        M[1, 2] = M_1_2
        M[2, 2] = M_2_2
        M[2, 3] = M_2_3
        M[3, 2] = M_3_2

        self.projection_matrix = M

    def set_projection_matrix_orthographic(self, left: float = -10.0, right: float = 10.0, bottom: float = -10.0, top: float = 10.0, near: float = 10.0, far: float = -10.0) -> None :
        self.projection_matrix = np.array([
            [2 / (right - left),         0.0,         0.0, -(right + left) / (right - left)],
            [        0.0, 2 / (top - bottom),         0.0, -(top + bottom) / (top - bottom)],
            [        0.0,         0.0, 2 / (near - far), -(near + far) / (near - far)],
            [        0.0,         0.0,         0.0,                  1]
        ])

    def rotate_about_target_pos(self, target_pos: npt.NDArray[np.float32]) -> None:
        from animated_drawings.model.quaternions import Quaternions
        from animated_drawings.model.vectors import Vectors
        angle_speed = -0.01

        try:
            # get rotation matrix from angle
            self.angle += angle_speed
            rot = Quaternions.from_angle_axis(np.array([self.angle]), Vectors([0, 1, 0])).to_rotation_matrix()

            # get position rotating about origin from rotation matrix and radius
            pos = np.array([0, 0, self.radius, 1.0])
            pos = rot @ pos
            pos = pos[:-1]

            # keep height constant
            pos[1] = self.update_and_get_world_position()[1]

            self.set_position(pos)

            # orient camera towards target
            look_at = -(target_pos - pos)
            look_at[1] = 0  # keep horizontal
            self.look_at(look_at)

        except Exception as e:
            logging.warn(e)
            self.angle = 0
            self.radius = 6
