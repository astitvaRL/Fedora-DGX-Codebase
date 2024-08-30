# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import OpenGL.GL as GL
from animated_drawings.model.transform import Transform
from animated_drawings.model.renderable import Renderable


class Rectangle(Transform, Renderable):

    def __init__(self, color: str = 'white', shader_name='light_shader', **kwargs) -> None:

        super().__init__(**kwargs)

        self.shader_name = shader_name

        if color == 'white':
            c = np.array([1.0, 1.0, 1.0], np.float32)
        elif color == 'black':
            c = np.array([0.3, 0.3, 0.3], np.float32)
        elif color == 'blue':
            c = np.array([0.00, 0.0, 1.0], np.float32)
        else:
            assert len(color) == 3
            c = np.array([*color], np.float32)

        vertices = np.array([
            [ 0.5, 0.0,  0.5, *c, 0.0, 1.0, 0.0, 1.0, 1.0],  # top right
            [-0.5, 0.0, -0.5, *c, 0.0, 1.0, 0.0, 0.0, 0.0],  # bottom left
            [-0.5, 0.0,  0.5, *c, 0.0, 1.0, 0.0, 0.0, 1.0],  # top left
            [ 0.5, 0.0, -0.5, *c, 0.0, 1.0, 0.0, 1.0, 0.0],  # bottom right
            [-0.5, 0.0, -0.5, *c, 0.0, 1.0, 0.0, 0.0, 0.0],  # bottom left
            [ 0.5, 0.0,  0.5, *c, 0.0, 1.0, 0.0, 1.0, 1.0],  # top right
        ], np.float32)

        attrib_pointers = {
            'position': (0, 3),
            'color': (3, 3),
            'normal': (6, 3),
            'texture': (9, 2)
        }

        self._initialize_opengl_resources(vertices, attrib_pointers)

    def _draw(self, **kwargs) -> None:

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glUseProgram(kwargs['shader_ids'][self.shader_name])
        model_loc = GL.glGetUniformLocation(kwargs['shader_ids'][self.shader_name], "model")
        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, self._world_transform.T)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
