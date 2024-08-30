# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import OpenGL.GL as GL
from animated_drawings.model.transform import Transform
from animated_drawings.model.renderable import Renderable


class Box(Transform, Renderable):

    def __init__(self, shader_name: str = 'light_shader', s=0.5, c=[0.0, 0.0, 0.0], **kwargs) -> None:
        super().__init__(**kwargs)

        vertices = np.array([
            [ s,  s, -s, *c, 0.0,  0.0, -1.0],
            [ s, -s, -s, *c, 0.0,  0.0, -1.0],
            [-s, -s, -s, *c, 0.0,  0.0, -1.0],
            [-s, -s, -s, *c, 0.0,  0.0, -1.0],
            [-s,  s, -s, *c, 0.0,  0.0, -1.0],
            [ s,  s, -s, *c, 0.0,  0.0, -1.0],

            [-s, -s,  s, *c, 0.0,  0.0,  1.0],
            [ s, -s,  s, *c, 0.0,  0.0,  1.0],
            [ s,  s,  s, *c, 0.0,  0.0,  1.0],
            [ s,  s,  s, *c, 0.0,  0.0,  1.0],
            [-s,  s,  s, *c, 0.0,  0.0,  1.0],
            [-s, -s,  s, *c, 0.0,  0.0,  1.0],

            [ s, -s, -s, *c, 1.0,  0.0,  0.0],
            [ s,  s, -s, *c, 1.0,  0.0,  0.0],
            [ s,  s,  s, *c, 1.0,  0.0,  0.0],
            [ s,  s,  s, *c, 1.0,  0.0,  0.0],
            [ s, -s,  s, *c, 1.0,  0.0,  0.0],
            [ s, -s, -s, *c, 1.0,  0.0,  0.0],

            [-s,  s,  s, *c, 1.0,  0.0,  0.0],
            [-s,  s, -s, *c, 1.0,  0.0,  0.0],
            [-s, -s, -s, *c, 1.0,  0.0,  0.0],
            [-s, -s, -s, *c, 1.0,  0.0,  0.0],
            [-s, -s,  s, *c, 1.0,  0.0,  0.0],
            [-s,  s,  s, *c, 1.0,  0.0,  0.0],

            [-s, -s, -s, *c, 0.0, -1.0,  0.0],
            [ s, -s, -s, *c, 0.0, -1.0,  0.0],
            [ s, -s,  s, *c, 0.0, -1.0,  0.0],
            [ s, -s,  s, *c, 0.0, -1.0,  0.0],
            [-s, -s,  s, *c, 0.0, -1.0,  0.0],
            [-s, -s, -s, *c, 0.0, -1.0,  0.0],

            [ s,  s,  s, *c, 0.0,  1.0,  0.0],
            [ s,  s, -s, *c, 0.0,  1.0,  0.0],
            [-s,  s, -s, *c, 0.0,  1.0,  0.0],
            [-s,  s, -s, *c, 0.0,  1.0,  0.0],
            [-s,  s,  s, *c, 0.0,  1.0,  0.0],
            [ s,  s,  s, *c, 0.0,  1.0,  0.0],

        ], np.float32)

        attrib_pointers = {
            'position': (0, 3),
            'color': (3, 3),
            'normal': (6, 3)
        }

        self._initialize_opengl_resources(vertices, attrib_pointers)

        self.shader_name: str = shader_name

    def _draw(self, **kwargs) -> None:

        GL.glUseProgram(kwargs['shader_ids'][self.shader_name])
        model_loc = GL.glGetUniformLocation(kwargs['shader_ids'][self.shader_name], "model")
        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, self._world_transform.T)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)
