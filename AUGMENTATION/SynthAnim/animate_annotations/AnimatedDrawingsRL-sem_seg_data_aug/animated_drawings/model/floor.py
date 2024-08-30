# Copyright (c) Meta Platforms, Inc. and affiliates.

from animated_drawings.model.transform import Transform
from animated_drawings.model.renderable import Renderable
import numpy as np
import OpenGL.GL as GL


class Floor(Transform, Renderable):

    def __init__(self, shader_name='light_shader', length=10, width=10):
        super().__init__()

        self.shader_name = shader_name

        self.tile_count = length * width

        _vertices = []
        for ldx in range(length):
            for wdx in range(width):
                if (ldx + wdx) % 2:
                    c = [1.0, 1.0, 1.0]
                else:
                    c = [0.0, 0.0, 0.0]

                _vertices.append(np.array([
                    [-length//2 + ldx +  0.5, 0.0, -width//2 + wdx +  0.5, *c, 0.0, 1.0, 0.0, 1.0, 1.0],  # top right
                    [-length//2 + ldx + -0.5, 0.0, -width//2 + wdx + -0.5, *c, 0.0, 1.0, 0.0, 0.0, 0.0],  # bottom left
                    [-length//2 + ldx + -0.5, 0.0, -width//2 + wdx +  0.5, *c, 0.0, 1.0, 0.0, 0.0, 1.0],  # top left
                    [-length//2 + ldx +  0.5, 0.0, -width//2 + wdx + -0.5, *c, 0.0, 1.0, 0.0, 1.0, 0.0],  # bottom right
                    [-length//2 + ldx + -0.5, 0.0, -width//2 + wdx + -0.5, *c, 0.0, 1.0, 0.0, 0.0, 0.0],  # bottom left
                    [-length//2 + ldx +  0.5, 0.0, -width//2 + wdx +  0.5, *c, 0.0, 1.0, 0.0, 1.0, 1.0],  # top right
                ], np.float32)
                )
        vertices = np.vstack(_vertices)

        attrib_pointers = {
            'position': (0, 3),
            'color': (3, 3),
            'normal': (6, 3),
            'texture': (9, 2)
        }

        self._initialize_opengl_resources(vertices, attrib_pointers)

    def _draw(self, **kwargs) -> None:
        if not kwargs['viewer_cfg'].draw_floor:
            return

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glUseProgram(kwargs['shader_ids'][self.shader_name])
        model_loc = GL.glGetUniformLocation(kwargs['shader_ids'][self.shader_name], "model")
        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, self._world_transform.T)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6 * self.tile_count)
