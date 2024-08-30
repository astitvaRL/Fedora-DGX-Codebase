from typing import Optional
import numpy as np
import OpenGL.GL as GL
from animated_drawings.model.transform import Transform
from animated_drawings.model.renderable import Renderable
from animated_drawings.model.vectors import Vectors
from animated_drawings.model.quaternions import Quaternions


class BoneWidget(Transform, Renderable):

    def __init__(self, shader_name: str = 'light_shader', base=0.5, height=1.0, starting_orientation: Optional[Vectors] = None, **kwargs) -> None:

        if starting_orientation:
            kwargs['rotation'] = Quaternions.rotate_between_vectors(Vectors([0.0, 1.0, 0.0]), starting_orientation)

        super().__init__(**kwargs)

        b = base / 2   # half of base length
        h = height  # bone length
        self.shader_name = shader_name

        c = [0.4, 0.9, 0.6]

        vertices = np.array([
            # proxal part

            [ b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],  # side 1
            [-b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],
            [ 0,      0,  0, *c, 0.0, 0.0, 0.0],

            [-b, 0.25*h, -b, *c, 0.0, 0.0, 0.0],  # side 2
            [ b, 0.25*h, -b, *c, 0.0, 0.0, 0.0],
            [ 0,      0,  0, *c, 0.0, 0.0, 0.0],

            [-b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],  # side 3
            [-b, 0.25*h, -b, *c, 0.0, 0.0, 0.0],
            [ 0,      0,  0, *c, 0.0, 0.0, 0.0],

            [ b, 0.25*h, -b, *c, 0.0, 0.0, 0.0],  # side 4
            [ b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],
            [ 0,      0,  0, *c, 0.0, 0.0, 0.0],

            # distal part

            [ 0,      h,  0, *c, 0.0, 0.0, 0.0],  # side 1
            [-b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],
            [ b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],

            [ 0,      h,  0, *c, 0.0, 0.0, 0.0],  # side 2
            [ b, 0.25*h, -b, *c, 0.0, 0.0, 0.0],
            [-b, 0.25*h, -b, *c, 0.0, 0.0, 0.0],

            [ 0,      h,  0, *c, 0.0, 0.0, 0.0],  # side 3
            [-b, 0.25*h, -b, *c, 0.0, 0.0, 0.0],
            [-b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],

            [ 0,      h,  0, *c, 0.0, 0.0, 0.0],  # side 4
            [ b, 0.25*h,  b, *c, 0.0, 0.0, 0.0],
            [ b, 0.25*h, -b, *c, 0.0, 0.0, 0.0]], np.float32)

        # calculate the normals
        iter_vertices = iter(vertices)
        for v1, v2, v3 in zip(iter_vertices, iter_vertices, iter_vertices):
            normal = Vectors.compute_normal(Vectors(v2[:3]-v1[:3]), Vectors(v3[:3]-v1[:3]))
            v1[-3:] = v2[-3:] = v3[-3:] = normal.vs

        attrib_pointers = {
            'position': (0, 3),
            'color': (3, 3),
            'normal': (6, 3)
        }

        self._initialize_opengl_resources(vertices, attrib_pointers)

    def _draw(self, **kwargs):
        GL.glUseProgram(kwargs['shader_ids'][self.shader_name])
        model_loc = GL.glGetUniformLocation(kwargs['shader_ids'][self.shader_name], "model")
        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, self._world_transform.T)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.vertices))
