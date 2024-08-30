from OpenGL import GL
import ctypes
from typing import Dict, Tuple


class Renderable():

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize_opengl_resources(self, vertices, attrib_pointers: Dict[str, Tuple[int, int]]):
        """attrib_pointers is a dictionary mapping the name of the attribute to it's starting position and length in vertex data array"""

        """ vertices is (v, 6) """

        self.vertices = vertices

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)

        # buffer vertex data
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices, GL.GL_STATIC_DRAW)

        for attribute_name, (position, length) in attrib_pointers.items():

            if attribute_name == 'position':
                array_id = 0
            elif attribute_name == 'color':
                array_id = 1
            elif attribute_name == 'normal':
                array_id = 2
            elif attribute_name == 'texture':
                array_id = 3
            else:
                raise ValueError

            offset = 4 * position
            GL.glVertexAttribPointer(array_id, length, GL.GL_FLOAT, False, 4 * self.vertices.shape[1], ctypes.c_void_p(offset))  # 4 is byte size of np.float32
            GL.glEnableVertexAttribArray(array_id)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
