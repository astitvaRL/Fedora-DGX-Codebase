from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from animated_drawings.model.transform import Transform
from animated_drawings.model.renderable import Renderable
from animated_drawings.model.joint import Joint
from abc import abstractmethod

import json
import copy
import heapq
import logging
from skimage import measure
import triangle as tr
import igl
from OpenGL import GL
import base64
from PIL import Image
import io
import ctypes


class AnimatedDrawingMeshClient(Transform, Renderable):

    def __init__(
        self,
        vertices_xy: List[List[float]],
        vertices_uv: List[List[float]],
        vertices_normal: List[List[float]],
        triangles: List[List[int]],
        submesh_names: List[str],
        submesh_starting_indices: List[int],
        txtr_names: List[str],
        txtrs: List[str],
        hide_outside_of
    ):

        self.hide_outside_of = hide_outside_of

        super().__init__()

        self.vertices: npt.NDArray[np.float32] = np.zeros([len(vertices_xy), 11], dtype=np.float32)
        self.vertices[:, :2] = vertices_xy
        self.vertices[:, 6:8] = vertices_uv
        self.vertices[:, 8:11] = vertices_normal

        self.indices: npt.NDArray[np.int32] = np.array(triangles, dtype=np.int32).flatten()  # the order in which to render the vertices

        # info needed for later rendering submeshes
        self.submesh_names = submesh_names
        self.submesh_starting_indices = submesh_starting_indices

        self.txtr_dict: Dict[str, npt.NDArray[np.uint8]] = {}
        for i in range(len(txtr_names)):
            # decode from base64 to numpy array
            name, txtr = txtr_names[i], txtrs[i]
            png_bytes = base64.b64decode(txtr)
            image = Image.open(io.BytesIO(png_bytes))
            numpy_array = np.array(image)

            # renderer expects RGBA images. Add channels until = 4
            channel_num = numpy_array.shape[2]
            if channel_num != 4:  # is not RGBA
                mock_channels = np.full((512, 512, 4-channel_num), 255)
                numpy_array = np.concatenate((np.array(image), mock_channels), axis=2)

            self.txtr_dict[name] = numpy_array

        self.active_txtr_name: str = 'front'  # specifies name of texture to apply to character.

        self._initialize_opengl_resources()

    def _initialize_opengl_resources(self) -> None:

        attrib_pointers = {
            'position': (0, 3),
            'color': (3, 3),
            'texture': (6, 2),
            'normal': (8, 3)
        }
        super()._initialize_opengl_resources(self.vertices, attrib_pointers)

        # initialize textures
        h, w, _ = self.txtr_dict['front'].shape
        self.txtr_id_dict: Dict[str, GL.GLuint] = {}
        GL.glActiveTexture(GL.GL_TEXTURE1)
        for name, txtr in self.txtr_dict.items():
            self.txtr_id_dict[name] = GL.glGenTextures(1)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.txtr_id_dict[name])
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, txtr)

        # buffer element index data, since Renderable class doesn't support that yet
        self.ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)

        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER,
                        self.indices, GL.GL_STATIC_DRAW)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def _rebuffer_vertex_data(self):

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        # self.rebuffer_indices(self.indices)

        self._vertex_buffer_dirty_bit = False

    def rebuffer_indices(self, indices):
        new_indices = np.zeros(self.indices.shape[0], dtype=np.int32)
        new_indices[:len(indices)] = indices.copy()

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, new_indices, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def _draw(self, **kwargs):
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)

        if kwargs['submesh_name']:
            starting_index = self.submesh_starting_indices[self.submesh_names.index(kwargs['submesh_name'])]
            try:
                ending_index = self.submesh_starting_indices[self.submesh_names.index(kwargs['submesh_name']) + 1]
            except IndexError:
                ending_index = len(self.indices)
        else:
            starting_index = 0
            ending_index = len(self.indices)

        if kwargs['viewer_cfg'].draw_ad_txtr:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.txtr_id_dict[self.active_txtr_name])

            GL.glUseProgram(kwargs['shader_ids']['texture_shader'])
            model_loc = GL.glGetUniformLocation(kwargs['shader_ids']['texture_shader'], "model")
            GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, self._world_transform.T)

            reflection_flag_log = GL.glGetUniformLocation(kwargs['shader_ids']['texture_shader'], "reflection_flag")
            GL.glUniform1i(reflection_flag_log, False)

            # render once with depth_test so shadow shows up
            GL.glDrawElements(GL.GL_TRIANGLES, ending_index - starting_index, GL.GL_UNSIGNED_INT, ctypes.c_void_p(4 * (starting_index)))  # 4 is byte size of np.float32

            # render character for real
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDrawElements(GL.GL_TRIANGLES, ending_index - starting_index, GL.GL_UNSIGNED_INT, ctypes.c_void_p(4 * (starting_index)))  # 4 is byte size of np.float32
            GL.glEnable(GL.GL_DEPTH_TEST)

        if kwargs['viewer_cfg'].draw_ad_mesh_lines:

            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glUseProgram(kwargs['shader_ids']['color_shader'])
            model_loc = GL.glGetUniformLocation(kwargs['shader_ids']['color_shader'], "model")
            GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, self._world_transform.T)

            color_black_loc = GL.glGetUniformLocation(kwargs['shader_ids']['color_shader'], "color_red")
            GL.glUniform1i(color_black_loc, 1)
            GL.glDrawElements(GL.GL_TRIANGLES, ending_index - starting_index, GL.GL_UNSIGNED_INT, ctypes.c_void_p(4 * (starting_index)))  # 4 is byte size of np.float32
            GL.glUniform1i(color_black_loc, 0)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glBindVertexArray(0)
