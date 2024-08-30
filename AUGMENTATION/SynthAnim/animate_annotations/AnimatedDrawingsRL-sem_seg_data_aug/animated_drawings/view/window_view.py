# Copyright (c) Meta Platforms, Inc. and affiliates.

from animated_drawings.view.view import View
from animated_drawings.view.shaders.shader import Shader
from animated_drawings.utils import read_background_image
from animated_drawings.model.scene import Scene
from animated_drawings.model.camera import Camera
from animated_drawings.model.transform import Transform
from animated_drawings.config import ViewConfig
import glfw
import OpenGL.GL as GL
import logging
from typing import Tuple, Dict
import numpy as np
import numpy.typing as npt
from pathlib import Path
from pkg_resources import resource_filename


class WindowView(View):
    """Window View for rendering into a visible window"""

    def __init__(self, cfg: ViewConfig) -> None:

        super().__init__(cfg)

        glfw.init()

        self.win: glfw._GLFWwindow
        self._create_window(*cfg.window_dimensions)  # pyright: ignore[reportGeneralTypeIssues]

        self.camera.set_projection_matrix_perspective(*self.get_framebuffer_size())  # default viewing camera to perspective

        self.light = Camera([0.0, 15.0, 0.0], [0.00001, 1.0, 0.0])
        self.light.set_projection_matrix_perspective(*self.get_framebuffer_size())  # default viewing camera to perspective
        self.light_space_matrix = self.light.projection_matrix @ self.light.get_world_transform()

        self.shaders: Dict[str, Shader] = {}
        self.shader_ids: Dict[str, int] = {}
        self._prep_shaders()

        self.fboId: GL.GLint
        self._prep_background_image()

        self._initialize_shadow_buffer()

    def _prep_background_image(self) -> None:
        """ Initialize framebuffer object for background image, if specified. """

        # if nothing specified, return
        if not self.cfg.background_image_path:
            return

        # load background image
        _txtr: npt.NDArray[np.uint8] = read_background_image(self.cfg.background_image_path)

        # create the opengl texture and send it data
        self.txtr_h, self.txtr_w, _ = _txtr.shape
        self.txtr_id = GL.glGenTextures(1)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.txtr_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.txtr_w, self.txtr_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, _txtr)

        # make framebuffer object
        self.fboId: GL.GLint = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fboId)
        GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.txtr_id, 0)

    def _prep_shaders(self) -> None:
        COLOR_VERT = Path(resource_filename(__name__, "shaders/color.vert"))
        COLOR_FRAG = Path(resource_filename(__name__, "shaders/color.frag"))
        self._initiatize_shader('color_shader', str(COLOR_VERT), str(COLOR_FRAG))

        NORMAL_VERT = Path(resource_filename(__name__, "shaders/normal.vert"))
        NORMAL_FRAG = Path(resource_filename(__name__, "shaders/normal.frag"))
        self._initiatize_shader('normal_shader', str(NORMAL_VERT), str(NORMAL_FRAG))

        LIGHT_VERT = Path(resource_filename(__name__, "shaders/light.vert"))
        LIGHT_FRAG = Path(resource_filename(__name__, "shaders/light.frag"))
        self._initiatize_shader('light_shader', str(LIGHT_VERT), str(LIGHT_FRAG))

        SHADOW_VERT = Path(resource_filename(__name__, "shaders/shadow.vert"))
        SHADOW_FRAG = Path(resource_filename(__name__, "shaders/shadow.frag"))
        self._initiatize_shader('shadow_shader', str(SHADOW_VERT), str(SHADOW_FRAG), texture=True)

        TEXTURE_VERT = Path(resource_filename(__name__, "shaders/texture.vert"))
        TEXTURE_FRAG = Path(resource_filename(__name__, "shaders/texture.frag"))
        self._initiatize_shader('texture_shader', str(TEXTURE_VERT), str(TEXTURE_FRAG), texture=True)

    def _update_shaders_view_transform(self, camera: Camera) -> None:
        try:
            view_transform: npt.NDArray[np.float32] = np.linalg.inv(camera.get_world_transform())
        except Exception as e:
            msg = f'Error inverting camera world transform: {e}'
            logging.critical(msg)
            assert False, msg

        for shader_name in self.shaders:
            GL.glUseProgram(self.shader_ids[shader_name])
            view_loc = GL.glGetUniformLocation(self.shader_ids[shader_name], "proj_view")
            GL.glUniformMatrix4fv(view_loc, 1, GL.GL_TRUE, self.camera.projection_matrix @ view_transform)

            if shader_name == 'light_shader':
                viewpos_loc = GL.glGetUniformLocation(self.shader_ids[shader_name], "viewPos")
                GL.glUniform3fv(viewpos_loc, 1, camera.get_world_transform()[:-1, -1])

    def _initiatize_shader(self, shader_name: str, vert_path: str, frag_path: str, **kwargs) -> None:
        self.shaders[shader_name] = Shader(vert_path, frag_path)
        self.shader_ids[shader_name] = self.shaders[shader_name].glid  # pyright: ignore[reportGeneralTypeIssues]

        if shader_name == 'texture_shader':
            GL.glUseProgram(self.shader_ids[shader_name])
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_ids[shader_name], "texture_sampler"), 1)

        if shader_name == 'light_shader':
            # until we need a better way to handle scene lighting, this will go here
            light = {
                'position': np.array([0.0, 1.0, 0.0], np.float32),
                'ambient': np.array([0.4, 0.4, 0.4], np.float32),
                'diffuse': np.array([0.5, 0.5, 0.5], np.float32),
                'specular': np.array([0.5, 0.5, 0.5], np.float32),
            }

            GL.glUseProgram(self.shader_ids[shader_name])

            l_position_loc = GL.glGetUniformLocation(self.shader_ids[shader_name], "light.position")
            GL.glUniform3fv(l_position_loc, 1, light['position'])
            l_ambient_loc = GL.glGetUniformLocation(self.shader_ids[shader_name], "light.ambient")
            GL.glUniform3fv(l_ambient_loc, 1, light['ambient'])
            l_diffuse_loc = GL.glGetUniformLocation(self.shader_ids[shader_name], "light.diffuse")
            GL.glUniform3fv(l_diffuse_loc, 1, light['diffuse'])
            l_specular_loc = GL.glGetUniformLocation(self.shader_ids[shader_name], "light.specular")
            GL.glUniform3fv(l_specular_loc, 1, light['specular'])

            shadow_view_proj_matrix = self.light.projection_matrix @ np.linalg.inv(self.light.get_world_transform())
            shadow_view_proj_matrix_loc = GL.glGetUniformLocation(self.shader_ids['light_shader'], "shadow_view_proj_matrix")
            GL.glUniformMatrix4fv(shadow_view_proj_matrix_loc, 1, GL.GL_TRUE, shadow_view_proj_matrix)

    def _create_window(self, width: int, height: int) -> None:

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        glfw.make_context_current(self.win)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(*self.cfg.clear_color)

        logging.info(f'OpenGL Version: {GL.glGetString(GL.GL_VERSION).decode()}')  # pyright: ignore[reportGeneralTypeIssues]
        logging.info(f'GLSL: { GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode()}')  # pyright: ignore[reportGeneralTypeIssues]
        logging.info(f'Renderer: {GL.glGetString(GL.GL_RENDERER).decode()}')  # pyright: ignore[reportGeneralTypeIssues]

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def render(self, scene: Transform) -> None:
        # render depth buffer for shadows, if needed
        if self.cfg.draw_shadows:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_map_fbo)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

            self._update_shaders_view_transform(self.light)
            scene.draw(shader_ids=self.shader_ids, viewer_cfg=self.cfg)

        # prep for new scene render
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_map)
        self._update_shaders_view_transform(self.camera)

        # draw the background image, if exists
        if self.cfg.draw_background_image:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fboId)
            win_w, win_h = self.get_framebuffer_size()
            GL.glBlitFramebuffer(0, 0, self.txtr_w, self.txtr_h, 0, 0, win_w, win_h, GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)

        # draw the scene
        scene.draw(shader_ids=self.shader_ids, viewer_cfg=self.cfg)

    def clear_shadow_buffer(self) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_map_fbo)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def get_framebuffer_size(self) -> Tuple[int, int]:
        """ Return (width, height) of view's window. """
        return glfw.get_framebuffer_size(self.win)

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self.win)

    def clear_window(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # type: ignore

    def cleanup(self) -> None:
        """ Destroy the window when it's no longer being used. """
        glfw.destroy_window(self.win)

    def _initialize_shadow_buffer(self) -> None:

        self.shadow_width, self.shadow_height = self.get_framebuffer_size()

        # create depth texture
        self.depth_map = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_map)

        # allocate space for it
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)

        # set default filtering modes
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        # set up depth comparison mode
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_COMPARE_REF_TO_TEXTURE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LEQUAL)

        # set up wrapping modes
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

        # create fbo to render into
        self.depth_map_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_map_fbo)

        # attach depth texture to it
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.depth_map, 0)

        # disable color rendering
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        # bind original frame buffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
