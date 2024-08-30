# Copyright (c) Meta Platforms, Inc. and affiliates.

from animated_drawings.model.animated_drawing import AnimatedDrawing
from animated_drawings.model.transform import Transform
from animated_drawings.config import Config
from pkg_resources import resource_filename
import os
import pytest


@pytest.mark.skipif(os.environ.get('IS_CI_RUNNER') == 'True', reason='skipping video rendering for CI/CD')
def test_init():
    import OpenGL.GL as GL
    import glfw
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(100, 100, 'Viewer', None, None)
    glfw.make_context_current(win)

    mvc_cfg_fn = resource_filename(__name__, 'test_animated_drawing_files/test_mvc.yaml')
    mvc_config = Config(mvc_cfg_fn)
    char_cfg, retarget_cfg, motion_cfg = mvc_config.scene.animated_characters[0]

    viewer: Transform = Transform()

    AnimatedDrawing(char_cfg, retarget_cfg, motion_cfg, viewer)

    assert True


@pytest.mark.skipif(os.environ.get('IS_CI_RUNNER') == 'True', reason='skipping video rendering for CI/CD')
def test_character1():
    import OpenGL.GL as GL
    import glfw
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(100, 100, 'Viewer', None, None)
    glfw.make_context_current(win)

    mvc_cfg_fn = resource_filename(__name__, 'test_animated_drawing_files/test_mvc_character1.yaml')
    mvc_config = Config(mvc_cfg_fn)
    char_cfg, retarget_cfg, motion_cfg = mvc_config.scene.animated_characters[0]

    viewer: Transform = Transform()

    AnimatedDrawing(char_cfg, retarget_cfg, motion_cfg, viewer)

    assert True


@pytest.mark.skipif(os.environ.get('IS_CI_RUNNER') == 'True', reason='skipping video rendering for CI/CD')
def test_character2():
    import OpenGL.GL as GL
    import glfw
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(100, 100, 'Viewer', None, None)
    glfw.make_context_current(win)

    mvc_cfg_fn = resource_filename(__name__, 'test_animated_drawing_files/test_mvc_character2.yaml')
    mvc_config = Config(mvc_cfg_fn)
    char_cfg, retarget_cfg, motion_cfg = mvc_config.scene.animated_characters[0]

    viewer: Transform = Transform()

    AnimatedDrawing(char_cfg, retarget_cfg, motion_cfg, viewer)

    assert True


@pytest.mark.skipif(os.environ.get('IS_CI_RUNNER') == 'True', reason='skipping video rendering for CI/CD')
def test_character3():
    import OpenGL.GL as GL
    import glfw
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(100, 100, 'Viewer', None, None)
    glfw.make_context_current(win)

    mvc_cfg_fn = resource_filename(__name__, 'test_animated_drawing_files/test_mvc_character3.yaml')
    mvc_config = Config(mvc_cfg_fn)
    char_cfg, retarget_cfg, motion_cfg = mvc_config.scene.animated_characters[0]

    viewer: Transform = Transform()

    AnimatedDrawing(char_cfg, retarget_cfg, motion_cfg, viewer)

    assert True


@pytest.mark.skipif(os.environ.get('IS_CI_RUNNER') == 'True', reason='skipping video rendering for CI/CD')
def test_character4():
    import OpenGL.GL as GL
    import glfw
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(100, 100, 'Viewer', None, None)
    glfw.make_context_current(win)

    mvc_cfg_fn = resource_filename(__name__, 'test_animated_drawing_files/test_mvc_character4.yaml')
    mvc_config = Config(mvc_cfg_fn)
    char_cfg, retarget_cfg, motion_cfg = mvc_config.scene.animated_characters[0]

    viewer: Transform = Transform()

    AnimatedDrawing(char_cfg, retarget_cfg, motion_cfg, viewer)

    assert True
