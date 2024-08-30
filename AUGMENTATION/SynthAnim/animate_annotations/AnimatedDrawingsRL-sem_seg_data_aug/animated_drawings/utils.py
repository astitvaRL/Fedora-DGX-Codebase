# Copyright (c) Meta Platforms, Inc. and affiliates.

from PIL import Image, ImageOps
import numpy as np
import numpy.typing as npt
import cv2
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from pkg_resources import resource_filename
from typing import List, Optional
import json

TOLERANCE = 10**-5


def resolve_ad_filepath(file_name: str, file_type: str) -> Path:
    """
    Given input filename, attempts to find the file, first by relative to cwd,
    then by absolute, the relative to animated_drawings root directory.
    If not found, prints error message indicating which file_type it is.
    """
    if Path(file_name).exists():
        return Path(file_name)
    elif Path.joinpath(Path.cwd(), file_name).exists():
        return Path.joinpath(Path.cwd(), file_name)
    elif Path(resource_filename(__name__, file_name)).exists():
        return Path(resource_filename(__name__, file_name))
    elif Path(resource_filename(__name__, str(Path('..', file_name)))).exists():
        return Path(resource_filename(__name__, str(Path('..', file_name))))

    msg = f'Could not find the {file_type} specified: {file_name}'
    logging.critical(msg)
    assert False, msg


def read_background_image(file_name: str) -> npt.NDArray[np.uint8]:
    """
    Given path to input image file, opens it, flips it based on EXIF tags, if present, and returns image with proper orientation.
    """
    # Check the file path
    file_path = resolve_ad_filepath(file_name, 'background_image')

    # Open the image and rotate as needed depending upon exif tag
    image = Image.open(str(file_path))
    image = ImageOps.exif_transpose(image)

    # Convert to numpy array and flip rightside up
    image_np = np.asarray(image)
    image_np = cv2.flip(image_np, 0)

    # Ensure we have RGBA
    if len(image_np.shape) == 3 and image_np.shape[-1] == 3:  # if RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    if len(image_np.shape) == 2:  # if grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGBA)

    return image_np.astype(np.uint8)


def load_mask(self, mask_p: Path, height: Optional[int] = None, width: Optional[int] = None, square_output_dim: Optional[int] = None) -> npt.NDArray[np.uint8]:
    """ Load and perform preprocessing upon the mask """
    try:
        _mask: npt.NDArray[np.uint8] = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        if height and _mask.shape[0] != height:
            raise AssertionError('height in character config and mask height do not match')
        if width and _mask.shape[1] != width:
            raise AssertionError('width in character config and mask height do not match')
    except Exception as e:
        msg = f'Error loading mask {mask_p}: {str(e)}'
        logging.critical(msg)
        assert False, msg

    _mask = np.rot90(_mask, 3, )  # rotate to upright

    # pad to square
    if square_output_dim:
        mask = np.zeros([self.char_cfg.img_dim, self.char_cfg.img_dim], _mask.dtype)
        mask[0:_mask.shape[0], 0:_mask.shape[1]] = _mask
    else:
        mask = _mask

    return mask


def visualize_mesh(vertices=None, triangles=None, edges=None, v_colors=[1, 0, 0], t_colors=[0, 0, 0], e_colors=[0, 1, 0], show_plot=True):
    if vertices is not None:
        x = vertices[:, 0]
        y = vertices[:, 1]

        max_val = max(np.max(x), np.max(y))
        plt.xlim([-20, max_val+20])
        plt.ylim([-20, max_val+20])

        if triangles is not None:
            for v012 in triangles:
                t_x, t_y = [], []
                t_x.extend([x[v] for v in v012])
                t_x.extend([x[v012[0]]])

                t_y.extend([y[v] for v in v012])
                t_y.extend([y[v012[0]]])

                plt.plot(t_x, t_y, c=t_colors)

        if edges is not None:
            for v01 in edges:
                e_x, e_y = [], []
                e_x.extend([x[v] for v in v01])

                e_y.extend([y[v] for v in v01])
                plt.plot(e_x, e_y, c=e_colors)

        plt.scatter(x, y, c=v_colors)
    if show_plot:
        plt.show()


def visualize_ad_meshes(meshes):  # mesh is AnimatedDrawingsMesh
    for mesh in meshes:
        visualize_mesh(mesh.vertices, mesh.triangles, show_plot=False)
    # plt.show()


def visualize_rig_joint_locations(positions_xy: npt.NDArray[np.float32], names: List[str]) -> None:
    plt.scatter(positions_xy[:, 0], positions_xy[:, 1])
    for i, name in enumerate(names):
        plt.annotate(name, (positions_xy[i, 0], positions_xy[i, 1]))
    plt.show()


def mirror_point( a, b, c, x1, y1):
    """ return mirror of x1, y1 when mirrored by ax + by + c = 0"""
    temp = -2 * (a * x1 + b * y1 + c) / (a * a + b * b)
    x = temp * a + x1
    y = temp * b + y1
    return (x, y)


def visualize_exported_json(file_name: str) -> None:

    with open(file_name, 'r') as f:
        data = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    line1, = ax.plot([], [], [], 'ro')

    def init():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return line1,

    def update(frame, ln):
        character_frame = frame[0]
        x_, y_, z_ = np.array(character_frame['verts_xyz']).T
        ln.set_data_3d(x_, y_, z_)
        return ln,

    FuncAnimation(
        fig,
        partial(update, ln=line1),
        frames=data['frames'],
        init_func=init,
        blit=True,
        interval=10
    )

    plt.show()


def view_gl_buffer_contents(width, height, buffer_type='color'):
    import OpenGL.GL as GL
    if buffer_type == 'color':
        data = GL.glReadPixels(0, 0, width, height, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, np.empty([height, width, 4]))
        color_array = np.flipud(data)
        Image.fromarray(color_array).show()
    elif buffer_type == 'stencil':
        data = GL.glReadPixels(0, 0, width, height, GL.GL_STENCIL_INDEX, GL.GL_UNSIGNED_BYTE)
        stencil_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        stencil_array = np.flipud(stencil_array)
        Image.fromarray(255 * stencil_array).show()
    elif buffer_type == 'depth':
        data = GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        depth_array = np.frombuffer(data, dtype=np.float32).reshape((height, width))
        depth_array = np.flipud(depth_array)

        depth_array = depth_array - np.min(depth_array)
        max_val = np.max(depth_array)
        if max_val != 0:
            depth_array *= (255 / max_val)
        Image.fromarray(depth_array).show()
