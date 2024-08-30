import numpy.typing as npt
import numpy as np
from pathlib import Path
import logging
import cv2
from typing import Optional, Tuple, List


def load_mask(mask_p: Path, output_dim: int, expected_height: Optional[int] = None, expected_width: Optional[int] = None) -> npt.NDArray[np.uint8]:
    """ Performs tasks related to going from binary mask file to numpy array suitable for use by AnimatedDrawing """

    # load the file
    try:
        _mask: npt.NDArray[np.uint8] = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        if expected_height is not None and _mask.shape[0] != expected_height:  # optional height check
            raise AssertionError(f'expected_height and mask height do not match: {expected_height} vs. {_mask.shape[0]}')
        if expected_width is not None and _mask.shape[1] != expected_width:  # optional width check
            raise AssertionError(f'expected width and mask width do not match: {expected_width} vs. {_mask.shape[1]}')
    except Exception as e:
        msg = f'Error loading mask {mask_p}: {str(e)}'
        logging.critical(msg)
        assert False, msg

    # rotate to upright
    _mask = np.rot90(_mask, 3, )

    # pad to square
    mask = np.zeros([output_dim, output_dim], _mask.dtype)
    mask[0:_mask.shape[0], 0:_mask.shape[1]] = _mask

    return mask


def load_txtr(txtr_p: Path, output_dim: int, expected_height: Optional[int] = None, expected_width: Optional[int] = None) -> npt.NDArray[np.uint8]:
    """ Performs tasks related to going from rgba image file to numpy array suitable for use by AnimatedDrawing """

    # load the file
    try:
        _txtr: npt.NDArray[np.uint8] = cv2.imread(str(txtr_p), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_UNCHANGED).astype(np.uint8)
        _txtr = cv2.cvtColor(_txtr, cv2.COLOR_BGRA2RGBA).astype(np.uint8)
        if _txtr.shape[-1] != 4:
            raise AssertionError('texture must be RGBA')
        if expected_height is not None and _txtr.shape[0] != expected_height:  # optional height check
            raise AssertionError(f'expected_height and txtr height do not match: {expected_height} vs. {_txtr.shape[0]}')
        if expected_width is not None and _txtr.shape[1] != expected_width:  # optional width check
            raise AssertionError(f'expected width and txtr width do not match: {expected_width} vs. {_txtr.shape[1]}')
    except Exception as e:
        msg = f'Error loading texture {txtr_p}: {str(e)}'
        logging.critical(msg)
        assert False, msg

    # rotate to upright
    _txtr = np.rot90(_txtr, 3, )

    # pad to square
    txtr = np.zeros([output_dim, output_dim, _txtr.shape[-1]], _txtr.dtype)
    txtr[0:_txtr.shape[0], 0:_txtr.shape[1], :] = _txtr

    return txtr


def xy_to_barycentric_coords(
        points: npt.NDArray[np.float32],
        vertices: npt.NDArray[np.float32],
        triangles: npt.NDArray[np.int32]
        ) -> Tuple[
        List[Tuple[Tuple[np.int32, np.float32],
        Tuple[np.int32, np.float32],
        Tuple[np.int32, np.float32]]],
        npt.NDArray[np.bool8]
]:
    """
    Given and array containing xy locations and the vertices & triangles making up a mesh,
    find the triangle that each points in within and return it's representation using barycentric coordinates.
    points: ndarray [N,2] of point xy coords
    vertices: ndarray of vertex locations, row position is index id
    triangles: ndarraywith ordered vertex ids of vertices that make up each mesh triangle

    Is point inside triangle? : https://mathworld.wolfram.com/TriangleInterior.html

    Returns a list of barycentric coords for points inside the mesh,
    and a list of True/False values indicating whether a given pin was inside the mesh or not.
    Needed for removing pins during subsequent solve steps.

    """
    def det(u: npt.NDArray[np.float32], v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """ helper function returns determinents of two [N,2] arrays"""
        ux, uy = u[:, 0], u[:, 1]
        vx, vy = v[:, 0], v[:, 1]
        return ux*vy - uy*vx

    tv_locs: npt.NDArray[np.float32] = np.asarray([vertices[t].flatten() for t in triangles])  # triangle->vertex locations, [T x 6] array

    v0 = tv_locs[:, :2]
    v1 = np.subtract(tv_locs[:, 2:4], v0)
    v2 = np.subtract(tv_locs[:, 4: ], v0)

    b_coords: List[Tuple[Tuple[np.int32, np.float32], Tuple[np.int32, np.float32], Tuple[np.int32, np.float32]]] = []
    pin_mask: List[bool] = []

    for p_xy in points:

        p_xy = np.expand_dims(p_xy, axis=0)
        a =  (det(p_xy, v2) - det(v0, v2)) / det(v1, v2)
        b = -(det(p_xy, v1) - det(v0, v1)) / det(v1, v2)

        # find the indices of triangle containing
        in_triangle = np.bitwise_and(np.bitwise_and(a > 0, b > 0), a + b < 1)
        containing_t_idxs = np.argwhere(in_triangle)

        # if length is zero, check if on triangle(s) perimeters
        if not len(containing_t_idxs):
            on_triangle_perimeter = np.bitwise_and(np.bitwise_and(a >= 0, b >= 0), a + b <= 1)
            containing_t_idxs = np.argwhere(on_triangle_perimeter)

        # point is outside mesh. Log a warning and continue
        if not len(containing_t_idxs):
            msg = f'point {p_xy} not inside or on edge of any triangle in mesh. Skipping it'
            print(msg)
            logging.warning(msg)
            pin_mask.append(False)
            continue

        # grab the id of first triangle the point is in or on
        t_idx = int(containing_t_idxs[0])

        vertex_ids = triangles[t_idx]                               # get ids of verts in triangle
        a_xy, b_xy, c_xy = vertices[vertex_ids]                     # get xy coords of verts
        uvw = get_barycentric_coords(p_xy, a_xy, b_xy, c_xy)  # get barycentric coords
        b_coords.append(list(zip(vertex_ids, uvw)))                 # append to our list  # pyright: ignore[reportGeneralTypeIssues]
        pin_mask.append(True)

    return (b_coords, np.array(pin_mask, dtype=np.bool8))


def get_barycentric_coords(
                            p: npt.NDArray[np.float32],
                            a: npt.NDArray[np.float32],
                            b: npt.NDArray[np.float32],
                            c: npt.NDArray[np.float32]
                            ) -> npt.NDArray[np.float32]:
    """
    As described in Christer Ericson's Real-Time Collision Detection.
    p: the input point
    a, b, c: the vertices of the triangle

    Returns ndarray [u, v, w], the barycentric coordinates of p wrt vertices a, b, c
    """
    v0: npt.NDArray[np.float32] = np.subtract(b, a)
    v1: npt.NDArray[np.float32] = np.subtract(c, a)
    v2: npt.NDArray[np.float32] = np.subtract(p, a)
    d00: np.float32 = np.dot(v0, v0)
    d01: np.float32 = np.dot(v0, v1)
    d11: np.float32 = np.dot(v1, v1)
    d20: np.float32 = np.dot(v2, v0)
    d21: np.float32 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v: npt.NDArray[np.float32] = (d11 * d20 - d01 * d21) / denom  # pyright: ignore[reportGeneralTypeIssues]
    w: npt.NDArray[np.float32] = (d00 * d21 - d01 * d20) / denom  # pyright: ignore[reportGeneralTypeIssues]
    u: npt.NDArray[np.float32] = 1.0 - v - w

    return np.array([u, v, w]).squeeze()
