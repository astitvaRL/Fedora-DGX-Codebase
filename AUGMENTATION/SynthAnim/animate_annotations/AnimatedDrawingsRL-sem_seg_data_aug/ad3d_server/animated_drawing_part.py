from __future__ import annotations

from ad3d_server.transform import Transform
from ad3d_server.quaternions import Quaternions
from ad3d_server.vectors import Vectors
from ad3d_server.config import CharacterConfig
from ad3d_server.animated_drawing_mesh import AnimatedDrawingMeshCardboardBase
import ad3d_server.animated_drawing_utils as ad_utils

import yaml
from pathlib import Path
import math
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Dict, List
from collections import defaultdict


class AnimatedDrawingInternalPart(Transform):

    def __init__(self, part_dict: dict, mask: npt.NDArray, txtr: Optional[npt.NDArray], img_dim: int, animated_drawing):
        super().__init__()

        self.name: str = part_dict['name']
        self.parent_name: str = part_dict['parent_name']
        self.animated_drawing = animated_drawing

        self.img_dim = img_dim
        self.mask = mask
        self.txtr = txtr

        # stuff moved over from InternalPart goes here
        self.does_rdtwp: bool = part_dict['does_rdtwp']
        self.forward_orientation: Optional[str] = None if part_dict['forward_orientation'] == '' else part_dict['forward_orientation']
        self.hide_outside_parent: bool = part_dict['hide_outside_part']

        self.hide_on_backside: bool = part_dict['hide_on_backside']

        self.hide_between_view_angles: List[List[int, int]]
        try:
            self.hide_between_view_angles = part_dict['hide_between_view_angles']
        except KeyError:
            print("part doesn't have 'hide_between_view_angles'.")
            self.hide_between_view_angles = []

        self.is_visible: bool = True

        self.attached_to_mesh = None  # will be updated to another mesh if it's attached to it
        self.attachment_meshes: Optional[Dict] = None  # the meshes which this part ought to be attached to

        self.attachment_points: Dict[str, Dict[int, MeshAttachmentPoint]] = defaultdict(dict)  # an attachment point for each integer representing view angle. one dict for each possible mesh it can be attached to
        self.current_attachment_point: int = None

        _p = part_dict['point_approximation']
        _p[1], _p[0] = img_dim - _p[1], _p[0]
        self.point_approximation: Tuple[float, float] = [v / self.img_dim for v in _p]  # x, y

        # where the point approximation lies in image space
        self.as_drawn_image_space_point_location = np.identity(3)
        self.as_drawn_image_space_point_location[:2, -1] = self.point_approximation

        # if part is attached to another part, we need to know this when calculating the attachment points
        self.as_drawn_image_space_parent_point_location: Optional[npt.NDArray] = None

        self.mesh = None
        if self.has_visible_pixels():
            self.mesh = AnimatedDrawingMeshCardboardBase(self.mask, {'front': self.txtr})

            # define the mesh vertices relative to the point_approximation as drawn
            self.mesh.vertices[:, 0] -= self.as_drawn_image_space_point_location[0, -1]  # x location of vertex
            self.mesh.vertices[:, 1] -= self.as_drawn_image_space_point_location[1, -1]  # y location of vertex

            self.add_child(self.mesh)

        self.transforms: Dict[str, npt.NDArray] = {
            'as_drawn': np.identity(3),
            'left': np.array(part_dict['view_left_transform']),
            'right': np.array(part_dict['view_right_transform']),
        }
        self.transforms['left'][0, -1] /= self.img_dim
        self.transforms['left'][1, -1] /= self.img_dim
        self.transforms['right'][0, -1] /= self.img_dim
        self.transforms['right'][1, -1] /= self.img_dim

    def has_visible_pixels(self):
        """ if mask contains True values, this part has visible pixels """
        return len(self.mask[self.mask != 0]) > 0

    def _get_attachment_mesh(self):

        if self.parent_name in self.animated_drawing.name_to_part.keys():  # is attached to another part

            parent_part = self.animated_drawing.name_to_part[self.parent_name]

            self.as_drawn_image_space_parent_point_location = parent_part.as_drawn_image_space_point_location

            return {key: parent_part.mesh for key in self.animated_drawing.meshes.keys()}  # it's always the same part regardless of base mesh

        else:  # is attached to base mesh

            return self.animated_drawing.meshes

    def _compute_attachment_points(self):

        # find attachment points to use for every angle between 90 and 270, representing the front half of viewing angles

        point_t = np.identity(3)
        if self.as_drawn_image_space_parent_point_location is not None:  # if this part is attached to another part not base mesh
            point_t +=  self.as_drawn_image_space_point_location - self.as_drawn_image_space_parent_point_location
        else:
            point_t[:2, -1] = self.update_and_get_world_position()[:2]

        far_left = 90
        far_right = 270

        # ensure far left point is inside all meshes
        while True:
            point_left = (self.transforms['left'] @ point_t)[:2, -1]
            try:
                for name, mesh in self.attachment_meshes.items():
                    MeshAttachmentPoint(mesh, point_left)
            except Exception as e:
                print(f'error generating attachment point. Far left point outside mesh. Trying again. transform = {self.transforms["left"][0][2]}. error: {e}')
                self.transforms['left'][0][2] += 0.001
                continue
            break

        # ensure far right point is inside all meshes
        while True:
            point_right = (self.transforms['right'] @ point_t)[:2, -1]
            try:
                for name, mesh in self.attachment_meshes.items():
                    MeshAttachmentPoint(mesh, point_right)
            except Exception as e:
                print(f'error generating attachment point. Far right point outside mesh. Trying again. transform = {self.transforms["right"][0][2]}. error: {e}')
                self.transforms['right'][0][2] -= 0.001
                continue
            break

        for vdx in range(far_left, far_right):

            # added to make the movement of parts less 'smooth
            delta = 30
            # delta = 1
            jumped_vdx = round(vdx / delta) * delta

            point_left = self.transforms['left'] @ point_t
            point_right = self.transforms['right'] @ point_t

            interp_left = 1 - ((jumped_vdx-far_left)/(far_right-far_left))
            interp_right = 1 - interp_left

            interp_point = interp_left * point_left[:2, -1] + interp_right * point_right[:2, -1]
            for name, mesh in self.attachment_meshes.items():
                self.attachment_points[name][vdx] = MeshAttachmentPoint(mesh, interp_point)

    def set_attachment_mesh_and_compute_barycentric_coordinates(self):

        # get mesh this is attached to
        self.attachment_meshes: Dict = self._get_attachment_mesh()

        self._compute_attachment_points()

    def update(self, view_theta: float):
        """ called by update() in animated_drawing. view_theta comes from retargeter, is the angle it's being viewed from
        Sets the appropriate attachment point and adjusts visibility, if needed. """

        # if it's attached to another mesh, find the coords of attachment point and use to set position
        if self.attachment_meshes:
            current_mesh_attachment_points = self.attachment_points[self.animated_drawing.active_mesh_name]
            current_view_angle = max(90, min(269, round(view_theta)))
            attachment_point: MeshAttachmentPoint = current_mesh_attachment_points[current_view_angle]
            self.set_position(np.array([*attachment_point.compute_current_xy(), 0]))
            self.set_rotation(attachment_point.compute_local_rotation())

        self.is_visible = True

        if self.hide_on_backside:
            if round(view_theta) <= 90 or round(view_theta) >= 270:
                self.is_visible = False

        if self.hide_between_view_angles != [] and self.is_visible is True:
            for (low, high) in self.hide_between_view_angles:
                if low <= view_theta <= high:
                    self.is_visible = False
                    break

        self.update_transforms(recurse_on_children=True)

        for c in self.get_children():
            if type(c) is not AnimatedDrawingInternalPart:
                continue
            c.update(view_theta)

    # @staticmethod
    # def generate_base_mesh_names_and_masks(parts_fn: str, char_cfg: CharacterConfig, name_to_part_dict) -> List[AnimatedDrawingInternalPart]:
    #     # get parts exported from annotation tool view script
    #     with open(parts_fn, 'r') as f:
    #         part_dicts = yaml.safe_load(f)

    #     ret = {}
    #     for part_dict in part_dicts:

    #         # if not a direct parent of base mesh, ignore
    #         if part_dict['parent_name'] != 'Full Character':
    #             continue

    #         # if doesn't move relative to base mesh, ignore
    #         if part_dict['does_rdtwp'] == True:
    #             continue

    #         # load part mask
    #         mask_p = Path(parts_fn).parent/'masks'/f'{part_dict["name"]}.png'
    #         mask = ad_utils.load_mask(mask_p, char_cfg.img_dim)
    #         mask = ad_utils.load_txtr(mask_p, self.char_view_cfg.img_dim, self.char_view_cfg.img_height, self.char_view_cfg.img_width)
    #         mask[:, :, 3] = mask[:, :, 0]  # masks are rgb black and white. use b channel as alpha

    #         ret[part_dict['name']] = mask
    #     return ret

    @staticmethod
    def generate_parts(parts_fn: str, char_cfg: CharacterConfig, animated_drawing) -> List[AnimatedDrawingInternalPart]:
        # get parts exported from annotation tool view script
        with open(parts_fn, 'r') as f:
            part_dicts = yaml.safe_load(f)

        ret_parts: List[AnimatedDrawingInternalPart] = []
        for part_dict in part_dicts:
            # if it's an external part, it's just segmentation of base mesh not a real part
            if part_dict['type'] == 'external':
                continue

            # load part mask
            mask_p = Path(parts_fn).parent/'masks'/f'{part_dict["name"]}.png'
            mask = ad_utils.load_mask(mask_p, char_cfg.img_dim)

            # load part txtr
            txtr_p = Path(parts_fn).parent/'texture'/f'{part_dict["name"]}.png'
            if txtr_p.exists():
                txtr = ad_utils.load_txtr(txtr_p, char_cfg.img_dim)
            else:
                txtr = None

            # TODO: adjust annotations so subsections of base mesh, like Face, isn't it's own part
            if part_dict['name'] == 'Face':
                continue

            # generate the part
            ret_parts.append(AnimatedDrawingInternalPart(part_dict, mask, txtr, char_cfg.img_dim, animated_drawing))

        # create dictionary of parts
        name_to_part: Dict[str, AnimatedDrawingInternalPart] = {}
        for part in ret_parts:
            name_to_part[part.name] = part

        # create the tree
        _parts_attached_to_base_mesh = []
        for part in ret_parts:
            if part.parent_name in name_to_part.keys():
                name_to_part[part.parent_name].add_child(part)
                parent_drawn_location = name_to_part[part.parent_name].as_drawn_image_space_point_location[:2, -1]
            else:
                _parts_attached_to_base_mesh.append(part)
                parent_drawn_location = animated_drawing.rig.root_joint.update_and_get_world_position()[:2]

            part_drawn_location = part.as_drawn_image_space_point_location[:2, -1]

            part.set_position(np.array([*(part_drawn_location - parent_drawn_location), 0]))

        # update transforms
        for part in name_to_part.values():
            part.update_transforms()

        return _parts_attached_to_base_mesh, name_to_part


class MeshAttachmentPoint:
    def __init__(self, mesh: AnimatedDrawingMeshCardboardBase, xy: npt.NDArray[np.float32]):
        """ Given a mesh and the xy coordinates of a point, calculates and stores the barycentric coordinates of that point within the mesh """

        self.mesh = mesh  # the mesh this point will define attachment to

        self.b_coords: Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]
        self.b_coords_left: Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]
        self.b_coords_right: Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]
        self._compute_b_coords(xy)

    def _compute_b_coords(self, xy: npt.NDArray[np.float32]):
        """ Compute barycentric coordinates of attachment point to mesh, as well as a a part of initially horizontal points near attachment point """

        mesh_front_vertices_xy = self.mesh.vertices[:self.mesh.front_vertex_count, :2]
        mesh_front_triangles = self.mesh.triangles[self.mesh.front_triangles_start:self.mesh.front_triangles_end]
        self.b_coords = ad_utils.xy_to_barycentric_coords(np.expand_dims(xy, axis=0), mesh_front_vertices_xy, mesh_front_triangles)[0][0]

        # to determine the local rotation of the mesh at the point, we will find a point slightly to the left and right of it
        delta = 0.001
        try:
            self.b_coords_left = ad_utils.xy_to_barycentric_coords(np.expand_dims(xy + np.array([-delta, 0]), axis=0), mesh_front_vertices_xy, mesh_front_triangles)[0][0]
        except Exception:
            self.b_coords_left = self.b_coords

        try:
            self.b_coords_right = ad_utils.xy_to_barycentric_coords(np.expand_dims(xy + np.array([delta, 0]), axis=0), mesh_front_vertices_xy, mesh_front_triangles)[0][0]
        except Exception:
            self.b_coords_right = self.b_coords

        assert self.b_coords_left is not self.b_coords_right, "Error find unique horizontal points in the local neighborhood of attachment point"

    def compute_current_xy(self) -> npt.NDArray[np.float32]:
        """ returns the xy coordinates of the point in self.mesh's local (vertex) space """
        current_xy = np.array([0.0, 0.0], dtype=np.float32)
        for vert_id, vert_weight in self.b_coords:
            current_xy += vert_weight * self.mesh.vertices[vert_id, :2]
        return current_xy

    def compute_local_rotation(self) -> Quaternions:
        """ using points slightly to the left and to the right of attachment point when mesh is not deformed, compute the rotation at attachment point and return as quaternion """
        left_point_xy = np.array([0.0, 0.0], dtype=np.float32)
        for vert_id, vert_weight in self.b_coords_left:
            left_point_xy += vert_weight * self.mesh.vertices[vert_id, :2]

        right_point_xy = np.array([0.0, 0.0], dtype=np.float32)
        for vert_id, vert_weight in self.b_coords_right:
            right_point_xy += vert_weight * self.mesh.vertices[vert_id, :2]

        right_point_xy -= left_point_xy
        left_point_xy -= left_point_xy
        theta_rads = math.atan2(right_point_xy[1], right_point_xy[0])
        return Quaternions.from_angle_axis(np.array([theta_rads]), Vectors([0, 0, 1]))
