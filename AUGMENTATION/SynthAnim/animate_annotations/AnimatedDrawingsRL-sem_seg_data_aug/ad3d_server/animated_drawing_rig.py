from __future__ import annotations

from ad3d_server.joint import Joint
from ad3d_server.transform import Transform
from ad3d_server.config import CharacterConfig
from ad3d_server.quaternions import Quaternions
from ad3d_server.vectors import Vectors

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple


class AnimatedDrawingRig(Transform):
    """ The skeletal rig used to deform the character """

    def __init__(self, char_cfg: CharacterConfig):
        """ Initializes character rig.  """
        super().__init__()

        # build initial dictionary with image keypoint annotations
        joints_d_: Dict[str, CharacterConfig.JointDict] = {joint['name']: joint for joint in char_cfg.skeleton}

        joints: Dict[str, Tuple[Joint, float]] = {}  # intermediary dictionary used to get hierarchy and local transform info

        for name, joint_d in joints_d_.items():  # set up intermediary dictionary

            joint = Joint(name=name)  # create it

            parent_name = joint_d['parent']  # get the parent name

            if parent_name:  # if has parent

                # compute parent-relative image-space coordinates
                loc = joints_d_[name]['loc'][0] - joints_d_[parent_name]['loc'][0], (1-joints_d_[name]['loc'][1]) - (1-joints_d_[parent_name]['loc'][1])

                starting_theta = np.arctan2(loc[1], loc[0])  # compute joint vector orientation within drawing

                joint.offset(np.array([loc[0], loc[1], 0]))  # set local space offset from parent

            else:  # has no parent, it's root

                loc = joints_d_[name]['loc'][0], 1-joints_d_[name]['loc'][1]

                starting_theta = 0

                joint.offset(np.array([loc[0], loc[1], 0]))

            joints[name] = joint, starting_theta

        for name, joint_d in joints_d_.items():  # use intermediary dictionary to create the rig

            joint, starting_theta = joints[name]
            parent_name = joint_d['parent']

            if not parent_name:
                self.root_joint = joint
                continue

            parent_joint = joints[parent_name][0]
            parent_joint.add_child(joint)

            joint.update_transforms()

            current_theta = np.arctan2(joint.get_world_position()[1] - parent_joint.get_world_position()[1],
                                       joint.get_world_position()[0] - parent_joint.get_world_position()[0])
            theta = starting_theta - current_theta

            # rotate the parent joint to properly orient the current joint
            parent_joint.set_rotation(Quaternions.from_angle_axis(np.array([theta]), Vectors([0, 0, 1])))
            parent_joint.update_transforms(recurse_on_children=True)

        # # create dictionary populated with joints
        # joints_d: Dict[str, Joint]
        # joints_d = {joint['name']: Joint(joint['name'], *joint['loc']) for joint in char_cfg.skeleton}

        # # assign joints within dictionary as childre of their parents
        # for joint_d in char_cfg.skeleton:
        #     if joint_d['parent'] is None:
        #         continue
        #     joints_d[joint_d['parent']].add_child(joints_d[joint_d['name']])

        # # updates joint positions to reflect local offsets from their parent joints
        # def _update_positions(t: Transform):
        #     """ Now that kinematic parent-> child chain is formed, subtract parent world positions to get actual child offsets"""
        #     parent: Optional[Transform] = t.get_parent()
        #     if parent is not None:
        #         offset = np.subtract(t.get_local_position(), parent.update_and_get_world_position())
        #         t.set_position(offset)
        #     for c in t.get_children():
        #         _update_positions(c)
        # _update_positions(joints_d['root'])

        # # attach root joint
        # self.root_joint = joints_d['root']
        # self.add_child(self.root_joint)

        # # self.root_joint.add_transorm_widget()

        self.add_child(self.root_joint)
        # cache for later
        self.joint_count = self.root_joint.joint_count()

        # set up buffer for visualizing vertices
        self.vertices = np.zeros([2 * (self.joint_count - 1), 6], np.float32)

        self._is_opengl_initialized: bool = False
        self._vertex_buffer_dirty_bit: bool = True

        # self.add_transorm_widget()

    def set_global_orientations(self, bvh_frame_orientations: Dict[str, float]) -> None:
        """ Applies orientation from bvh_frame_orientation to the rig. """

        self._set_global_orientations(self.root_joint, bvh_frame_orientations)
        self._vertex_buffer_dirty_bit = True

    def get_joint_positions(self) -> npt.NDArray[np.float32]:
        """ Returns array of 3D joints positions for rig, relative to AnimatedDrawingRig transform.  """
        wp = np.array(self.root_joint.get_chain_worldspace_positions()).reshape([-1, 3])
        ones = np.full((wp.shape[0], 1), 1)
        wp_1 = np.concatenate([wp, ones], axis=-1)
        inv_m = np.linalg.inv(self._world_transform)
        res = (inv_m @ wp_1.T).T

        return res[:, :3]

        # return np.array(self.root_joint.get_chain_worldspace_positions()).reshape([-1, 3])

    def get_joint_chain_length(self, joint_names_: List[str]) -> float:
        """ Given a list of joint names, computes the current euclidean distance from the first joint to the second,
        then to third, etc. """

        joint_names = joint_names_.copy()

        if len(joint_names) < 2:
            raise ValueError('attempted to compute length of joint chain with less than two joints.')

        total_length = 0.0

        current_joint = self.root_joint.get_transform_by_name(joint_names.pop(0))
        next_joint = self.root_joint.get_transform_by_name(joint_names.pop(0))

        while len(joint_names):

            c_pos = current_joint.update_and_get_world_position()
            n_pos = next_joint.update_and_get_world_position()

            total_length += np.linalg.norm(np.subtract(n_pos, c_pos))

            current_joint = next_joint
            next_joint = self.root_joint.get_transform_by_name(joint_names.pop(0))

        return float(total_length)

    def _compute_buffer_vertices(self, parent: Optional[Transform], pointer: List[int]) -> None:
        """ Recomputes values to pass to vertex buffer. Called recursively, pointer is List[int] to emulate pass-by-reference """
        if parent is None:
            parent = self.root_joint

        inv_m = np.linalg.inv(self._world_transform)

        for c in parent.get_children():
            if not isinstance(c, Joint):
                continue

            p1 = (inv_m @ np.concatenate([c.get_world_position(), [1]]))[:3]
            p2 = (inv_m @ np.concatenate([parent.get_world_position(), [1]]))[:3]

            self.vertices[pointer[0], :3] = p1
            self.vertices[pointer[0] + 1, :3] = p2
            pointer[0] += 2

            self._compute_buffer_vertices(c, pointer)

    def _set_global_orientations(self, joint: Joint, bvh_orientations: Dict[str, float]) -> None:
        if joint.name in bvh_orientations.keys():

            target_angle: float = bvh_orientations[str(joint.name)]

            # we shouldn't ever be trying to rotate the root
            parent_joint = joint.get_parent()
            if not parent_joint:
                raise ValueError

            # determine the bone segment's current angle within the plane
            parent_pos = np.linalg.inv(self._world_transform) @ np.array([*(parent_joint.get_world_position()), 1])
            joint_pos = np.linalg.inv(self._world_transform) @ np.array([*(joint.get_world_position()), 1])
            x, y, _, _ = joint_pos - parent_pos
            current_angle = np.arctan2(x, y)

            # compute the rotation needed to move from current_angle to target_angle
            rotation_q = Quaternions.from_angle_axis(np.array([current_angle - target_angle]), axes=Vectors([0.0, 0.0, 1.0]))

            # apply and update the transforms
            parent_joint.rotation_offset(rotation_q)
            parent_joint.update_transforms()

        # recurse on children
        for c in joint.get_children():
            if isinstance(c, Joint):
                self._set_global_orientations(c, bvh_orientations)

    @staticmethod
    def get_rig_from_char_cfg(char_cfg) -> AnimatedDrawingRig:
        if char_cfg.rig_type == '2D':
            return AnimatedDrawingRig(char_cfg)
        elif char_cfg.rig_type == '3D':
            return AnimatedDrawing3DRig(char_cfg)
        else:
            assert False, 'bad rig type'


class AnimatedDrawings3DJoint(Joint):
    """ Joints within Animated Drawings Rig."""

    def __init__(self, name: str, x: float, y: float, z: float):
        super().__init__(name=name, offset=np.array([x, 1 - y, z]))

        # self.starting_theta: float
        # self.current_theta: float


class AnimatedDrawing3DRig(AnimatedDrawingRig):
    def __init__(self, char_cfg: CharacterConfig):
        super().__init__(char_cfg)

    def set_global_bone_vectors(self, bvh_frame_positions: Dict[str, npt.NDArray[np.float32]]) -> None:
        self._set_global_bone_vectors(self.root_joint, bvh_frame_positions)
        self._vertex_buffer_dirty_bit = True

    def _set_global_bone_vectors(self, joint: Joint, bvh_bone_vectors: Dict[str, npt.NDArray[np.float32]]) -> None:
        if joint.name in bvh_bone_vectors.keys():
            target = Vectors(bvh_bone_vectors[joint.name])
            target.norm()

            rot_m = Quaternions.from_angle_axis(np.array([90]), Vectors(np.array([0.0, -1.0, 0.0]))).to_rotation_matrix()
            target = (rot_m @ np.array([list(target.vs[0]) + [0]]).T).T[0, :3]

            bone_length = Vectors(joint.update_and_get_world_position() - joint.get_parent().update_and_get_world_position()).length
            joint.set_position(bone_length * target)

        # recurse on children.
        for c in joint.get_children():
            if isinstance(c, Joint):
                self._set_global_bone_vectors(c, bvh_bone_vectors)

    def get_joint_positions(self) -> npt.NDArray[np.float32]:
        """ Returns array of 3D joints positions for rig.  """
        return np.array(self.root_joint.get_chain_worldspace_positions()).reshape([-1, 3])
