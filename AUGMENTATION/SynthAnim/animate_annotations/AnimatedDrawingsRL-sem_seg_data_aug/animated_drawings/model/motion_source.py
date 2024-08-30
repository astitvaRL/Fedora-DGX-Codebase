from animated_drawings.model.bvh import BVH
from animated_drawings.model.quaternions import Quaternions
from animated_drawings.model.vectors import Vectors
from animated_drawings.model.joint import Joint
from animated_drawings.config import MotionConfig
from animated_drawings.model.transform import Transform
from animated_drawings.model.box import Box
import numpy as np
import numpy.typing as npt
import logging
from abc import abstractclassmethod
from typing import List
import math


class MotionSource():
    def __init__(self, motion_cfg: MotionConfig):
        self.motion_cfg = motion_cfg

    @abstractclassmethod
    def get_joint_names(self) -> List[str]:
        raise NotImplementedError

    @abstractclassmethod
    def get_max_frame_count(self) -> int:
        raise NotImplementedError

    @abstractclassmethod
    def get_frame_time(self) -> float:
        raise NotImplementedError

    @abstractclassmethod
    def apply_frame_rotations(self, frame_idx: int):
        raise NotImplementedError

    @abstractclassmethod
    def apply_frame_root_position(self, frame_idx: int):
        raise NotImplementedError

    @abstractclassmethod
    def apply_frame(self, frame_idx: int) -> None:
        raise NotImplementedError

    @abstractclassmethod
    def get_joint_chain_length(self, joint_names_: List[str]) -> float:
        raise NotImplementedError

    @abstractclassmethod
    def get_fwd_vector(self) -> npt.NDArray[np.float32]:
        raise NotImplementedError

    @abstractclassmethod
    def get_current_joint_positions(self) -> npt.NDArray[np.float32]:
        raise NotImplementedError

    @abstractclassmethod
    def get_current_joint_position_from_name(self, joint_name: str) -> npt.NDArray[np.float32]:
        raise NotImplementedError

    @abstractclassmethod
    def get_root_position(self) -> npt.NDArray[np.float32]:
        raise NotImplementedError

    @abstractclassmethod
    def get_visualization_transform(self) -> Transform:
        raise NotImplementedError

    @staticmethod
    def create_motion_source(motion_cfg: MotionConfig):
        if motion_cfg.source_type == 'file_bvh':
            return MotionSourceBVHFile(motion_cfg)
        elif motion_cfg.source_type == 'file_jointpositionarray':
            return MotionSourceJointPositionArray(motion_cfg)
        elif motion_cfg.source_type == 'stream_jointpositionarray':
            return MotionSourceJointPositionArrayStream(motion_cfg)
        else:
            raise NotImplementedError


class MotionSourceBVHFile(MotionSource):
    def __init__(self, motion_cfg: MotionConfig) -> None:
        super().__init__(motion_cfg)

        try:
            self.bvh = BVH.from_file(str(self.motion_cfg.ms_p), self.motion_cfg.start_frame_idx, self.motion_cfg.end_frame_idx)
        except Exception as e:
            msg = f'Error loading BVH: {e}'
            logging.critical(msg)
            assert False, msg

        self._set_initial_bvh_position()

    def _set_initial_bvh_position(self) -> None:

        # rotate so +y is up
        if self.motion_cfg.up == '+y':
            pass  # no rotation needed
        elif self.motion_cfg.up == '+z':
            self.bvh.set_rotation(Quaternions.from_euler_angles('yx', np.array([-90.0, -90.0])))
        else:
            msg = f'up value not implemented: {self.motion_cfg.up}'
            logging.critical(msg)
            raise NotImplementedError(msg)

        # rotate so forward is +z
        skeleton_fwd: Vectors = self.bvh.get_skeleton_fwd(self.motion_cfg.forward_perp_joint_vectors)
        q: Quaternions = Quaternions.rotate_between_vectors(skeleton_fwd, Vectors([0.0, 0.0, 1.0]))
        self.bvh.rotation_offset(q)

        # scale BVH
        self.bvh.set_scale(self.motion_cfg.scale)

        # position above origin
        self.bvh.offset(-self.bvh.root_joint.update_and_get_world_position())

        # adjust bvh skeleton y pos by getting groundplane joint...
        try:
            groundplane_joint = self.bvh.root_joint.get_transform_by_name(self.motion_cfg.groundplane_joint)
            assert isinstance(groundplane_joint, Joint), f'could not find joint by name: {self.motion_cfg.groundplane_joint}'
        except Exception as e:
            msg = f'Error getting groundplane joint: {e}'
            logging.warning(msg)
            assert False

        # ... and moving the bvh so it is on the y=0 plane
        bvh_groundplane_y = groundplane_joint.update_and_get_world_position()[1]
        self.bvh.offset(np.array([0, -bvh_groundplane_y, 0]))

    def get_joint_names(self) -> List[str]:
        return self.bvh.get_joint_names()

    def apply_frame(self, frame_idx: int) -> None:
        self.bvh.apply_frame(frame_idx)

    def apply_frame_rotations(self, frame_idx: int):
        self.bvh.apply_frame_rotations(frame_idx)

    def get_joint_chain_length(self, joint_names_: List[str]) -> float:
        """ Given a list of joint names, computes the current euclidean distance from the first joint to the second,
        then to third, etc. """

        joint_names = joint_names_.copy()

        if len(joint_names) < 2:
            raise ValueError('attempted to compute length of joint chain with less than two joints.')

        total_length = 0.0

        current_joint = self.bvh.root_joint.get_transform_by_name(joint_names.pop(0))
        next_joint = self.bvh.root_joint.get_transform_by_name(joint_names.pop(0))

        while len(joint_names):

            c_pos = current_joint.update_and_get_world_position()
            n_pos = next_joint.update_and_get_world_position()

            total_length += np.linalg.norm(np.subtract(n_pos, c_pos))

            current_joint = next_joint
            next_joint = self.bvh.get_transform_by_name(joint_names.pop(0))

        return float(total_length)

    def get_fwd_vector(self) -> npt.NDArray[np.float32]:
        return self.bvh.get_skeleton_fwd(self.motion_cfg.forward_perp_joint_vectors).vs[0]

    def get_normalized_joint_positions(self) -> npt.NDArray[np.float32]:
        """ Returns joint positions. Per frame, zeros out their XZ root positions and rotates skeleton so it faces along +X axis. """

        # get joint positions and forward vectors
        joint_positions = np.empty([self.bvh.frame_max_num, 3 * self.bvh.joint_num], dtype=np.float32)
        fwd_vectors = np.empty([self.bvh.frame_max_num, 3], dtype=np.float32)

        for frame_idx in range(self.bvh.frame_max_num):
            self.bvh.apply_frame(frame_idx)
            joint_positions[frame_idx] = self.bvh.root_joint.get_chain_worldspace_positions()
            fwd_vectors[frame_idx] = self.get_fwd_vector()

        # reposition over origin
        self.bvh_root_positions = joint_positions[:, :3]
        joint_positions = joint_positions - np.tile(self.bvh_root_positions, [1, len(self.get_joint_names())])

        # compute angle between skelton's forward vector and x axis
        v1 = np.tile(np.array([1.0, 0.0], dtype=np.float32), reps=(joint_positions.shape[0], 1))
        v2 = fwd_vectors
        dot: npt.NDArray[np.float32] = v1[:, 0]*v2[:, 0] + v1[:, 1]*v2[:, 2]
        det: npt.NDArray[np.float32] = v1[:, 0]*v2[:, 2] - v2[:, 0]*v1[:, 1]
        angle: npt.NDArray[np.float32] = np.arctan2(det, dot).astype(np.float32)
        angle %= 2*np.pi
        angle = np.where(angle < 0.0, angle + 2*np.pi, angle)

        # rotate the skeleton's joint so it faces +X axis
        for idx in range(joint_positions.shape[0]):
            rot_mat = np.identity(3).astype(np.float32)
            rot_mat[0, 0] = math.cos(angle[idx])
            rot_mat[0, 2] = math.sin(angle[idx])
            rot_mat[2, 0] = -math.sin(angle[idx])
            rot_mat[2, 2] = math.cos(angle[idx])

            rotated_joints: npt.NDArray[np.float32] = rot_mat @ joint_positions[idx].reshape([-1, 3]).T
            joint_positions[idx] = rotated_joints.T.reshape(joint_positions[idx].shape)

        return joint_positions

    def get_max_frame_count(self) -> int:
        return self.bvh.frame_max_num

    def get_frame_time(self) -> float:
        return self.bvh.frame_time

    def get_current_joint_positions(self) -> npt.NDArray[np.float32]:
        return np.array(self.bvh.root_joint.get_chain_worldspace_positions(), dtype=np.float32).reshape([len(self.get_joint_names()), 3])

    def get_current_joint_position_from_name(self, joint_name: str) -> npt.NDArray[np.float32]:
        try:
            joint_idx = self.get_joint_names().index(joint_name)
        except ValueError as e:
            logging.critical(e)
            assert False
        return self.get_current_joint_positions()[joint_idx, :]

    def get_root_position(self) -> npt.NDArray[np.float32]:
        return self.bvh.root_joint.update_and_get_world_position()

    def get_visualization_transform(self) -> Transform:
        return self.bvh


class MotionSourceBVHStream(MotionSource):
    def __init__(self, motion_cfg: MotionConfig):
        raise NotImplementedError
        pass

        self.joint_names = self.bvh.get_joint_names()

    def apply_frame(self, frame_idx: int):
        """ Setting the frame doesn't do anything if the motion source is a stream. """
        pass


class MotionSourceJointPositionArray(MotionSource):

    def __init__(self, motion_cfg: MotionConfig) -> None:
        super().__init__(motion_cfg)

        self.joint_positions = np.load(self.motion_cfg.ms_p)

        assert len(self.joint_positions.shape) == 3  # should be [framenum, jointnum, xyz]
        assert self.joint_positions.shape[1] == len(self.get_joint_names())
        assert self.joint_positions.shape[2] == 3

        # scale
        self.joint_positions *= self.motion_cfg.scale

        # translate
        self.joint_positions[:, :, 1] -= self.joint_positions[0, self.get_joint_names().index(self.motion_cfg.groundplane_joint), 1]
        self.joint_positions[:, :, 0] -= 1 - self.joint_positions[0, self.get_joint_names().index(self.motion_cfg.groundplane_joint), 0]
        self.joint_positions[:, :, 2] -= self.joint_positions[0, self.get_joint_names().index(self.motion_cfg.groundplane_joint), 2]

        self.current_frame = 0

        self.widget = Transform()
        for _ in range(len(self.get_joint_names())):
            self.widget.add_child(Box(s=0.03, c=[0.4, 0.9, 0.6]))

    def get_joint_names(self) -> List[str]:
        return self.motion_cfg.jointpositionarray_jointnames

    def apply_frame(self, frame_idx: int) -> None:
        self.current_frame = frame_idx % self.get_max_frame_count()

        # update visualization widget
        for idx, box in enumerate(self.widget.get_children()):
            box.set_position(self.joint_positions[self.current_frame, idx, :])

    def apply_frame_rotations(self, frame_idx: int):
        """ no need to do anything """
        pass

    def get_joint_chain_length(self, joint_names_: List[str]) -> float:
        joint_names = joint_names_.copy()

        if len(joint_names) < 2:
            raise ValueError('attempted to compute length of joint chain with less than two joints.')

        current_joint_name = joint_names.pop(0)
        next_joint_name = joint_names.pop(0)

        total_length = 0.0
        while len(joint_names):

            c_pos = self.joint_positions[self.current_frame, self.get_joint_names().index(current_joint_name), :]
            n_pos = self.joint_positions[self.current_frame, self.get_joint_names().index(next_joint_name), :]

            total_length += np.linalg.norm(np.subtract(n_pos, c_pos))
            current_joint_name = next_joint_name
            next_joint_name = joint_names.pop(0)

        return total_length

    def get_fwd_vector(self) -> npt.NDArray[np.float32]:

        vectors_cw_perpendicular_to_fwd: List[Vectors] = []

        for (start_joint_name, end_joint_name) in self.motion_cfg.forward_perp_joint_vectors:
            end_joint_xyz = self.get_current_joint_position_from_name(end_joint_name)
            start_joint_xyz = self.get_current_joint_position_from_name(start_joint_name)

            bone_vector: Vectors = Vectors(end_joint_xyz) - Vectors(start_joint_xyz)
            bone_vector.norm()
            vectors_cw_perpendicular_to_fwd.append(bone_vector)

        return Vectors(vectors_cw_perpendicular_to_fwd).average().perpendicular().vs[0]

    def get_normalized_joint_positions(self) -> npt.NDArray[np.float32]:
        # Not called anymore
        raise NotImplementedError

    def get_max_frame_count(self) -> int:
        return self.joint_positions.shape[0]

    def get_frame_time(self) -> float:
        return self.motion_cfg.jointpositionarray_frametime

    def get_current_joint_positions(self) -> npt.NDArray[np.float32]:
        return self.joint_positions[self.current_frame, :, :].copy()

    def get_current_joint_position_from_name(self, joint_name: str) -> npt.NDArray[np.float32]:
        try:
            joint_idx = self.get_joint_names().index(joint_name)
        except ValueError as e:
            logging.critical(e)
            assert False

        return self.get_current_joint_positions()[joint_idx, :]

    def get_root_position(self) -> npt.NDArray[np.float32]:
        return self.get_current_joint_position_from_name(self.motion_cfg.jointpositionarray_root_joint_name)

    def get_visualization_transform(self) -> Transform:
        return self.widget


class MotionSourceJointPositionArrayStream(MotionSource):

    def __init__(self, motion_cfg: MotionConfig) -> None:
        super().__init__(motion_cfg)

        self.joint_positions_contain_valid_pose = False
        self.joint_positions = np.zeros([1, len(self.motion_cfg.jointpositionarray_jointnames), 3])

        self.widget = Transform()
        for _ in range(len(self.get_joint_names())):
            self.widget.add_child(Box(s=0.03, c=[0.4, 0.9, 0.6]))

    def _reposition(self, joint_positions: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # TODO: for development purposes. Joint positions passed to AD should be at a reseasonable scale and location
        # scale
        # joint_positions *= self.motion_cfg.scale

        # translate
        # joint_positions[:, 1] -= joint_positions[self.get_joint_names().index(self.motion_cfg.groundplane_joint), 1]
        # joint_positions[:, 0] -= 1 - joint_positions[self.get_joint_names().index(self.motion_cfg.groundplane_joint), 0]
        # joint_positions[:, 2] -= joint_positions[self.get_joint_names().index(self.motion_cfg.groundplane_joint), 2]

        return joint_positions

    def get_joint_names(self) -> List[str]:
        return self.motion_cfg.jointpositionarray_jointnames

    def apply_frame(self, frame_idx: int) -> None:
        """ this doesn't need to do anything, as set_joint_positions() will be updating things. """
        pass

    def set_joint_positions(self, new_joint_positions: npt.NDArray[np.float32]) -> None:
        assert new_joint_positions.shape == self.joint_positions.shape[1:]

        self.joint_positions[0] = self._reposition(new_joint_positions)
        self.joint_positions_contain_valid_pose = True

        # update visualization widget
        for idx, box in enumerate(self.widget.get_children()):
            box.set_position(self.joint_positions[0, idx, :])

    def apply_frame_rotations(self, frame_idx: int):
        """ no need to do anything """
        pass

    def get_joint_chain_length(self, joint_names_: List[str]) -> float:
        if not self.joint_positions_contain_valid_pose:  # if no valid joint locations are in data, return -1
            return -1
        # TODO: refactor retargeter so it doesn't always call get_joint_chain_length at initialization when using streaming sources

        joint_names = joint_names_.copy()

        if len(joint_names) < 2:
            raise ValueError('attempted to compute length of joint chain with less than two joints.')

        current_joint_name = joint_names.pop(0)
        next_joint_name = joint_names.pop(0)

        total_length = 0.0
        while len(joint_names):

            c_pos = self.joint_positions[0, self.get_joint_names().index(current_joint_name), :]
            n_pos = self.joint_positions[0, self.get_joint_names().index(next_joint_name), :]

            total_length += np.linalg.norm(np.subtract(n_pos, c_pos))
            current_joint_name = next_joint_name
            next_joint_name = joint_names.pop(0)

        return total_length

    def get_fwd_vector(self) -> npt.NDArray[np.float32]:

        vectors_cw_perpendicular_to_fwd: List[Vectors] = []

        for (start_joint_name, end_joint_name) in self.motion_cfg.forward_perp_joint_vectors:
            end_joint_xyz = self.get_current_joint_position_from_name(end_joint_name)
            start_joint_xyz = self.get_current_joint_position_from_name(start_joint_name)

            bone_vector: Vectors = Vectors(end_joint_xyz) - Vectors(start_joint_xyz)
            bone_vector.norm()
            vectors_cw_perpendicular_to_fwd.append(bone_vector)

        return Vectors(vectors_cw_perpendicular_to_fwd).average().perpendicular().vs[0]

    def get_normalized_joint_positions(self) -> npt.NDArray[np.float32]:
        assert False
        # Not called anymore
        raise NotImplementedError

    def get_max_frame_count(self) -> int:
        return 1  # TODO: refactor retargeter so this isn't called when using streaming sources

    def get_frame_time(self) -> float:
        return 1  # TODO: refactor retargeter so this isn't called when using streaming sources

    def get_current_joint_positions(self) -> npt.NDArray[np.float32]:
        return self.joint_positions[0, :, :].copy()

    def get_current_joint_position_from_name(self, joint_name: str) -> npt.NDArray[np.float32]:
        try:
            joint_idx = self.get_joint_names().index(joint_name)
        except ValueError as e:
            logging.critical(e)
            assert False

        return self.get_current_joint_positions()[joint_idx, :]

    def get_root_position(self) -> npt.NDArray[np.float32]:
        return self.get_current_joint_position_from_name(self.motion_cfg.jointpositionarray_root_joint_name)

    def get_visualization_transform(self) -> Transform:
        return self.widget
