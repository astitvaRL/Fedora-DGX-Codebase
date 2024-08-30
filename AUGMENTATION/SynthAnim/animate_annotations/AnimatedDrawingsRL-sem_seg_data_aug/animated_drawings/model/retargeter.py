import logging
from abc import abstractclassmethod
from typing import Tuple, List, Dict, Optional
from animated_drawings.config import MotionConfig, RetargetConfig
from animated_drawings.model.motion_source import MotionSource, MotionSourceBVHFile
from animated_drawings.model.vectors import Vectors
from animated_drawings.model.animated_drawing_rig import AnimatedDrawingRig, AnimatedDrawing3DRig
from animated_drawings.model.joint import Joint
from animated_drawings.model.transform import Transform
from animated_drawings.model.view_cost_minimizer import ViewCostMinimizer
import numpy as np
import numpy.typing as npt
import math
from sklearn.decomposition import PCA

x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)


class Retargeter():
    """ Base retargeting class"""

    def __init__(self, motion_cfg: MotionConfig, retarget_cfg: RetargetConfig, character_rig: AnimatedDrawingRig):

        self.retarget_cfg = retarget_cfg
        self.motion_source = MotionSource.create_motion_source(motion_cfg)
        self.character_rig = character_rig

        self.root_offset_scale: Optional[float] = None  # amount by which to scale offset from motion source when applied to character.
        self._set_root_offset_scaling_factor()

        self.motion_source_fwd_vector: npt.NDArray[npt.float32] = self.motion_source.get_fwd_vector()

    def _set_root_offset_scaling_factor(self):
        """ Using the lists of joints with retarget_cfg.char_bvh_root_offset, compute the amount by which to scale root offset. """

        # compute average length of character legs
        c_limbs_length = 0
        c_joint_groups: List[List[str]] = self.retarget_cfg.char_bvh_root_offset['char_joints']
        for joint_group_names in c_joint_groups:
            c_limbs_length += self.character_rig.get_joint_chain_length(joint_group_names)

        c_average_limb_length = c_limbs_length / len(c_joint_groups)

        # compute average length of motion source legs
        b_limbs_length = 0
        b_joint_groups: List[List[str]] = self.retarget_cfg.char_bvh_root_offset['bvh_joints']
        for joint_group_names in b_joint_groups:
            b_limbs_length += self.motion_source.get_joint_chain_length(joint_group_names)
        b_average_limb_length = b_limbs_length / len(b_joint_groups)

        # if this is called with streaming motion source before a valid set of joint positions has been given, get_joint_chain_length will return -1.
        # In this case, do not adjust root_offset_scale.
        if b_average_limb_length < 0:
            return

        self.root_offset_scale = float(c_average_limb_length / b_average_limb_length)

    @classmethod
    @abstractclassmethod
    def set_motion_source(self, motion_source: MotionConfig, retarget_cfg: RetargetConfig):
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def retarget(self, time: float):
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def get_root_offset(self):
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def get_current_frame_source_joint_depths(self) -> Dict[str, npt.NDArray[np.float32]]:
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def get_destination_joint_render_order(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def get_retargeted_frame_data(self):
        pass

    @classmethod
    @abstractclassmethod
    def get_character_fwd_angle(self) -> float:
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def get_feet_orientation(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def set_viewer(self, viewer: Transform) -> None:
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def get_viewer_position(self) -> npt.NDArray[np.float32]:
        raise NotImplementedError

    @staticmethod
    def create_retargeter(motion_cfg: MotionConfig, retarget_cfg: RetargetConfig, character_rig: AnimatedDrawingRig):

        if retarget_cfg.retargeting_method.startswith('twisted_perspective_2D'):
            retargeter = RetargeterTwistedPerspective2D(motion_cfg, retarget_cfg, character_rig)
        else:
            raise AssertionError(f'bad retargeting_method: {retarget_cfg.retargeting_method}')

        """ Certain checks can only be run once we've set up the retargeter's motion source. Those go here. """
        bvh_joint_names = retargeter.bvh_joint_names
        character_joint_names = character_rig.root_joint.get_chain_joint_names()

        motion_cfg.validate_joint_names(bvh_joint_names)
        retarget_cfg.validate_char_and_bvh_joint_names(character_joint_names, bvh_joint_names)

        return retargeter


class RetargeterTwistedPerspective2D(Retargeter):
    """ A retargeter class """
    def __init__(self, motion_cfg: MotionConfig, retarget_cfg: RetargetConfig, character_rig: AnimatedDrawingRig):

        super().__init__(motion_cfg, retarget_cfg, character_rig)

        self.bvh_joint_names = self.motion_source.get_joint_names()
        self.motion_source_depth_driver_joint_names = [x for group in self.retarget_cfg.char_bodypart_groups for x in group['bvh_depth_drivers']]
        self.motion_source_orientation_driver_joint_names = list(set([name for joint_names in self.retarget_cfg.char_joint_bvh_joints_mapping.values() for name in joint_names]))
        self.motion_source_orientation_driver_joint_indices = [self.motion_source.get_joint_names().index(joint_name) for joint_name in self.motion_source_orientation_driver_joint_names]

        self.joint_group_name_to_projection_plane: Dict[str, npt.NDArray[np.float32]] = {}
        self.joint_to_projection_plane: Dict[str, npt.NDArray[np.float32]] = {}
        self._initialize_joint_to_projection_plane()

        # Tells us whether to translate character root based upon bvh skeleton's forward or lateral motion
        self.root_offset_projection_plane: npt.NDArray[np.float32]
        self.set_root_offset_projection_plane()

        self.last_motion_source_root_location: Optional[npt.NDArray[np.float32]] = None

        self.leftfoot_orientation = 'footright'  # direction characterleft foot is facing
        self.rightfoot_orientation = 'footleft'  # direciton characterright foot is face

        self._viewer: Optional[Transform] = None  # transform with position of the viewer. Needed for view-dependent retargeting

        self.view_theta: Optional[float] = None  # the angle, in degrees, between the character's fwd vector and the view vector, in the xz plane

        self.per_limb_last_projection_vector: Dict[str, npt.NDArray[np.float32]] = {}  # keep track of the last projection plane used to represent character

    def _initialize_joint_to_projection_plane(self):
        """ For every joint_projection group, calculate it's projection plane. """

        for joint_projection_group in self.retarget_cfg.bvh_projection_bodypart_groups:

            group_name = joint_projection_group['name']
            plane: npt.NDArray[np.float32] = self._determine_joint_projection_group_plane(joint_projection_group)

            self.joint_group_name_to_projection_plane[group_name] = plane

            for joint_name in joint_projection_group['bvh_joint_names']:
                self.joint_to_projection_plane[joint_name] = plane

    def _determine_joint_projection_group_plane(self, joint_projection_group: RetargetConfig.BvhProjectionBodypartGroup):
        """
        Given joint_projection_group, determines the plane to project onto and returns its normal.
        By convention, the projection plane contains the origin.
        """

        projection_method = joint_projection_group['method']
        group_name = joint_projection_group['name']
        joint_names = joint_projection_group['bvh_joint_names']

        if projection_method == 'frontal':
            logging.info(f'{group_name} projection_method is {projection_method}. Using {x_axis}')
            return x_axis
        elif projection_method == 'saggital':
            logging.info(f'{group_name} projection_method is {projection_method}. Using {z_axis}')
            return z_axis
        elif projection_method == 'pca':
            logging.info(f'{group_name} projection_method is {projection_method}. Running PCA on {joint_names}')

            # get 3rd pc
            pc3 = self._get_third_principal_component_from_joint_names(joint_names)

            # compute cosine similarity between 3rd pc and x-axis, 3rd pc and z-axis
            x_cos_sim: float = np.dot(x_axis, pc3) / (np.linalg.norm(x_axis) * np.linalg.norm(pc3))
            z_cos_sim: float = np.dot(z_axis, pc3) / (np.linalg.norm(z_axis) * np.linalg.norm(pc3))

            # use closer of the two
            axis = x_axis if abs(x_cos_sim) > abs(z_cos_sim) else z_axis
            logging.info(f'PCA complete. {group_name} using {axis}')
            return axis

        else:
            msg = f'bad project method for {group_name}: {projection_method}'
            logging.critical(msg)
            raise NotImplementedError(msg)

    def _get_third_principal_component_from_joint_names(self, joint_names: List[str]) -> npt.NDArray[np.float32]:
        """ Given a set of joint names, get their world cartesian coordinates, run PCA, and return third PC"""

        # if the motion isn't a prerecorded file, we can't create the joint position cloud to run pca
        if not isinstance(self.motion_source, MotionSourceBVHFile):
            raise AssertionError('pca projection method can only be performed on MotionSourceBVHFile')

        # get the positions for all joints...
        joint_positions = self.motion_source.get_normalized_joint_positions()

        # ...and then retain only the ones we care about
        joints_mask = np.full(joint_positions.shape[1], False, dtype=np.bool8)
        for joint_name in joint_names:
            idx = self.bvh_joint_names.index(joint_name)
            joints_mask[3*idx:3*(idx+1)] = True
        joints_positions = joint_positions[:, joints_mask].reshape([-1, 3])

        # perform PCA
        pca = PCA()
        pca.fit(joints_positions)

        # return the 3rd PC
        return pca.components_[2]

    def _get_global_projection_plane_normal(self) -> npt.NDArray[np.float32]:
        character_root_pos = self.character_rig.root_joint.update_and_get_world_position()
        v = self.get_viewer_position() - character_root_pos
        v[1] = 0
        v = Vectors(v)
        v.norm()
        plane_normal = v.vs[0]
        return plane_normal

    def _get_projection_plane_normal_from_joint_name(self, bvh_joint_name: str) -> npt.NDArray[np.float32]:

        """ cmu names and joints """
        bvh_joint_name_to_limb_name = {
            'LeftElbow': 'LeftArm',
            'LeftWrist': 'LeftArm',
            'RightElbow': 'RightArm',
            'RightWrist': 'RightArm',
            'LeftKnee': 'LeftLeg',
            'LeftAnkle': 'LeftLeg',
            'RightKnee': 'RightLeg',
            'RightAnkle': 'RightLeg',
        }

        limb_name_to_joint_names = {
            'LeftArm': ['LeftShoulder', 'LeftElbow', 'LeftWrist'],
            'RightArm': ['RightShoulder', 'RightElbow', 'RightWrist'],
            'LeftLeg': ['LeftHip', 'LeftKnee', 'LeftAnkle'],
            'RightLeg': ['RightHip', 'RightKnee', 'RightAnkle'],
        }

        # """ blueman names and joints """
        # bvh_joint_name_to_limb_name = {
        #     'b_l_forearm': 'LeftArm',
        #     'b_l_wrist': 'LeftArm',
        #     'b_r_forearm': 'RightArm',
        #     'b_r_wrist': 'RightArm',
        #     'b_l_leg': 'LeftLeg',
        #     'b_l_talocrural': 'LeftLeg',
        #     'b_r_leg': 'RightLeg',
        #     'b_r_talocrural': 'RightLeg',
        # }

        # limb_name_to_joint_names = {
        #     'LeftArm': ['b_l_shoulder', 'b_l_forearm', 'b_l_wrist'],
        #     'RightArm': ['b_r_shoulder', 'b_r_forearm', 'b_r_wrist'],
        #     'LeftLeg': ['b_l_upleg', 'b_l_leg', 'b_l_talocrural'],
        #     'RightLeg': ['b_r_upleg', 'b_r_leg', 'b_r_talocrural'],
        # }

        if True:
            if bvh_joint_name in bvh_joint_name_to_limb_name.keys():
                limb_name = bvh_joint_name_to_limb_name[bvh_joint_name]
                joint_names = limb_name_to_joint_names[limb_name]
                bvhjoint_name2xyz = {}
                for joint_name in joint_names:
                    bvhjoint_name2xyz[joint_name] = self.motion_source.bvh.get_transform_by_name(joint_name).get_world_position()

                bone_vector1 = bvhjoint_name2xyz[joint_names[1]] - bvhjoint_name2xyz[joint_names[0]]
                bone_vector2 = bvhjoint_name2xyz[joint_names[2]] - bvhjoint_name2xyz[joint_names[1]]

                character_root_pos = self.character_rig.root_joint.update_and_get_world_position()
                view_vector = self.get_viewer_position() - character_root_pos
                view_vector[1] = 0

                view_cost_minimizer = ViewCostMinimizer()
                view_cost_minimizer.set_bone_vectors(bone_vector1, bone_vector2)
                view_cost_minimizer.set_global_view_vector(view_vector)

                try:
                    view_cost_minimizer.set_last_view_vector(self.per_limb_last_projection_vector[limb_name])
                except:
                    view_cost_minimizer.set_last_view_vector(view_vector)
                answer_xyz = view_cost_minimizer.minimize()
                v = Vectors(answer_xyz)
                v.norm()
                plane_normal = v.vs[0]

                self.per_limb_last_projection_vector[limb_name] = plane_normal

                return plane_normal


        """ View dependent projection """
        if self.retarget_cfg.retargeting_method == 'twisted_perspective_2D_viewdependent':
            plane_normal = self._get_global_projection_plane_normal()

        else:
            """ Old, non-view dependent method """
            try:
                projection_plane_type = self.joint_to_projection_plane[bvh_joint_name]
            except Exception:
                raise AssertionError(f' error finding projection plane for bvh_end_joint_name: {bvh_joint_name}')

            # frontal projection
            if np.array_equal(projection_plane_type, x_axis):
                plane_normal = self.motion_source_fwd_vector

            # sagittal projection
            elif np.array_equal(projection_plane_type, z_axis):
                R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])    # mat to rotate it 90 degress CCW around Y axis
                plane_normal = np.dot(R, self.motion_source_fwd_vector)  # compute new plane is sagittal

        return plane_normal

    # def get_current_frame_source_joint_depths(self) -> Dict[str, npt.NDArray[np.float32]]:
    #     """
    #     For each BVH joint within bvh_projection_mapping_groups, compute distance to projection plane.
    #     This distance used if the joint is a char_body_segmentation_groups depth_driver.
    #     """

    #     bvh_joint_to_projection_depth: Dict[str, npt.NDArray[np.float32]] = {}

    #     joint_positions = self.motion_source.get_current_joint_positions()  # world positions

    #     for joint_name in self.motion_source_depth_driver_joint_names:

    #         # self.get_distance_from_joint_to_torso_projection_along_limb_specific_projection_vector(joint_name)

    #         joint_idx = self.bvh_joint_names.index(joint_name)
    #         joint_x0z = joint_positions[joint_idx, :]
    #         joint_x0z[1] = 0

    #         projection_plane_normal = self._get_global_projection_plane_normal()
    #         joint_depth = np.dot(joint_x0z, projection_plane_normal)
    #         bvh_joint_to_projection_depth[joint_name] = joint_depth

    #     return bvh_joint_to_projection_depth

    def get_distance_from_joint_to_torso_projection_along_limb_specific_projection_vector(self, joint_name: str):

        # get the projection vector
        projection_vector = self._get_projection_plane_normal_from_joint_name(joint_name)

        # get the cartesian coordinates of the joint
        joint_location = self.motion_source.bvh.get_transform_by_name(joint_name).get_world_position()

        # zero out y component
        joint_location[1] = 0

        # get the cartesian coordinates of the torso depth driver
        try:
            torso_joint_location = self.motion_source.bvh.get_transform_by_name('Hips').get_world_position()
            # torso_joint_location = self.motion_source.bvh.get_transform_by_name('b_root').get_world_position()
        except Exception:
            assert False, 'bad hardcoded joint name in retargeter'

        # zero out y component
        torso_joint_location[1] = 0

        # project joint_location on projection_vector
        joint_projection = np.dot(joint_location, projection_vector)

        # project torso_location on projection_vector
        torso_projection = np.dot(torso_joint_location, projection_vector)

        # compute delta between these two:
        projection_delta = joint_projection - torso_projection

        return projection_delta, joint_name, projection_vector


    def get_destination_joint_render_order(self) -> List[str]:

        if False:
            view_theta = np.deg2rad(self.view_theta)
            if np.pi/4 < view_theta < 3*np.pi/4:  # character is facing towards the left, use a set render order
                return [
                    'left_hand',
                    'left_elbow',
                    'left_shoulder',
                    'left_foot',
                    'left_knee',
                    'left_hip',
                    'hip',
                    'torso',
                    'neck',
                    'right_hip',
                    'right_knee',
                    'right_foot',
                    'right_shoulder',
                    'right_elbow',
                    'right_hand'
                    ]
            elif 5*np.pi/4 < view_theta < 7*np.pi/4:  # character is facing towards the right, use a set render order
                return [
                    'right_hand',
                    'right_elbow',
                    'right_shoulder',
                    'right_foot',
                    'right_knee',
                    'right_hip',
                    'hip',
                    'torso',
                    'neck',
                    'left_hip',
                    'left_knee',
                    'left_foot',
                    'left_shoulder',
                    'left_elbow',
                    'left_hand'
                    ]
            # otherwise, use the input motion to figure it out

        delta_name_projection = []
        for joint_name in self.motion_source_depth_driver_joint_names:
            delta_name_projection.append(self.get_distance_from_joint_to_torso_projection_along_limb_specific_projection_vector(joint_name))

        bvh_joint_to_projection_depth = {}
        for delta, name, projection in delta_name_projection:
            bvh_joint_to_projection_depth[name] = delta

        # bvh_joint_to_projection_depth = self.get_current_frame_source_joint_depths()

        _bodypart_render_order: List[Tuple[int, np.float32]] = []
        render_order_names: List[str] = []

        # sort segmentation groups by decreasing depth_driver's distance to camera
        for idx, bodypart_group_dict in enumerate(self.retarget_cfg.char_bodypart_groups):
            bodypart_depth: np.float32 = np.mean([bvh_joint_to_projection_depth[joint_name] for joint_name in bodypart_group_dict['bvh_depth_drivers']])
            _bodypart_render_order.append((idx, bodypart_depth))
        _bodypart_render_order.sort(key=lambda x: float(x[1]))

        # Add vertices belonging to joints in each segment group in the order they will be rendered
        for idx, dist in _bodypart_render_order:
            intra_bodypart_render_order = 1 if dist > 0 else -1  # if depth driver is behind plane, render bodyparts in reverse order
            for joint_name in self.retarget_cfg.char_bodypart_groups[idx]['char_joints'][::intra_bodypart_render_order]:
                render_order_names.append(self._adjust_left_right_bodypart_mapping_from_viewing_angle(joint_name))

        return render_order_names

    def _adjust_left_right_bodypart_mapping_from_viewing_angle(self, bodypart_name: str):
        """ Depending upon the angle between viewing vector and character fwd vector, left/right limbs get flipped"""
        view_theta = np.deg2rad(self.view_theta)
        if view_theta < np.pi/2 or 3*np.pi/2 < view_theta:
            if bodypart_name.startswith('left'):
                return bodypart_name.replace('left', 'right')
            elif bodypart_name.startswith('right'):
                return bodypart_name.replace('right', 'left')
            else:
                pass

        return bodypart_name

    def set_root_offset_projection_plane(self) -> None:
        try:
            projection_bodypart_group_for_root_offset = self.retarget_cfg.char_bvh_root_offset['bvh_projection_bodypart_group_for_offset']
            self.root_offset_projection_plane = self.joint_group_name_to_projection_plane[projection_bodypart_group_for_root_offset]
        except Exception as e:
            raise AssertionError(f'Error getting projection plane: {str(e)}')

    def set_root_offset_scale(self, root_offset_scale: float):
        self.root_offset_scale = root_offset_scale

    def compute_bone_vector(self, bvh_prox_joint_name: str, bvh_dist_joint_name: str):
        """ Given the name of a bvh proximal joint and distal joint, computes the 3D vector between the two"""
        # TODO: Delete this. No longer used

        # get distal end joint
        dist_joint = self.motion_source.bvh.root_joint.get_transform_by_name(bvh_dist_joint_name)
        if dist_joint is None or not isinstance(dist_joint, Joint) or dist_joint.name is None:
            raise AssertionError(f'error finding joint {bvh_dist_joint_name}')

        # get prox joint
        prox_joint = self.motion_source.bvh.root_joint.get_transform_by_name(bvh_prox_joint_name)
        if prox_joint is None or not isinstance(prox_joint, Joint) or prox_joint.name is None:
            raise AssertionError(f'error finding joint {bvh_prox_joint_name}')

        # get joint xyz locations
        dist_joint_idx = self.bvh_joint_names.index(dist_joint.name)
        dist_joint_xyz = self.motion_source.get_current_joint_positions()[3*dist_joint_idx:3*(dist_joint_idx+1)]

        prox_joint_idx = self.bvh_joint_names.index(prox_joint.name)
        prox_joint_xyz = self.motion_source.get_current_joint_positions()[3*prox_joint_idx:3*(prox_joint_idx+1)]

        # return the vector
        return dist_joint_xyz - prox_joint_xyz

    def get_current_frame_orientations(self) -> Dict[str, float]:
        """ Using the current frame of motion_source, compute the 1D orientations needed for each character joint.  """

        ms_joint_xyz = self.motion_source.get_current_joint_positions()[self.motion_source_orientation_driver_joint_indices, :]

        orientations: Dict[str, float] = {}
        for char_joint_name, (bvh_prox_joint_name, bvh_dist_joint_name) in self.retarget_cfg.char_joint_bvh_joints_mapping.items():
            dist_joint_xyz = ms_joint_xyz[self.motion_source_orientation_driver_joint_names.index(bvh_dist_joint_name)]
            prox_joint_xyz = ms_joint_xyz[self.motion_source_orientation_driver_joint_names.index(bvh_prox_joint_name)]

            plane_normal = self._get_projection_plane_normal_from_joint_name(bvh_dist_joint_name)
            # plane_normal = self._get_global_projection_plane_normal()  # if you want to remove the twisted perspective retargeting, just replace above line with this one

            angle = math.atan2(plane_normal[2], plane_normal[0]) - np.pi/2
            rot_mat = np.array([[np.cos(angle), 0, np.sin(angle)],
                                [0, 1, 0],
                                [-np.sin(angle), 0, np.cos(angle)]])

            bone_vector = dist_joint_xyz - prox_joint_xyz
            x, y, _ = rot_mat @ bone_vector.T

            orientations[char_joint_name] = np.arctan2(x, y)

        return {self._adjust_left_right_bodypart_mapping_from_viewing_angle(key): val for key, val in orientations.items()}

    def get_root_offset(self) -> npt.NDArray[np.float32]:
        """ Upon first call, returns 0, 0, 0 and caches the current position of the motion_source root.
        On every additional call, gets the current position of motion_source root, calculates the delta from
        when self was last called, updates the saved current position, and returns the scaled delta.
        """
        # TODO: This whole method for computing root offset feel brittle and hacky. Figure out something better.

        # first time this is called
        if self.last_motion_source_root_location is None:
            self.last_motion_source_root_location = self.motion_source.get_root_position()
            return np.array([0.0, 0.0, 0.0])

        # get current root location
        current_motion_source_root_location = self.motion_source.get_root_position()

        # compute offset
        root_offset = current_motion_source_root_location - self.last_motion_source_root_location

        # update the saved root location for next call
        self.last_motion_source_root_location = current_motion_source_root_location.copy()

        if not self.root_offset_scale:
            self._set_root_offset_scaling_factor()
        assert self.root_offset_scale is not None, 'root_offset_scale is None'
        # return the scaled offset
        return root_offset * self.root_offset_scale

    def get_character_fwd_angle(self) -> float:
        """ returns angle, in rads, between +x axis and character's fwd vector (in x-z plane)"""

        """ point directly at camera """
        if self.retarget_cfg.retargeting_method == 'twisted_perspective_2D_viewdependent':
            character_pos = self.character_rig.get_parent().update_and_get_world_position()
            v = self.get_viewer_position() - character_pos
            theta = np.arctan2(v[0], v[2])
            return theta

        """ match character's front vector with motion source direction """
        ms_fwd = self.motion_source.get_fwd_vector()  # motion source fwd
        theta = np.arctan2(ms_fwd[0], ms_fwd[2])  # frontal projection
        if np.array_equal(self.root_offset_projection_plane, z_axis):  # change to sagittal projection, if needed
            theta -= np.pi / 2

        # match angle completely
        if self.retarget_cfg.retargeting_method == 'twisted_perspective_2D':
            return theta

        # restrict to certain possible values
        if self.retarget_cfg.retargeting_method == 'twisted_perspective_2D_clamped':
            possible_vals = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
            theta_d = np.degrees(theta) % 360
            clamped_theta_idx = np.argmin(abs(possible_vals - theta_d))
            return np.radians(possible_vals[clamped_theta_idx])

        assert False, "bad retarget_cfg.retargeting_method specified"

    def update_view_angle(self) -> None:
        """
        Uses the global fwd vector of the motion source skeleton, along with the vector from the view camera,
        to character to determine what the character's fwd vector is, relative to the viewer, in the xz plane.
        0: character is facing away from viewer
        90:character facing view-image-space left
        180: character facing viewer
        270: character is facing view-image-space right.
        """
        view_vec = (self.character_rig.get_parent().update_and_get_world_position() - self.get_viewer_position())[[0, 2]]  # viewing vector from cam to character, xz
        fwd_vec = self.motion_source_fwd_vector[[0, 2]]  # motion source fwd, xz

        view_vec_n = view_vec / np.linalg.norm(view_vec)
        fwd_vec_n = fwd_vec / np.linalg.norm(fwd_vec)

        _y =  fwd_vec_n[1] * -view_vec_n[0] - fwd_vec_n[0] * -view_vec_n[1]
        _x = -fwd_vec_n[0] * -view_vec_n[0] + fwd_vec_n[1] * view_vec_n[1]
        theta = np.arctan2(_y, _x)
        self.view_theta = np.degrees(theta) % 360

    def get_side_facing_viewer(self) -> str:
        """ Compute the character's 'side' that is facing the viewer, using the orientation of the motion source and vector from camera to character """
        view_theta = np.deg2rad(self.view_theta)
        if np.pi/2 < view_theta < 3*np.pi/2:
            return 'front'
        else:
            return 'back'
        # depending upon angle, return what txtr we want to see
        # TODO: Add these magic numbers and magic strings a config file somewhere
        # if 315.0 <= self.view_theta or self.view_theta < 45.0:
        #     return 'back'
        # elif 45.0 <= self.view_theta < 135.0:
        #     return 'left'
        # elif 135.0 <= self.view_theta < 225.0:
        #     return 'front'
        # elif 225.0 <= self.view_theta < 315.0:
        #     return 'right'

    def set_time(self, time: float) -> None:
        """ Takes in time, converts to a valid frame number for motion source. Reposes skeleton and recomputes it's fwd vector for new frame. """
        frame_idx = int(round(time / self.motion_source.get_frame_time(), 0))
        frame_idx %= self.motion_source.get_max_frame_count()

        self.motion_source.apply_frame(frame_idx)

        self.motion_source_fwd_vector = self.motion_source.get_fwd_vector()

    def set_viewer(self, viewer: Transform) -> None:
        self._viewer = viewer

    def get_viewer_position(self) -> npt.NDArray[np.float32]:
        if self._viewer is None:
            raise ValueError('_viewer_pos is not set')

        return self._viewer.update_and_get_world_position()

    def get_character_bone_orientations(self) -> None:

        # depending upon type of rig, update limb orientations
        if isinstance(self.character_rig, AnimatedDrawing3DRig):
            raise AssertionError('not implemented')
        elif isinstance(self.character_rig, AnimatedDrawingRig):
            return self.get_current_frame_orientations()
        else:
            raise ValueError

    def get_feet_orientation(self) -> str:
        """ check whether lines from ankle to knee to hip joints are CW or CCW to determine foot orientation """
        joint_positions = self.character_rig.get_joint_positions()
        joint_names = self.character_rig.root_joint.get_chain_joint_names()
        # right_knee_plus_adjacent_joints = ['right_foot', 'right_knee', 'right_hip']

        """ right leg """
        v1 = joint_positions[joint_names.index('right_foot')]
        v2 = joint_positions[joint_names.index('right_knee')]
        v3 = joint_positions[joint_names.index('right_hip')]

        # only adjust foot orientation if bend is greater than certain amount. Don't want it flipping wildly when leg locked out
        vec1 = (joint_positions[joint_names.index('right_knee')] - joint_positions[joint_names.index('right_foot')])[:2]
        vec2 = (joint_positions[joint_names.index('right_hip')] - joint_positions[joint_names.index('right_knee')] )[:2]
        angle = np.degrees(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
        if angle < 5.0:
            pass
        else:
            right_det = np.linalg.det(np.array([
                [1.0, v1[0], v1[1]],
                [1.0, v2[0], v2[1]],
                [1.0, v3[0], v3[1]]
            ]))
            if right_det < 0:
                self.rightfoot_orientation = 'footleft'
            elif right_det > 0:
                self.rightfoot_orientation = 'footright'
            else:
                # TODO: cache last result and just don't change here
                self.rightfoot_orientation = 'footright'

        """ left leg """
        v1 = joint_positions[joint_names.index('left_foot')]
        v2 = joint_positions[joint_names.index('left_knee')]
        v3 = joint_positions[joint_names.index('left_hip')]

        # only adjust foot orientation if bend is greater than certain amount. Don't want it flipping wildly when leg locked out
        vec1 = (joint_positions[joint_names.index('left_knee')] - joint_positions[joint_names.index('left_foot')])[:2]
        vec2 = (joint_positions[joint_names.index('left_hip')] - joint_positions[joint_names.index('left_knee')] )[:2]
        angle = np.degrees(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
        if angle < 5.0:
            pass
        else:
            left_det = np.linalg.det(np.array([
                [1.0, v1[0], v1[1]],
                [1.0, v2[0], v2[1]],
                [1.0, v3[0], v3[1]]
            ]))
            if left_det < 0:
                self.leftfoot_orientation = 'footleft'
            elif left_det > 0:
                self.leftfoot_orientation = 'footright'
            else:
                # TODO: cache last result and just don't change here
                pass

        return f"{self.leftfoot_orientation}-{self.rightfoot_orientation}"

# class Retargeter_StreamingBVH(Retargeter):
#     """
#     Retargeter_StreamingBVH listens on a socket for a stream of poses.
#     Instead of using a time value to choose and retarget a pose from a prerecorded BVH file and return the proper joint orientation,
#     it insteads gets a skeletal pose from the server, retargets it, and computes and returns the orientations.
#     This is a subclass of the original Retargeter class, and it must be initialized with a BVH file whose skeleton matches that being streamed from the server.
#
#     TODO: Refactor Retarget class to abstract out the source of the motion in a different way. Possibly create BVH_Clip and BVH_Streaming classes that are both accessed in
#     the same manner by a single retargeter class.
#     """
#
#     def __init__(self, motion_cfg: MotionConfig, retarget_cfg: RetargetConfig):
#         self.cfg = retarget_cfg
#         super().__init__(motion_cfg, retarget_cfg)
#         self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.client.connect((motion_cfg.stream_ipv4, motion_cfg.stream_port))
#
#     def get_retargeted_frame_data(self, time: float) -> Tuple[Dict[str, float], Dict[str, float], npt.NDArray[np.float32]]:
#         """
#         Gets the most up-to-date frame from the server, uses it to reposition its BVH skeleton, projects joints to 2D, computes bone orientations.
#         Float parameter time is passed in but not used.
#         Currently doesn't compute joint depths or root offsets.
#         """
#
#         # get current frame from server
#         self.client.send("Requesting current frame\n".encode())
#         from_server = self.client.recv(4096)
#         data = json.loads(from_server.decode())
#
#         # reposition bvh skeleton
#         self.bvh.apply_root_pos(np.array(data['pos_data']))
#         self.bvh.apply_rotations(np.array(data['rot_data']))
#
#         # get bvh joint positions and skeleton's forawrd vector
#         joint_positions = np.array(self.bvh.root_joint.get_chain_worldspace_positions())
#         fwd_vector = self.bvh.get_skeleton_fwd(self.forward_perp_vector_joint_names).vs[0]
#
#         # reposition skeleton so root is over origin
#         bvh_root_position = joint_positions[:3]
#         joint_positions = np.subtract(joint_positions, np.tile(bvh_root_position, [len(self.bvh_joint_names)]))
#
#         # compute angle between skeleton's forward vector and x axis...
#         v1 = np.array([1.0, 0.0], dtype=np.float32)  # x axis
#         v2 = fwd_vector
#         dot: npt.NDArray[np.float32] = v1[0]*v2[0] + v1[1]*v2[2]
#         det: npt.NDArray[np.float32] = v1[0]*v2[2] - v2[0]*v1[1]
#         angle: npt.NDArray[np.float32] = np.arctan2(det, dot).astype(np.float32)
#         angle %= 2*np.pi
#         angle = np.where(angle < 0.0, angle + 2*np.pi, angle)
#
#         # and use to rotate so skeleton faces +X axis
#         rot_mat = np.identity(3).astype(np.float32)
#         rot_mat[0, 0] = math.cos(angle)
#         rot_mat[0, 2] = math.sin(angle)
#         rot_mat[2, 0] = -math.sin(angle)
#         rot_mat[2, 2] = math.cos(angle)
#         joint_positions = rot_mat @ joint_positions.reshape([-1, 3]).T
#
#         # for each bone of the character
#         char_name_to_orientation = {}
#         for char_joint_name, (bvh_prox_joint_name, bvh_dist_joint_name) in self.cfg.char_joint_bvh_joints_mapping.items():
#
#             # compute the 3D bvh bone vector
#             dist_joint_idx = self.bvh_joint_names.index(bvh_dist_joint_name)
#             dist_joint_xyz = joint_positions[:, dist_joint_idx]
#             prox_joint_idx = self.bvh_joint_names.index(bvh_prox_joint_name)
#             prox_joint_xyz = joint_positions[:, prox_joint_idx]
#             bone_vector = dist_joint_xyz - prox_joint_xyz  # type: ignore
#
#             # get bvh distal joint's projection plane
#             try:
#                 projection_plane_normal = self.joint_to_projection_plane[bvh_dist_joint_name]
#             except Exception:
#                 msg = f' error finding projection plane for bvh_end_joint_name: {bvh_dist_joint_name}'
#                 logging.critical(msg)
#                 assert False, msg
#
#             # project 3D bvh bone onto 2D plane
#             if np.array_equal(projection_plane_normal, x_axis):
#                 projected_bone_xy = np.stack((-bone_vector[ 2], bone_vector[ 1]))
#             elif np.array_equal(projection_plane_normal, z_axis):
#                 projected_bone_xy = np.stack((bone_vector[0], bone_vector[1]))
#             else:
#                 msg = 'error projection_plane_normal'
#                 logging.critical(msg)
#                 assert False, msg
#
#             # get the angle between the 2D y axis and the 2D bone projection
#             projected_bone_xy /= np.expand_dims(np.linalg.norm(projected_bone_xy), axis=-1)  # norm vector
#             y_axis = np.array([0.0, 1.0])
#             at1 = np.arctan2(projected_bone_xy[1], projected_bone_xy[0], dtype=np.float32)
#             at2 = np.arctan2(y_axis[1], y_axis[0], dtype=np.float32)
#             theta: npt.NDArray[np.float32]  = at1 - at2  # type: ignore
#             theta = np.degrees(theta) % 360.0
#             theta = np.where(theta < 0.0, theta + 360, theta)
#
#             # save it so it can be returned
#             char_name_to_orientation[char_joint_name] = theta
#
#         # return dummy joint depth values for now
#         char_name_to_joint_depths = {}
#         for bvh_joint_name in self.bvh_joint_names:
#             char_name_to_joint_depths[bvh_joint_name] = 0.0
#
#         # return dummy root positions for now
#         root_position = bvh_root_position
#
#         return char_name_to_orientation, char_name_to_joint_depths, root_position
