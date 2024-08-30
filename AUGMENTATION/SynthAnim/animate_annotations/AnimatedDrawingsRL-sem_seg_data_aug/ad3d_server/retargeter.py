from ad3d_server.config import RetargetConfig
from ad3d_server.vectors import Vectors
from ad3d_server.transform import Transform
from ad3d_server.animated_drawing_rig import AnimatedDrawingRig
from ad3d_server.view_cost_minimizer import ViewCostMinimizer

from typing import Tuple, List, Dict, Optional
import numpy as np
import numpy.typing as npt
import math
import logging


x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)


class RetargeterTwistedPerspective2D():
    """ A retargeter class """
    def __init__(self, character_rig: AnimatedDrawingRig):

        self.character_rig = character_rig

        # these will all be set when a motion source is registered with this retargeter
        self.motion_source_depth_driver_joint_names = None
        self.root_offset_scaling_char_joint_chains = None
        self.root_offset_scaling_ms_joint_chains = None
        self.char_bodypart_groups = None
        self.char_joint_to_ms_joints_orientation_mapping = None

        self.joint_group_name_to_projection_plane: Dict[str, npt.NDArray[np.float32]] = {}
        self.joint_to_projection_plane: Dict[str, npt.NDArray[np.float32]] = {}

        self.last_motion_source_root_location: Optional[npt.NDArray[np.float32]] = None

        self.leftfoot_orientation = 'footright'  # direction characterleft foot is facing
        self.rightfoot_orientation = 'footleft'  # direciton characterright foot is face

        self._viewer: Optional[Transform] = None  # transform with position of the viewer. Needed for view-dependent retargeting

        self.per_limb_last_projection_vector: Dict[str, npt.NDArray[np.float32]] = {}  # keep track of the last projection plane used to represent character
        self.per_limb_viewcostminimzer: Dict[str, ViewCostMinimizer] = {}
        self.twisted_perspective_limbnames_to_jointnames: Dict[str, List[str]] = {
            'RightArm': ['RightShoulder', 'RightElbow', 'RightWrist'],
            'LeftArm': ['LeftShoulder', 'LeftElbow', 'LeftWrist'],
            'RightLeg': ['RightHip', 'RightKnee', 'RightAnkle'],
            'LeftLeg': ['LeftHip', 'LeftKnee', 'LeftAnkle'],
        }

    def register_motion_source(self, motion_source_retarget_json):
        self._modify_retargeting_cfg_for_character(motion_source_retarget_json)
        self.motion_source_depth_driver_joint_names = motion_source_retarget_json['ms_depth_driver_to_char_joints'].keys()
        self.root_offset_scaling_char_joint_chains = motion_source_retarget_json['char_ms_root_offset_scaling_joint_chains']['char_joint_chains']
        self.root_offset_scaling_ms_joint_chains = motion_source_retarget_json['char_ms_root_offset_scaling_joint_chains']['ms_joint_chains']
        self.char_bodypart_groups = motion_source_retarget_json['ms_depth_driver_to_char_joints']
        self.char_joint_to_ms_joints_orientation_mapping = motion_source_retarget_json['char_joint_to_ms_joints_orientation_mapping']

    def _modify_retargeting_cfg_for_character(self, motion_source_retarget_json):
        """
        If the character is drawn in particular poses, the orientation-matching retargeting framework produce poor results.
        Therefore, the retargeter config can specify a number of runtime checks and retargeting modifications to make if those checks fail.
        """
        for position_test, target_joint_name, joint1_name, joint2_name in motion_source_retarget_json['char_runtime_checks']:
            if position_test == 'above':
                """ Checks whether target_joint is 'above' the vector from joint1 to joint2. If it's below, removes it.
                This was added to account for head flipping when nose was below shoulders. """

                # get joints 1, 2 and target joint
                joint1 = self.character_rig.root_joint.get_transform_by_name(joint1_name)
                if joint1 is None:
                    msg = f'Could not find joint1 in runtime check: {joint1_name}'
                    logging.critical(msg)
                    assert False, msg
                joint2 = self.character_rig.root_joint.get_transform_by_name(joint2_name)
                if joint2 is None:
                    msg = f'Could not find joint2 in runtime check: {joint2_name}'
                    logging.critical(msg)
                    assert False, msg
                target_joint = self.character_rig.root_joint.get_transform_by_name(target_joint_name)
                if target_joint is None:
                    msg = f'Could not find target_joint in runtime check: {target_joint_name}'
                    logging.critical(msg)
                    assert False, msg

                # get world positions
                joint1_xyz = joint1.update_and_get_world_position()
                joint2_xyz = joint2.update_and_get_world_position()
                target_joint_xyz = target_joint.update_and_get_world_position()

                # rotate target vector by inverse of test_vector angle. If then below x axis discard it.
                test_vector = np.subtract(joint2_xyz, joint1_xyz)
                target_vector = np.subtract(target_joint_xyz, joint1_xyz)
                angle = math.atan2(test_vector[1], test_vector[0])
                if (math.sin(-angle) * target_vector[0] + math.cos(-angle) * target_vector[1]) < 0:
                    logging.info(f'char_runtime_check failed, removing {target_joint_name} from retargeter :{target_joint_name, position_test, joint1_name, joint2_name}')
                    del retarget_json['char_joint_to_ms_joints_orientation_mapping'][target_joint_name]
            else:
                msg = f'Unrecognized char_runtime_checks position_test: {position_test}'
                logging.critical(msg)
                assert False, msg

    def set_viewer(self, viewer: Transform) -> None:
        self._viewer = viewer

    def get_root_offset_scaling_factor(self, ms_joint_names, ms_joint_positions):
        """ Using the lists of joints with retarget_cfg.char_bvh_root_offset, compute the amount by which to scale root offset. """

        if self.root_offset_scaling_char_joint_chains is None or self.root_offset_scaling_ms_joint_chains is None:
            assert False, 'Attempted to get root scaling factor before a motion source was registered'

        # compute average length of character legs
        c_limbs_length = 0
        c_joint_groups: List[List[str]] = self.root_offset_scaling_char_joint_chains
        for joint_group_names in c_joint_groups:
            c_limbs_length += self.character_rig.get_joint_chain_length(joint_group_names)
        c_average_limb_length = c_limbs_length / len(c_joint_groups)

        # compute average length of motion source legs
        b_limbs_length = 0
        b_joint_groups: List[List[str]] = self.root_offset_scaling_ms_joint_chains
        for joint_group_names in b_joint_groups:
            b_limbs_length += self.get_ms_joint_chain_length(joint_group_names, ms_joint_names, ms_joint_positions)
        b_average_limb_length = b_limbs_length / len(b_joint_groups)

        # if this is called with streaming motion source before a valid set of joint positions has been given, get_joint_chain_length will return -1.
        # In this case, do not adjust root_offset_scale.
        if b_average_limb_length < 0:
            return

        root_offset_scale = float(c_average_limb_length / b_average_limb_length)
        return root_offset_scale

    def get_ms_joint_chain_length(self, joint_names_: List[str], ms_joint_names: List[str], ms_joint_positions: npt.NDArray) -> float:

        joint_names = joint_names_.copy()

        if len(joint_names) < 2:
            raise ValueError('attempted to compute length of joint chain with less than two joints.')

        current_joint_name = joint_names.pop(0)
        next_joint_name = joint_names.pop(0)

        total_length = 0.0
        while len(joint_names):

            c_pos = ms_joint_positions[ms_joint_names.index(current_joint_name), :]
            n_pos = ms_joint_positions[ms_joint_names.index(next_joint_name), :]

            total_length += np.linalg.norm(np.subtract(n_pos, c_pos))
            current_joint_name = next_joint_name
            next_joint_name = joint_names.pop(0)

        return total_length

    def _get_global_projection_plane_normal(self) -> npt.NDArray[np.float32]:
        character_root_pos = self.character_rig.root_joint.update_and_get_world_position()
        v = self._viewer.update_and_get_world_position() - character_root_pos
        v[1] = 0
        v = Vectors(v)
        v.norm()
        plane_normal = v.vs[0]
        return plane_normal

    def update_per_limb_projection_plane_normals(self, ms_joint_names, ms_joint_positions):

        for limb_name, joint_names in self.twisted_perspective_limbnames_to_jointnames.items():

            joint0_xyz = ms_joint_positions[ms_joint_names.index(joint_names[0]), :]
            joint1_xyz = ms_joint_positions[ms_joint_names.index(joint_names[1]), :]
            joint2_xyz = ms_joint_positions[ms_joint_names.index(joint_names[2]), :]

            bone_vector1 = joint1_xyz - joint0_xyz
            bone_vector2 = joint2_xyz - joint1_xyz

            character_root_pos = self.character_rig.root_joint.update_and_get_world_position()
            view_vector = self._viewer.update_and_get_world_position() - character_root_pos
            view_vector[1] = 0

            if limb_name not in self.per_limb_viewcostminimzer.keys():
                self.per_limb_viewcostminimzer[limb_name] = ViewCostMinimizer()
            view_cost_minimizer = self.per_limb_viewcostminimzer[limb_name]

            view_cost_minimizer.set_bone_vectors(bone_vector1, bone_vector2)
            view_cost_minimizer.set_global_view_vector(view_vector)
            view_cost_minimizer.minimize(limb_name)

    def _get_projection_plane_normal_from_joint_name(self, bvh_joint_name: str) -> npt.NDArray[np.float32]:

        # Uncomment to disable twisted perspective retargeting
        # return self._get_global_projection_plane_normal()

        for limb_name, joint_names in self.twisted_perspective_limbnames_to_jointnames.items():
            if bvh_joint_name in joint_names:
                answer_xyz = self.per_limb_viewcostminimzer[limb_name].get_optimal_xyz()
                v = Vectors(answer_xyz)
                v.norm()
                plane_normal = v.vs[0]
                return plane_normal
        else:
            return self._get_global_projection_plane_normal()

    def get_distance_from_joint_to_torso_projection_along_limb_specific_projection_vector(self, joint_name: str, ms_joint_names, ms_joint_positions):

        # get the projection vector
        projection_vector = self._get_projection_plane_normal_from_joint_name(joint_name)

        # get the cartesian coordinates of the joint
        joint_location = ms_joint_positions[ms_joint_names.index(joint_name), :].copy()

        # zero out y component
        joint_location[1] = 0

        # get the cartesian coordinates of character point approximate
        try:
            character_point_approximation = np.mean(ms_joint_positions, axis=0)  # approximation of center torso joint as average of all other joints
        except Exception:
            assert False, 'bad hardcoded joint name in retargeter'

        # zero out y component
        character_point_approximation[1] = 0

        # project joint_location on projection_vector
        joint_projection = np.dot(joint_location, projection_vector)

        # project torso_location on projection_vector
        torso_projection = np.dot(character_point_approximation, projection_vector)

        # compute delta between these two:
        projection_delta = joint_projection - torso_projection

        return projection_delta, joint_name, projection_vector

    def get_destination_joint_render_order(self, ms_joint_names, ms_joint_positions, view_theta) -> List[str]:

        if self.root_offset_scaling_char_joint_chains is None or self.root_offset_scaling_ms_joint_chains is None or self.char_bodypart_groups is None:
            assert False, 'Attempted to get render order before a motion source was registered'

        delta_name_projection = []
        for joint_name in self.motion_source_depth_driver_joint_names:
            delta_name_projection.append(self.get_distance_from_joint_to_torso_projection_along_limb_specific_projection_vector(joint_name, ms_joint_names, ms_joint_positions))

        bvh_joint_to_projection_depth = {}
        for delta, name, projection in delta_name_projection:
            bvh_joint_to_projection_depth[name] = delta

        # bvh_joint_to_projection_depth = self.get_current_frame_source_joint_depths()

        _bodypart_render_order: List[Tuple[int, np.float32]] = []
        render_order_names: List[str] = []

        # sort segmentation groups by decreasing depth_driver's distance to camera
        for ms_joint, _ in self.char_bodypart_groups.items():
            bodypart_depth: np.float32 = bvh_joint_to_projection_depth[ms_joint]
            _bodypart_render_order.append((ms_joint, bodypart_depth))
        _bodypart_render_order.sort(key=lambda x: float(x[1]))

        # Add vertices belonging to joints in each segment group in the order they will be rendered
        for ms_joint, dist in _bodypart_render_order:
            intra_bodypart_render_order = 1 if dist > 0 else -1  # if depth driver is behind plane, render bodyparts in reverse order
            for joint_name in self.char_bodypart_groups[ms_joint][::intra_bodypart_render_order]:
                render_order_names.append(self._adjust_left_right_bodypart_mapping_from_viewing_angle(joint_name, view_theta))

        return render_order_names

    def _adjust_left_right_bodypart_mapping_from_viewing_angle(self, bodypart_name: str, view_theta: float):
        """ Depending upon the angle between viewing vector and character fwd vector, left/right limbs get flipped"""
        view_theta_rads = np.deg2rad(view_theta)
        if view_theta_rads < np.pi/2 or 3*np.pi/2 < view_theta_rads:
            if bodypart_name.startswith('left'):
                return bodypart_name.replace('left', 'right')
            elif bodypart_name.startswith('right'):
                return bodypart_name.replace('right', 'left')
            else:
                pass

        return bodypart_name

    def get_current_frame_orientations(self, ms_joint_names, ms_joint_positions, view_theta) -> Dict[str, float]:
        """ Using the current frame of motion_source, compute the 1D orientations needed for each character joint.  """
        if self.char_joint_to_ms_joints_orientation_mapping is None:
            assert False, 'Attempted to get_current_frame_orientations without registering motion source first'

        # first compute the per-frame projection planes for the character
        self.update_per_limb_projection_plane_normals(ms_joint_names, ms_joint_positions)

        # then use them to get the angles for each bone
        orientations: Dict[str, float] = {}
        for char_joint_name, (bvh_prox_joint_name, bvh_dist_joint_name) in self.char_joint_to_ms_joints_orientation_mapping.items():
            dist_joint_xyz = ms_joint_positions[ms_joint_names.index(bvh_dist_joint_name)]
            prox_joint_xyz = ms_joint_positions[ms_joint_names.index(bvh_prox_joint_name)]

            plane_normal = self._get_projection_plane_normal_from_joint_name(bvh_dist_joint_name)
            # plane_normal = self._get_global_projection_plane_normal()  # if you want to remove the twisted perspective retargeting, just replace above line with this one

            angle = math.atan2(plane_normal[2], plane_normal[0]) - np.pi/2
            rot_mat = np.array([[np.cos(angle), 0, np.sin(angle)],
                                [0, 1, 0],
                                [-np.sin(angle), 0, np.cos(angle)]])

            bone_vector = dist_joint_xyz - prox_joint_xyz
            x, y, _ = rot_mat @ bone_vector.T

            orientations[char_joint_name] = np.arctan2(x, y)

        return {self._adjust_left_right_bodypart_mapping_from_viewing_angle(key, view_theta): val for key, val in orientations.items()}

    def get_side_facing_viewer(self, view_theta) -> str:
        """ Compute the character's 'side' that is facing the viewer, using the orientation of the motion source and vector from camera to character """
        view_theta_rads = np.deg2rad(view_theta)
        if np.pi/2 < view_theta_rads < 3*np.pi/2:
            return 'front'
        else:
            return 'back'

    def get_character_bone_orientations(self, ms_joint_names, ms_joint_positions, view_theta) -> None:
        return self.get_current_frame_orientations(ms_joint_names, ms_joint_positions, view_theta)

    def get_feet_orientation(self) -> str:
        """ check whether lines from ankle to knee to hip joints are CW or CCW to determine foot orientation """
        joint_positions = self.character_rig.get_joint_positions()
        joint_names = self.character_rig.root_joint.get_chain_joint_names()

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
