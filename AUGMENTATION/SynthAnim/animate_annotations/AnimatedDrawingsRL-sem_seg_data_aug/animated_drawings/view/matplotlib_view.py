# This view is primarily designed to be used with the window_view and interactive controller so you can view plots while also seeing the scene.
from __future__ import annotations
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from animated_drawings.model.retargeter import Retargeter
from animated_drawings.model.quaternions import Quaternions
from animated_drawings.model.vectors import Vectors


class MatplotlibView:

    def __init__(self, scene):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim([-1, 1])
        self.ax.set_xlabel("X")

        self.ax.set_ylim([-1, 1])
        self.ax.set_ylabel("Y")

        self.ax.set_zlim([-1, 1])
        self.ax.set_zlabel("Z")

        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # self.ax.set_zticks([])

        # self.ax.set_xlabel("")
        # self.ax.set_ylabel("")
        # self.ax.set_zlabel("")

        elev = 40
        azim = 180
        roll = -90
        self.ax.view_init(elev, azim, roll)

        self.retargeter: Retargeter = scene.get_children()[1].retargeter

        self.line_view_angle, = self.ax.plot([0, 0], [0, 0], [0, 2], 'r-')  # view is always along +Z axis

        self.bvh = self.retargeter.motion_source.bvh

        self.view_vector_line =  self.ax.plot([0, 0], [0, 0], [0, 0], 'g-')[0]





        # """ 3D skeleton """
        # self.lines_3d = []
        # def create_lines_3d(joint_name):
        #     for child_name in self.bvh.parent_children_dict[joint_name]:
        #         if child_name == 'End Site':
        #             continue
        #         self.lines_3d.append( self.ax.plot([0, 0], [0, 0], [0, 0], 'g-'),)
        #         create_lines_3d(child_name)
        # joint_name = self.bvh.root_joint.name
        # create_lines_3d(joint_name)

        # """ full body 2D skeleton projection """
        # self.lines_projected = []
        # def create_lines_projected(joint_name):
        #     for child_name in self.bvh.parent_children_dict[joint_name]:
        #         if child_name == 'End Site':
        #             continue
        #         self.lines_projected.append( self.ax.plot([0, 0], [0, 0], [0, 0], 'b-'),)
        #         create_lines_projected(child_name)
        # joint_name = self.bvh.root_joint.name
        # create_lines_projected(joint_name)

        # self.proj_z = -0.6
        # self.projection_plane_outlines = {}
        # self.add_projection_plane('fullbody')

        # self.add_projection_plane('leftleg')

        # self.projection_normals = {}
        # self.add_projection_normal('leftleg')

        # self.limb_bones = {}
        # self.add_limb_bone('leftleg_upper')
        # self.add_limb_bone('leftleg_lower')

        # if True:
        #     #self.left_leg_proj_normal, = self.ax.plot([0, 0], [0, 0], [0, 2], 'r-')  # view is always along +Z axis

        #     self.left_upper_leg, = self.ax.plot([0, 0], [0, 0], [0, 2], 'b-')  # view is always along +Z axis
        #     self.left_lower_leg, = self.ax.plot([0, 0], [0, 0], [0, 2], 'b-')  # view is always along +Z axis

    def _get_plane_viz_points(self, z: float) -> npt.NDArray[np.float32]:
        return np.array([
            [-0.5, -0.5, z, 1.0],
            [0.5, -0.5, z, 1.0],
            [-0.5, 0.5, z, 1.0],
            [-0.5, -0.5, z, 1.0],
            [0.5, 0.5, z, 1.0],
            [-0.5, 0.5, z, 1.0],
            [0.5, -0.5, z, 1.0],
            [0.5, 0.5, z, 1.0]
        ])

    def add_limb_bone(self, limb_name: str):
        self.limb_bones[limb_name], = self.ax.plot([0, 0], [0, 0], [0, 2], 'b-')

    def add_projection_normal(self, limb_name: str):
        self.projection_normals[limb_name], = self.ax.plot([0, 0], [0, 0], [0, 2], 'r-')

    def add_projection_plane(self, limb_name: str):
        """ side projection plane """
        pts = self._get_plane_viz_points(self.proj_z)

        rot_m = Quaternions.identity((1,)).to_rotation_matrix()
        pts = (rot_m @ pts.T).T

        projection_plane_lines = []
        for p1xyz1_p2xyz1 in pts.reshape(-1, 8):
            p1x, p1y, p1z, _, p2x, p2y, p2z, _ = p1xyz1_p2xyz1
            projection_plane_lines.append(
                self.ax.plot([p1x, p2x], [p1y, p2y], [p1z, p2z], 'black'),
            )
        self.projection_plane_outlines[limb_name] = projection_plane_lines

    def get_joint_xyz_when_pnormal_is_zaligned(self, bvh_prox_joint_name, bvh_dist_joint_name):
        ms_joint_xyz = self.retargeter.motion_source.get_current_joint_positions()[self.retargeter.motion_source_orientation_driver_joint_indices, :]
        prox_joint_xyz = ms_joint_xyz[self.retargeter.motion_source_orientation_driver_joint_names.index(bvh_prox_joint_name)]
        dist_joint_xyz = ms_joint_xyz[self.retargeter.motion_source_orientation_driver_joint_names.index(bvh_dist_joint_name)]

        # dist_joint_xyz = self.retargeter._get_joint_xyz_from_name(bvh_dist_joint_name)

        # prox_joint_xyz = self.retargeter._get_joint_xyz_from_name(bvh_prox_joint_name)

        bone_vector = dist_joint_xyz - prox_joint_xyz

        plane_normal: npt.NDArray[np.float32] = self.retargeter.get_projection_plane_normal()
        rot_m = self.retargeter.get_plane_normal_to_z_rot_m(plane_normal)

        return rot_m @ bone_vector.T

    def get_lines_3d_data(self, joint_name, line_data=[], px=0, py=0, pz=0):
        for child_name in self.bvh.parent_children_dict[joint_name]:
            if child_name == 'End Site':
                continue
            x, y, z = self.get_joint_xyz_when_pnormal_is_zaligned(joint_name, child_name)
            line_data.append((
                [px, px + x],
                [py, py + y],
                [pz, pz + z]
            ))
            self.get_lines_3d_data(child_name, line_data, px + x, py + y, pz + z)
        return line_data

    def get_lines_projected_data(self, joint_name, line_data=[], px=0.0, py=0.0, pz=0.0):
        for child_name in self.bvh.parent_children_dict[joint_name]:
            if child_name == 'End Site':
                continue
            x, y, z = self.get_joint_xyz_when_pnormal_is_zaligned(joint_name, child_name)

            z = 0  # projected onto XY plane

            line_data.append((
                [px, px + x],
                [py, py + y],
                [pz, pz + z]
            ))
            self.get_lines_projected_data(child_name, line_data, px + x, py + y, pz + z)
        return line_data

    def render(self) -> None:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
        """
        Okay. We're going to start with the bvh left arm and character image-space right arm. We want to know the vector angle of project and the vector angles
        of the bvh left upper arm and left lower arm.        
        """

        joint_name = self.bvh.root_joint.name

        view_vec = self.bvh.root_joint.get_world_position() - self.retargeter.get_viewer_position()
        view_vec[2] = 0
        view_vec /= np.linalg.norm(view_vec)
        
        self.view_vector_line.set_xdata([0, view_vec[0]])
        self.view_vector_line.set_ydata([0, view_vec[1]])
        self.view_vector_line.set_3d_properties([0, 0])

        # for line, in self.lines_3d:
        #     # lines_3d_data.pop(0)
        #     line.set_xdata([px, x])
        #     line.set_ydata([py, y])
        #     line.set_3d_properties([pz, z])

        # """ Update full body 3D skeleton """
        # lines_3d_data = self.get_lines_3d_data(joint_name)
        # for line, in self.lines_3d:
        #     (px, x), (py, y), (pz, z) = lines_3d_data.pop(0)
        #     line.set_xdata([px, x])
        #     line.set_ydata([py, y])
        #     line.set_3d_properties([pz, z])

        # """ Update full body projected 2D skeleton """
        # lines_projected_data = self.get_lines_projected_data(joint_name, pz=self.proj_z)
        # for line, in self.lines_projected:
        #     (px, x), (py, y), (pz, z) = lines_projected_data.pop(0)
        #     line.set_xdata([px, x])
        #     line.set_ydata([py, y])
        #     line.set_3d_properties([pz, z])

        # if True:
        #     """ Update projection normal for left leg """

        #     # """ start at the location of the knee. """
        #     x, y, z = self.get_joint_xyz_when_pnormal_is_zaligned(self.bvh.root_joint.name, 'LeftKnee')

        #     global_plane_normal = self.retargeter.get_projection_plane_normal()
        #     rot_m = self.retargeter.get_plane_normal_to_z_rot_m(global_plane_normal)  # world to z aligned

        #     left_leg_pn = self.retargeter.get_left_leg_projection_plane_normal(global_plane_normal, 'LeftHip', 'LeftKnee', 'LeftAnkle') # get vector for left leg projection, in world space
        #     dx, dy, dz = (rot_m @ left_leg_pn.T).T  # transform for world space into global_normal__z_aligned space

        #     self.projection_normals['leftleg'].set_xdata([x, x + dx])
        #     self.projection_normals['leftleg'].set_ydata([y, y + dy])
        #     self.projection_normals['leftleg'].set_3d_properties([z, z + dz])

        #     """ Update project plane for left leg """
        #     theta = np.arctan2(dx, dz)
        #     plane_rot_m = Quaternions.from_angle_axis(np.array([theta]), Vectors([0, 1, 0])).to_rotation_matrix()

        #     pts = self._get_plane_viz_points(self.proj_z)
        #     pts = (plane_rot_m @ pts.T).T

        #     for idx, p1xyz1_p2xyz1 in enumerate(pts.reshape(-1, 8)):
        #         p1x, p1y, p1z, _, p2x, p2y, p2z, _ = p1xyz1_p2xyz1
        #         line, = self.projection_plane_outlines['leftleg'][idx]
        #         line.set_xdata([p1x, p2x])
        #         line.set_ydata([p1y, p2y])
        #         line.set_3d_properties([p1z, p2z])



        #     """ show 2D projection of the leg """
        #     hx, hy, hz = self.get_joint_xyz_when_pnormal_is_zaligned(self.bvh.root_joint.name, 'LeftHip')
        #     kx, ky, kz = self.get_joint_xyz_when_pnormal_is_zaligned(self.bvh.root_joint.name, 'LeftKnee')
        #     ax, ay, az = self.get_joint_xyz_when_pnormal_is_zaligned(self.bvh.root_joint.name, 'LeftAnkle')

        #     # rotate by the angle that brings viewing angle to +z axis
        #     rot_m = self.retargeter.get_plane_normal_to_z_rot_m(np.array([dx, dy, dz]))

        #     hx, hy, hz = rot_m @ np.array([hx, hy, hz])
        #     hz = self.proj_z
        #     hx, hy, hz = rot_m.T @ np.array([hx, hy, hz])

        #     kx, ky, kz = rot_m @ np.array([kx, ky, kz])
        #     kz = self.proj_z
        #     kx, ky, kz = rot_m.T @ np.array([kx, ky, kz])

        #     ax, ay, az = rot_m @ np.array([ax, ay, az])
        #     az = self.proj_z
        #     ax, ay, az = rot_m.T @ np.array([ax, ay, az])

        #     self.limb_bones['leftleg_upper'].set_xdata([hx, kx])
        #     self.limb_bones['leftleg_upper'].set_ydata([hy, ky])
        #     self.limb_bones['leftleg_upper'].set_3d_properties([hz, kz])

        #     self.limb_bones['leftleg_lower'].set_xdata([kx, ax])
        #     self.limb_bones['leftleg_lower'].set_ydata([ky, ay])
        #     self.limb_bones['leftleg_lower'].set_3d_properties([kz, az])

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def clear_window(self) -> None:
        """ Clear output from previous render loop. """
        pass

    def cleanup(self) -> None:
        """ Cleanup after render loop is finished. """
        pass

    def get_framebuffer_size(self) -> Tuple[int, int]:
        """ Return (width, height) of framebuffer. """
        assert False, "This doesn't need to be called"
