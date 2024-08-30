from ad3d_server.vectors import Vectors
from ad3d_server.quaternions import Quaternions
import numpy as np


class ViewCostMinimizer():
    def __init__(self):
        self.bone_vector1 = None
        self.bone_vector2 = None
        self.global_view_vector = None

        self.last_view_vector = None

        self.current_optimal_view_vector = None

    def set_bone_vectors(self, bone_vector1_xyz, bone_vector2_xyz):

        bv1 = Vectors(bone_vector1_xyz)
        self.bone_vector1 = bv1
        self.bone_vector1.norm()

        bv2 = Vectors(bone_vector2_xyz)
        self.bone_vector2 = bv2
        self.bone_vector2.norm()

    def set_global_view_vector(self, global_viewing_vector_xyz):
        self.global_view_vector = Vectors(global_viewing_vector_xyz)
        self.global_view_vector.norm()

        if self.current_optimal_view_vector is None:
            self.current_optimal_view_vector = Vectors(global_viewing_vector_xyz)
            self.current_optimal_view_vector.norm()

    def minimize(self, limb_name):

        self.last_view_vector = Vectors(self.current_optimal_view_vector.vs.copy())

        # Using the two bone vectors for the limb, cross to get the pole of the great circle containing the limb bend
        pole = self.bone_vector1.cross(self.bone_vector2)
        pole.norm()

        # find quaternion that represents rotating the pole so it aligns with y axis
        pole_q = Quaternions.rotate_between_vectors(pole, Vectors([0, 1, 0]))

        # global view vector after rotation
        g_vec_prime = pole_q.to_rotation_matrix() @ np.array([*self.global_view_vector.vs[0], 0])
        g_vec_prime_x, g_vec_prime_y, g_vec_prime_z, _ = g_vec_prime

        # spherical coords of rotated global view vector
        g_vec_prime_phi = np.rad2deg(np.arctan2(g_vec_prime_x, g_vec_prime_z))
        g_vec_prime_theta = np.rad2deg(np.arccos(g_vec_prime_y))

        # last view vector after rotation
        l_vec_prime = pole_q.to_rotation_matrix() @ np.array([*self.last_view_vector.vs[0], 0])
        l_vec_prime_x, l_vec_prime_y, l_vec_prime_z, _ = l_vec_prime

        # spherical coords of rotated last view vector. only theta is used
        l_vec_prime_theta = np.rad2deg(np.arccos(l_vec_prime_y))

        """ 
        if the two bone limbs are parallel, then there isn't one unique great circle. 
        Numerical error can cause the computed one to vary wildly.
        To get around this, we weight the influence of great circle by the angle between the bones
        great_circle_weight = 1 - np.abs(np.dot(self.bone_vector1.vs[0], self.bone_vector2.vs[0]))
        """
        # try weighting great circle weight by distance to nearest bone vector
        great_circle_weight = max(
            np.abs(np.dot(self.bone_vector1.vs[0], self.global_view_vector.vs[0])),
            np.abs(np.dot(self.bone_vector2.vs[0], self.global_view_vector.vs[0]))
        ) * (1 - np.abs(np.dot(self.bone_vector1.vs[0], self.bone_vector2.vs[0])))

        # if limb is close to straight and has minimal bend, can just use the global view vector
        if great_circle_weight < 0.3:
            self.current_optimal_view_vector = Vectors(self.global_view_vector.vs[0].copy())
            self.current_optimal_view_vector.norm()
            return

        # with location of the global viewing vector theta, the great circle (theta = 90 degrees), and the last limb viewing vector theta,
        # we can find the optimizated view. Currently just doing a brute force search
        x = np.linspace(0, 180, 360)

        gv_cost = self.global_view_cost(x, g_vec_prime_theta)
        lv_cost = self.last_view_cost(x, l_vec_prime_theta)
        gc_cost = self.great_circle_cost(x, 90, great_circle_weight)
        y = gv_cost + lv_cost + gc_cost

        opt_view_theta = 180 / 360 * np.argmin(y)
        opt_view_phi = g_vec_prime_phi

        # convert from spherical to cartesian
        opt_x = np.sin(np.deg2rad(opt_view_theta)) * np.sin(np.deg2rad(opt_view_phi))
        opt_y = np.cos(np.deg2rad(opt_view_theta))
        opt_z = np.sin(np.deg2rad(opt_view_theta)) * np.cos(np.deg2rad(opt_view_phi))

        # convert from rotated vector to worldspace vector
        opt_view_vector_prime = np.array([opt_x, opt_y, opt_z, 0.0])
        opt_view_vector = pole_q.to_rotation_matrix().T @ opt_view_vector_prime

        self.current_optimal_view_vector = Vectors(opt_view_vector[:3])

    def get_optimal_xyz(self):
        return self.current_optimal_view_vector.vs[0]

    def global_view_cost(self, x, g_theta):
        spread = 15
        slope = 0.1
        weight = 7
        return weight * (
            (1 / (1 + np.exp(-slope * (x - (g_theta + spread))))) -
            (1 / (1 + np.exp(-slope * (x - (g_theta - spread)))))
        )

    def last_view_cost(self, x, last_theta):
        spread = 15
        weight = 5
        slope = 0.2
        return weight * (
            (1 / (1 + np.exp(-slope * (x - (last_theta + spread))))) - 
            (1 / (1 + np.exp(-slope * (x - (last_theta - spread)))))
        )

    def great_circle_cost(self, x, great_circle_theta, great_circle_weight):
        return great_circle_weight * 20 *  np.exp(-(x-great_circle_theta)**2 / 10)
