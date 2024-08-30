# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import numpy.typing as npt
from collections import defaultdict
import logging
from typing import List, Dict, Set, Tuple
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import math
import time
import matplotlib.pyplot as plt
from abc import abstractclassmethod

import ad3d_server.animated_drawing_utils as ad_utils


csr_matrix = sp._csr.csr_matrix  # for typing  # pyright: ignore[reportPrivateUsage]


class ARAP():
    def __init__(self):
        pass

    @abstractclassmethod
    def solve(self, constrains: npt.NDArray[np.float32]):
        assert False, 'ARAP subclasses must implement solve'


class ARAP_Igarashi(ARAP):
    """
    Implementation of:

    Takeo Igarashi and Yuki Igarashi.
    "Implementing As-Rigid-As-Possible Shape Manipulation and Surface Flattening."
    Journal of Graphics, GPU, and Game Tools, A.K.Peters, Volume 14, Number 1, pp.17-30, ISSN:2151-237X, June, 2009.
    https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf

    General idea is this:
    Start with an an input mesh, comprised of vertices (v in V) and edges (e in E),
    and an initial set of pins (or control handle) locations.

    Then, given new positions for the pins, find new vertex locations (v' in V')
    such that the edges (e' in E') are as similar as possible, in a least squares sense, to the original edges (e in E).
    Translation and rotation aren't penalized, but edge scaling is.
    Not allowing rotation makes this tricky, as edges are directed vectors.

    Solution involves finding vertex locations twice. First, you do so while allowing both rotation and scaling to be free.
    Then you collect the per-edge rotation transforms found by this solution.
    During the second solve, you rotate the original edges (e in E) by the rotation matrix prior to computing the difference
    between (e' in E') and (e in E). This way, rotation is essentially free, while scaling is not.
    """

    def __init__(self, vertices: npt.NDArray[np.float32], triangles: npt.NDArray[np.int32], pins_xy: npt.NDArray[np.float32], w: int = 1000):  # noqa: C901
        """
        Sets up the matrices needed for later solves.

        pins_xy: ndarray [N, 2] specifying initial xy positions of N control points
        vertices: ndarray [N, 2] containing xy positions of N vertices. A vertex's order within array is it's vertex ID
        triangles: ndarray [N, 3] triplets of vertex IDs that make up triangles comprising the mesh
        w: int the weights to use for control points in solve. Default value should work.
        """
        self.w = w

        # ARAP_Igarashi.plot_mesh(vertices, triangles, pins_xy)

        self.vertices = np.copy(vertices)

        # build a deduplicated list of edge->vertex IDS...
        self.e_v_idxs: List[Tuple[np.int32, np.int32]] = []
        for v0, v1, v2 in triangles:
            self.e_v_idxs.append(tuple(sorted((v0, v1))))
            self.e_v_idxs.append(tuple(sorted((v1, v2))))
            self.e_v_idxs.append(tuple(sorted((v2, v0))))
        self.e_v_idxs = list(set(self.e_v_idxs))  # ...and deduplicate it

        # build list of edge vectors
        _edge_vectors: List[npt.NDArray[np.float32]] = []
        for vi_idx, vj_idx in self.e_v_idxs:
            vi = self.vertices[vi_idx]
            vj = self.vertices[vj_idx]
            _edge_vectors.append(vj - vi)
        self.edge_vectors: npt.NDArray[np.float32] = np.array(_edge_vectors)

        # get barycentric coordinates of pins, and mask denoting which pins were initially outside the mesh
        pins_bc: List[Tuple[Tuple[np.int32, np.float32], Tuple[np.int32, np.float32], Tuple[np.int32, np.float32]]]
        self.pin_mask = npt.NDArray[np.bool8]
        pins_bc, self.pin_mask = ad_utils.xy_to_barycentric_coords(pins_xy, vertices, triangles)

        v_vnbr_idxs: Dict[np.int32, Set[np.int32]] = defaultdict(set)  # build a dict mapping vertex ID -> neighbor vertex IDs
        for v0, v1, v2 in triangles:
            v_vnbr_idxs[v0] |= {v1, v2}
            v_vnbr_idxs[v1] |= {v2, v0}
            v_vnbr_idxs[v2] |= {v0, v1}

        self.edge_num = len(self.e_v_idxs)
        self.vert_num = len(self.vertices)
        self.pin_num = len(pins_xy[self.pin_mask])

        self.A1: npt.NDArray[np.float32] = np.zeros([2 * (self.edge_num + self.pin_num), 2 * self.vert_num], dtype=np.float32)
        G: npt.NDArray[np.float32] = np.zeros([2 * self.edge_num, 2 * self.vert_num], dtype=np.float32)  # holds edge rotation calculations

        # populate top half of A1, one row per edge
        for k, (vi_idx, vj_idx) in enumerate(self.e_v_idxs):

            # initialize self.A1 with 1, -1 denoting beginning and end of x and y dims of vector
            self.A1[2*k:2*(k+1), 2*vi_idx:2*(vi_idx+1)] = -np.identity(2)
            self.A1[2*k:2*(k+1), 2*vj_idx:2*(vj_idx+1)] = np.identity(2)

            # Find the 'neighbor' vertices for this edge: {v_i, v_j,v_r, v_l}
            vi_vnbr_idxs: Set[np.int32] = v_vnbr_idxs[vi_idx]
            vj_vnbr_idxs: Set[np.int32] = v_vnbr_idxs[vj_idx]
            e_vnbr_idxs: List[np.int32] = list(vi_vnbr_idxs.intersection(vj_vnbr_idxs))
            e_vnbr_idxs.insert(0, vi_idx)
            e_vnbr_idxs.insert(1, vj_idx)

            e_vnbr_xys: Tuple[np.float32, np.float32] = tuple([self.vertices[v_idx] for v_idx in e_vnbr_idxs])

            _: List[Tuple[float, float]] = []
            for v in e_vnbr_xys[1:]:
                vx: float = v[0] - e_vnbr_xys[0][0]
                vy: float = v[1] - e_vnbr_xys[0][1]
                _.extend(((vx, vy), (vy, -vx)))
            G_k: npt.NDArray[np.float32] = np.array(_)

            G_k_star: npt.NDArray[np.float32] = np.linalg.inv(G_k.T @ G_k) @ G_k.T

            e_kx, e_ky = self.edge_vectors[k]
            e = np.array([
                [e_kx,  e_ky],
                [e_ky, -e_kx]
            ], np.float32)

            edge_matrix = np.hstack([np.tile(-np.identity(2), (len(e_vnbr_idxs)-1, 1)), np.identity(2*(len(e_vnbr_idxs)-1))])
            g = np.dot(G_k_star, edge_matrix)
            h = np.dot(e, g)

            for h_offset, v_idx in enumerate(e_vnbr_idxs):
                self.A1[2*k:2*(k+1), 2*v_idx:2*(v_idx+1)] -= h[:, 2*h_offset:2*(h_offset+1)]
                G[2*k:2*(k+1), 2*v_idx:2*(v_idx+1)] = g[:, 2*h_offset:2*(h_offset+1)]

        # populate bottom row of A1, one row per constraint-dimension
        for pin_idx, pin_bc in enumerate(pins_bc):
            for v_idx, v_w in pin_bc:
                self.A1[2*self.edge_num + 2*pin_idx  , 2*v_idx]     = self.w * v_w  # x component
                self.A1[2*self.edge_num + 2*pin_idx+1, 2*v_idx + 1] = self.w * v_w  # y component

        A2_top: npt.NDArray[np.float32] = np.zeros([self.edge_num, self.vert_num], dtype=np.float32)
        for k, (vi_idx, vj_idx) in enumerate(self.e_v_idxs):
            A2_top[k, vi_idx] = -1
            A2_top[k, vj_idx] = 1

        A2_bot: npt.NDArray[np.float32] = np.zeros([self.pin_num, self.vert_num], dtype=np.float32)
        for pin_idx, pin_bc in enumerate(pins_bc):
            for v_idx, v_w in pin_bc:
                A2_bot[pin_idx, v_idx] = self.w * v_w

        self.A2: npt.NDArray[np.float32] = np.vstack([A2_top, A2_bot])

        # for speed, convert to sparse matrices and cache for later
        self.tA1: csr_matrix = sp.csr_matrix(self.A1.transpose())
        self.tA2: csr_matrix = sp.csr_matrix(self.A2.transpose())
        self.G: csr_matrix = sp.csr_matrix(G)

        # perturbing singular matrix and calling det can trigger overflow warning- ignore it
        old_settings = np.seterr(over='ignore')

        # ensure tA1xA1 matrix isn't singular and cache sparse repsentation
        tA1xA1_dense: npt.NDArray[np.float32] = self.tA1 @ self.A1
        while np.linalg.det(tA1xA1_dense) == 0.0:
            logging.info('tA1xA1 is singular. perturbing...')
            tA1xA1_dense += 0.00000001 * np.identity(tA1xA1_dense.shape[0])
        self.tA1xA1: csr_matrix = sp.csr_matrix(tA1xA1_dense)

        # ensure tA2xA2 matrix isn't singular and cache sparse repsentation
        tA2xA2_dense: npt.NDArray[np.float32] = self.tA2 @ self.A2
        while np.linalg.det(tA2xA2_dense) == 0.0:
            logging.info('tA2xA2 is singular. perturbing...')
            tA2xA2_dense += 0.00000001 * np.identity(tA2xA2_dense.shape[0])
        self.tA2xA2: csr_matrix = sp.csr_matrix(tA2xA2_dense)

        # revert np overflow warnings behavior
        np.seterr(**old_settings)

        # set up edge matrix for faster solving
        l1 = np.array(list(range(4*len(self.edge_vectors))))
        l2 = np.repeat(np.array(list(range(2*len(self.edge_vectors)))), 2)
        p = np.array([l1, l2]).T

        self.E0 = np.zeros([4*len(self.edge_vectors), 2*len(self.edge_vectors)])
        for idx, (rdx, cdx) in enumerate(p.tolist()[::4]):
            self.E0[rdx, cdx] = self.edge_vectors[idx, 0]
            self.E0[rdx+1, cdx] = self.edge_vectors[idx, 1]
            self.E0[rdx+2, cdx+1] = self.edge_vectors[idx, 0]
            self.E0[rdx+3, cdx+1] = self.edge_vectors[idx, 1]
        self.E0: csr_matrix = sp.csr_matrix(self.E0)

        # cache other things for quicker solves
        self.c_idxs = list(range(0, 2*len(self.edge_vectors), 2))
        self.s_idxs = list(range(1, 2*len(self.edge_vectors) + 1, 2))
        self.T2 = np.empty([1, 4*len(self.edge_vectors)])

    def solve(self, pins_xy_: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """
        After ARAP has been initialized, pass in new pin xy positions and receive back the new mesh vertex positions
        pins *must* be in the same order they were passed in during initialization

        pins_xy: ndarray [N, 2] with new pin xy positions
        return: ndarray [N, 2], the updated xy locations of each vertex in the mesh
        """

        # remove any pins that were orgininally outside the mesh
        pins_xy: npt.NDArray[np.float32] = pins_xy_[self.pin_mask]  # pyright: ignore[reportGeneralTypeIssues]

        assert len(pins_xy) == self.pin_num

        self.b1: npt.NDArray[np.float64] = np.hstack([np.zeros([2 * self.edge_num], dtype=np.float64), self.w * pins_xy.reshape([-1, ])])
        v1: npt.NDArray[np.float64] = spla.spsolve(self.tA1xA1, self.tA1 @ self.b1.T)

        T1: npt.NDArray[np.float64] = self.G @ v1

        c: npt.NDArray[np.float32] = T1[self.c_idxs]
        s: npt.NDArray[np.float32] = T1[self.s_idxs]
        scale = 1.0 / np.sqrt(c * c + s * s)
        c *= scale
        s *= scale

        self.T2[0, 0::4] = c
        self.T2[0, 1::4] = s
        self.T2[0, 2::4] = -s
        self.T2[0, 3::4] = c

        B2_top = self.T2 @ self.E0

        B2_topx = B2_top[0, ::2]
        B2_topy = B2_top[0, 1::2]
        w_pins_xy = self.w * pins_xy

        b2x = np.concatenate([B2_topx, w_pins_xy[:, 0]])
        b2y = np.concatenate([B2_topy, w_pins_xy[:, 1]])

        v2x: npt.NDArray[np.float64] = spla.spsolve(self.tA2xA2, self.tA2 @ b2x)
        v2y: npt.NDArray[np.float64] = spla.spsolve(self.tA2xA2, self.tA2 @ b2y)

        return np.vstack((v2x, v2y)).T

    @staticmethod
    def plot_mesh(vertices, triangles, pins_xy):
        """ Helper function to visualize mesh deformation outputs """
        import matplotlib.pyplot as plt

        for tri in triangles:
            x_points = []
            y_points = []
            v0, v1, v2 = tri.tolist()
            x_points.append(vertices[v0][0])
            y_points.append(vertices[v0][1])
            x_points.append(vertices[v1][0])
            y_points.append(vertices[v1][1])
            x_points.append(vertices[v2][0])
            y_points.append(vertices[v2][1])
            x_points.append(vertices[v0][0])
            y_points.append(vertices[v0][1])

            plt.plot(x_points, y_points)
        plt.ylim((-15, 15))
        plt.xlim((-15, 15))

        for pin in pins_xy:
            plt.plot(pin[0], pin[1], color='red', marker='o')

        plt.show()


class ARAP_Sorkine(ARAP):
    """
    An implementation of Olga Sorkine and Marc Alexa's As-Rigid-As-Possible Surface Modeling.
    Assumes triangular meshes.
    """

    def __init__(self, V: npt.NDArray[np.float32], F: npt.NDArray[np.int32], constraint_joint_locations: npt.NDArray[np.float32]):
        """
        V: (n,3) ndarray with cartesian coordinates of mesh vertices
        F: (n,3) ndarray with vertex ids of each face within the triangular mesh
        constrained_joint_locations: (n,3) ndarray with initial xyz joint locations used to guide constraint vertices
        """

        assert constraint_joint_locations.shape[-1] == 3

        t = time.time()

        self.verts: npt.NDArray[np.float32] = V   # original vertex cartesian coordinates, p_i
        self.vert_num: int = self.verts.shape[0]  # number of vertices in the mesh

        self.faces: npt.NDArray[np.int32] = F     # triplet of vertex indices
        self.face_num: int = self.faces.shape[0]  # number of vertices in the mesh

        self.constrained_vertex_ids = []  # the indices of vertex to use as constraint points
        self.constraint_vertex_offsets = []  # the initial offsets between the joint locations and the vertex locations
        for x, y, z in constraint_joint_locations:

            # find the closest vertex to the joint
            cv = np.argmin(np.linalg.norm(self.verts - (x, y, z), axis=1))

            # add it to our list of constraint vertices
            self.constrained_vertex_ids.append(cv)

            # and record its offset for later use
            self.constraint_vertex_offsets.append(self.verts[cv, :3] - (x, y, z))

        self.verts_prime: npt.NDArray[np.float32] = self.verts.copy()  # initialize p_i' to a copy of p_i

        # given tuple of sorted (ascending) vertex ids of edge, returns a list of the third vertices of faces adjacent to edge
        self._edge_to_third_vert: defaultdict[Tuple[int, int], List[int]] = defaultdict(list)

        # mesh neighbor matrix. [i,j]=1 if edge between v_i and v_j, 0 otherwise
        self.neighbor_matrix: npt.NDArray[np.bool8] = np.zeros([self.vert_num, self.vert_num], np.bool8)

        for v1, v2, v3 in self.faces:
            self.neighbor_matrix[v1, v2] = self.neighbor_matrix[v2, v1] = 1
            self.neighbor_matrix[v2, v3] = self.neighbor_matrix[v3, v2] = 1
            self.neighbor_matrix[v3, v1] = self.neighbor_matrix[v1, v3] = 1

            self._edge_to_third_vert[tuple(sorted([v1, v2]))].append(v3)
            self._edge_to_third_vert[tuple(sorted([v2, v3]))].append(v1)
            self._edge_to_third_vert[tuple(sorted([v3, v1]))].append(v2)

        # map vertex id to neighboring vertex ids for quickly look up later
        self.v_id_to_nv_id = {}
        for v_id in range(self.vert_num):
            self.v_id_to_nv_id[v_id] = np.where(self.neighbor_matrix[v_id, :] == 1)[0]

        """ Build Weight Matrix """
        self.weight_matrix = np.zeros((self.vert_num, self.vert_num), np.float64)
        self.weight_sum = np.zeros((self.vert_num, self.vert_num), np.float32)

        for v_i in range(self.vert_num):  # for vertex v_i
            for v_j in self.v_id_to_nv_id[v_i]:  # for each neighbor of v_i, v_j

                # calculate the cotan weight of the edges
                for v_o in self._edge_to_third_vert[tuple(sorted([v_i, int(v_j)]))]:
                    v1 = self.verts[v_i] - self.verts[v_o]
                    v2 = self.verts[v_j] - self.verts[v_o]
                    theta = math.acos(v1.dot(v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
                    cot_theta = 1 / math.tan(theta)
                    self.weight_matrix[v_i, v_j] += 0.5 * cot_theta

        """ Build the Laplacian"""
        self.laplacian = -self.weight_matrix.copy()
        for v_id in range(self.vert_num):
            self.laplacian[v_id, v_id] = np.sum(self.weight_matrix[v_id, :])

        # Add additional rows and cols for each constraint
        new_matrix_n = self.vert_num + len(self.constrained_vertex_ids)
        new_matrix = np.zeros((new_matrix_n, new_matrix_n), np.float32)
        new_matrix[:self.vert_num, :self.vert_num] = self.laplacian
        for idx, v_id in enumerate(self.constrained_vertex_ids):
            new_matrix[idx + self.vert_num, v_id] = 1
            new_matrix[v_id, idx + self.vert_num] = 1
        self.laplacian = new_matrix

        # perturb if is singular
        if np.linalg.det(self.laplacian) == 0.0:
            print('laplacian is singular. Perturbing')
            self.laplacian += 0.0000001 * np.identity(self.laplacian.shape[0])

        # precompute p_i
        self.p_i_array = []
        for v_i in range(self.vert_num):
            n_vert_ids = self.v_id_to_nv_id[v_i]

            p_i = np.zeros((3, len(n_vert_ids)))
            for n_idx, v_j in enumerate(n_vert_ids):
                p_i[:, n_idx] = (self.verts[v_i] - self.verts[v_j])

            self.p_i_array.append(p_i)

        # precompute D_i, diagonal matrix containing the weights for edges from v_id to each of its neighbors
        self.D_i_array = []
        for v_id in range(self.vert_num):
            n_vert_ids = self.v_id_to_nv_id[v_id]
            D_i = np.zeros((len(n_vert_ids), len(n_vert_ids)))
            for n_i, n_id in enumerate(n_vert_ids):
                D_i[n_i, n_i] = self.weight_matrix[v_id, n_id]
            self.D_i_array.append(D_i)

        self.P_i_dot_D_i_array = []
        for v_id in range(self.vert_num):
            P_i = self.p_i_array[v_id]
            D_i = self.D_i_array[v_id]
            self.P_i_dot_D_i_array.append(P_i.dot(D_i))

        print(f'precomputation time: {time.time() - t}')

        # store the cell rotations used to perform deformation
        self.cell_rotations = np.zeros((self.vert_num, 3, 3))

    def solve(self, constrained_vertex_positions: npt.NDArray[np.float32], iterations: int = 8):

        assert len(self.constrained_vertex_ids) == constrained_vertex_positions.shape[0]

        # initialize b with constraints
        self.b_array = np.zeros((self.vert_num + len(self.constrained_vertex_ids), 3))
        for i in range(len(self.constrained_vertex_ids)):
            self.b_array[self.vert_num + i] = constrained_vertex_positions[i]

        for _ in range(iterations):

            for v_id in range(self.vert_num):

                vert_i_prime = self.verts_prime[v_id]
                n_vert_ids = self.v_id_to_nv_id[v_id]

                # the deformed edges extending out from v_id
                P_i_prime = (vert_i_prime - self.verts_prime[n_vert_ids]).T

                S_i = self.P_i_dot_D_i_array[v_id].dot(P_i_prime.transpose())

                U, _, V_transpose = np.linalg.svd(S_i)

                rotation = V_transpose.transpose().dot(U.transpose())
                if np.linalg.det(rotation) <= 0:
                    U[:0] *= -1
                    rotation = V_transpose.transpose().dot(U.transpose())

                self.cell_rotations[v_id] = rotation

                """ apply cell rotation and solve for updated verts_prime """

            for v_i in range(self.vert_num):
                b = np.zeros((1, 3))
                for v_j in self.v_id_to_nv_id[v_i]:
                    w_ij = self.weight_matrix[v_i, v_j] / 2.0
                    r_ij = self.cell_rotations[v_i] + self.cell_rotations[v_j]
                    p_ij = self.verts[v_i] - self.verts[v_j]
                    b += (w_ij * r_ij.dot(p_ij))
                self.b_array[v_i] = b

            self.verts_prime = np.linalg.solve(self.laplacian, self.b_array)[:self.vert_num]

        return self.verts_prime

    """ development and debugging methods"""
    @staticmethod
    def show_graph(verts_prime, faces):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = np.squeeze(np.asarray(verts_prime[:, 0]))
        ys = np.squeeze(np.asarray(verts_prime[:, 1]))
        zs = np.squeeze(np.asarray(verts_prime[:, 2]))

        ax.scatter(xs, ys, zs)

        for v1, v2, v3 in faces:
            x = []
            y = []
            z = []

            x.append(verts_prime[v1][0])
            y.append(verts_prime[v1][1])
            z.append(verts_prime[v1][2])

            x.append(verts_prime[v2][0])
            y.append(verts_prime[v2][1])
            z.append(verts_prime[v2][2])

            x.append(verts_prime[v3][0])
            y.append(verts_prime[v3][1])
            z.append(verts_prime[v3][2])

            x.append(verts_prime[v1][0])
            y.append(verts_prime[v1][1])
            z.append(verts_prime[v1][2])

            ax.plot(x, y, z)

        plt.show()

    @staticmethod
    def load_mesh_from_off_fn(mesh_fn: str):
        """ For develop purposes only. Sets self.verts and self.faces """
        with open(mesh_fn, 'r') as f:
            data = f.read().split('\n')

            data.pop(0)  # OFF
            data.pop(0)  # blank
            vert_num, face_num = [int(x) for x in data.pop(0).split(' ')[:-1]]
            data.pop(0)  # blank

            # initialize p_i
            verts = np.empty([vert_num, 3], np.float32)
            for vdx in range(vert_num):
                verts[vdx, :] = [float(x) for x in data.pop(0).split(' ')]

            data.pop(0)  # blank

            faces = np.empty((face_num, 3), np.int32)
            for fdx in range(face_num):
                faces[fdx, :] = [int(x) for x in data.pop(0).split(' ')[1:]]

        return verts, faces

    @staticmethod
    def test_with_off(off_fn: str):

        V, F = ARAP_Sorkine.load_mesh_from_off_fn(off_fn)
        cv1 = np.argmin(V[:, 0])
        cv2 = np.argmax(V[:, 0])

        arap = ARAP_Sorkine(V, F, np.array([cv1, cv2]))

        constrained_vertex_positions = np.array([[-1.0, 0.7, 0.0], [2.0, 0.7, 1.0]])
        arap.solve(constrained_vertex_positions)

        ARAP_Sorkine.show_graph(arap.verts_prime, F)
