from ad3d_server.transform import Transform
from ad3d_server.animated_drawing_rig import AnimatedDrawingRig
from ad3d_server.joint import Joint
from ad3d_server.arap import ARAP, ARAP_Igarashi
from ad3d_server.config import CharacterConfig
from ad3d_server.foot_flipper import foot_flip

from abc import abstractmethod
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
import numpy as np
import numpy.typing as npt
import copy
import heapq
import logging
from skimage import measure
import triangle as tr
import igl


class AnimatedDrawingMesh(Transform):
    """
    Mesh used by the AnimatedDrawing class. AnimatedDrawingMesh is responsible for keeping track of the vertices and faces of the mesh.
    Currently only triangle meshes are supported.
    Class is responsible for rendering itself.
    Therefore, it must also keep track of the order in which to render the vertices.
    """

    def __init__(self, vertices: npt.NDArray[np.float32], triangles: npt.NDArray[np.int32], txtr_dict: Dict[str, npt.NDArray[np.uint8]]):
        super().__init__()

        self.vertices: npt.NDArray[np.int32] = vertices

        self.triangles: npt.NDArray[np.int32] = triangles  # (n,3) ndarray with the vertex ids of the mesh faces

        self.indices: npt.NDArray[np.int32] = self.triangles.flatten()  # the order in which to render the vertices

        self.txtr_dict: Dict[str, npt.NDArray[np.uint8]] = txtr_dict  # image texture for the character
        self.active_txtr_name: str = 'front'  # specifies name of texture to apply to character.

        self.deformer: ARAP

    @abstractmethod
    def set_active_txtr_name(self, txtr_name: str):
        raise NotImplementedError

    @abstractmethod
    def process_motion_source_joint_depths(self, name_dist: List[str]):
        raise NotImplementedError

    @abstractmethod
    def deform(self, constraint_positions: npt.NDArray[np.float32]):
        assert False, 'AnimatedDrawingMesh subclasses are responsible for implementing deform() method.'

    @abstractmethod
    def _draw(self):
        assert False, 'AnimatedDrawingMesh subclasses are responsible for implementing _draw() method.'

    @abstractmethod
    def initialize_mesh_deformer(self, deformer_type: str, constraints: npt.NDArray[np.float32]):
        assert False, 'AnimatedDrawingMesh subclasses are responsible for implementing initialized_mesh_deformer() method.'

    def generate_2D_mesh_from_mask(self, mask: npt.NDArray[np.uint8], flip=[]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """ Takes in the mask and generates a 2D mesh from it. Returns 2D location of vertices, scaled from 0-1, and vertex indices making up the faces"""
        # get contour lines from the raster mask
        try:
            contours: List[npt.NDArray[np.float64]] = measure.find_contours(mask, 128)
        except Exception as e:
            msg = f'Error finding contours for character mesh: {str(e)}'
            logging.critical(msg)
            assert False, msg

        # if multiple polygons are in mask, use largest, discard others
        if len(contours) > 1:
            msg = f'{len(contours)} separate polygons found in mask. Using largest.'
            logging.info(msg)
            contours.sort(key=len, reverse=True)
        contour = contours[0]

        # smooth out the contour to reduce outside vertex count
        outside_vertices: npt.NDArray[np.float64] = measure.approximate_polygon(contour, tolerance=0.75)

        # if the first and last vertices are the same, remove the last one
        if np.all(outside_vertices[0] == outside_vertices[-1]):
            outside_vertices = outside_vertices[:-1, :]

        # build the dictionary needed for Triangle's Constrained Delaunay solve
        segs = np.array([[i, (i+1) % len(outside_vertices)] for i in range(len(outside_vertices))])

        # modify outside vertices and edges and txtrs if needed for foot orientation
        # TODO: Find all references to left and right foot for flipping, be explicit bout exactly what it refers to- character/motionsource/images left/right
        if "left_foot" in flip:
            outside_vertices, segs = foot_flip(self.char_cfg, outside_vertices, segs, self.txtr_dict, prox_joint_name='right_knee', dist_joint_name='right_foot')
        if "right_foot" in flip:
            outside_vertices, segs = foot_flip(self.char_cfg, outside_vertices, segs, self.txtr_dict, prox_joint_name='left_knee', dist_joint_name='left_foot')

        A = dict(vertices=np.array(outside_vertices), segments=segs, holes=[[0, 0]])

        print('starting solve')
        # triangulation = tr.triangulate(A, 'qpa1000.0')
        triangulation = tr.triangulate(A, 'qpa40.0')
        print('ended solve')

        # refine the mesh to reduce the maximum number of output faces
        _, vertices, triangles, *_ = igl.decimate(triangulation['vertices'], triangulation['triangles'], 5000)

        # scale vertices so they lie between 0-1
        vertices /= mask.shape[0]

        return vertices.astype(np.float32), triangles.astype(np.int32)


class AnimatedDrawingMeshCardboardBase(AnimatedDrawingMesh):
    def __init__(self, mask: npt.NDArray[np.uint8], txtr_dict: Dict[str, npt.NDArray[np.uint8]], flip=[]):
        self.txtr_dict = copy.deepcopy(txtr_dict)  # TODO: txtr dict should be getting set by the parent constructor. How to reshuffle so we can flip the textures without setting this or passing through as parameter.
        front_vertices, front_triangles = self.generate_2D_mesh_from_mask(mask, flip=flip)

        # do whatever is needed to make the cardboard mesh here

        # front vertices lie upon z=0 plane
        front_vertices = np.concatenate((front_vertices, np.zeros((front_vertices.shape[0], 1))), axis=1).astype(np.float32)

        # back vertices have same xy coordinates as front, but different z coordinate
        back_vertices = front_vertices.copy()
        back_vertices[:, 2] = -0.001  # thickness of cardboard character

        # back_triangles vertex indices are same as front, but offset by number of front vertices and ordered in reverse
        back_triangles = front_triangles.copy()[:, ::-1] + front_vertices.shape[0]

        # Use the number of faces incident upon edge to determine which are boundary edges
        vdxs_to_edge_count = {}
        for v0, v1, v2 in front_triangles:
            e0 = tuple(sorted([v0, v1]))
            if e0 in vdxs_to_edge_count:
                del vdxs_to_edge_count[e0]
            else:
                vdxs_to_edge_count[e0] = [v0, v1]

            e1 = tuple(sorted([v1, v2]))
            if e1 in vdxs_to_edge_count:
                del vdxs_to_edge_count[e1]
            else:
                vdxs_to_edge_count[e1] = [v1, v2]

            e2 = tuple(sorted([v2, v0]))
            if e2 in vdxs_to_edge_count:
                del vdxs_to_edge_count[e2]
            else:
                vdxs_to_edge_count[e2] = [v2, v0]
        boundary_edges = np.array(list(vdxs_to_edge_count.values()))

        # 1. Create indices of vertices within frontmesh that lie on the character boundary
        boundaryvertex_indices = list(set(boundary_edges.flatten()))

        # 2. Use it to create copies of the frontmesh boundary vertices
        frontmesh_boundary_vertices = front_vertices[boundaryvertex_indices]

        # 3. Use it to create copies of the backmesh boundary vertices
        backmesh_boundary_vertices = back_vertices[boundaryvertex_indices]

        # 4. Create the triangles needed for the boundary, using the indices of these new boundary vertices
        boundary_triangles: npt.NDArray[np.int32] = np.empty((2 * boundary_edges.shape[0], 3), np.int32)
        for idx, (frontmesh_vdx0, frontmesh_vdx1) in enumerate(boundary_edges):
            boundarymesh_vdx0 = len(front_vertices) + len(back_vertices) + boundaryvertex_indices.index(frontmesh_vdx0)  # boundarymesh vertex0 incident on frontmesh
            boundarymesh_vdx1 = len(front_vertices) + len(back_vertices) + boundaryvertex_indices.index(frontmesh_vdx1)  # boundarymesh vertex1 incident on frontmesh

            boundarymesh_bvdx0 = len(front_vertices) + len(back_vertices) + len(frontmesh_boundary_vertices) + boundaryvertex_indices.index(frontmesh_vdx0)  # boundarymesh vertex0 incident on frontmesh
            boundarymesh_bvdx1 = len(front_vertices) + len(back_vertices) + len(frontmesh_boundary_vertices) + boundaryvertex_indices.index(frontmesh_vdx1)  # boundarymesh vertex1 incident on frontmesh

            boundary_triangles[2*idx,   :] = [boundarymesh_vdx0, boundarymesh_bvdx1, boundarymesh_vdx1]  # boundary triangle1
            boundary_triangles[2*idx+1, :] = [boundarymesh_vdx0, boundarymesh_bvdx0, boundarymesh_bvdx1]  # boundary triangle2

        # 5. Create maps indicating which frontmesh/backmesh vertices and boundary vertices should have the same cartesian coorindates. Useful for updating after deformation.
        self.frontmeshvdx_to_boundarymeshvdx_map = {
            vdx: idx + len(front_vertices) + len(back_vertices) for idx, vdx in enumerate(boundaryvertex_indices)
        }
        self.backmeshvdx_to_boundarymeshvdx_map = {
            vdx + len(front_vertices): idx + len(front_vertices) + len(back_vertices) + len(boundaryvertex_indices) for idx, vdx in enumerate(boundaryvertex_indices)
        }

        # 6. Take vertices from xyz -> xyz, rgb, uv, nx_ny_nz
        front_vertices = np.hstack([
            front_vertices,                               # xyz
            np.zeros([len(front_vertices), 3]),           # rgb
            front_vertices[:, 0:2][:, ::-1].copy(),      # uv
            np.tile([0, 0, 1], [len(front_vertices), 1])  # nx, ny, nz
        ])

        back_vertices = np.hstack([
            back_vertices,                                 # xyz
            np.zeros([len(back_vertices), 3]),             # rgb
            back_vertices[:, 0:2][:, ::-1].copy(),         # uv
            np.tile([0, 0, -1], [len(front_vertices), 1])  # nx, ny, nz
        ])

        frontmesh_boundary_vertices = np.hstack([
            frontmesh_boundary_vertices,                          # xyz
            np.zeros([len(frontmesh_boundary_vertices), 3]),      # rgb
            frontmesh_boundary_vertices[:, 0:2][:, ::-1].copy(),  # uv
            np.zeros([len(frontmesh_boundary_vertices), 3])       # nx, ny, nz
        ])

        backmesh_boundary_vertices = np.hstack([
            backmesh_boundary_vertices,                          # xyz
            np.zeros([len(backmesh_boundary_vertices), 3]),      # rgb
            backmesh_boundary_vertices[:, 0:2][:, ::-1].copy(),  # uv
            np.zeros([len(backmesh_boundary_vertices), 3])       # nx, ny, nz
        ])

        vertices: npt.NDArray[np.float32] = np.concatenate([  # per-meshpart vertex order defined here
            front_vertices,
            back_vertices,
            frontmesh_boundary_vertices,
            backmesh_boundary_vertices
        ]).astype(np.float32)

        triangles: npt.NDArray[np.int32] = np.concatenate((boundary_triangles, front_triangles, back_triangles))
        super().__init__(vertices=vertices, triangles=triangles, txtr_dict=self.txtr_dict)

        self.boundary_triangles_start = 0
        self.boundary_triangles_end = self.boundary_triangles_start + len(boundary_triangles)
        self.front_triangles_start = len(boundary_triangles)
        self.front_triangles_end = self.front_triangles_start + len(front_triangles)
        self.back_triangles_start = len(boundary_triangles) + len(front_triangles)
        self.back_triangles_end = self.back_triangles_start + len(back_triangles)

        # also save the front triangles and front vertices, as we'll need them for igarashi solve
        self.front_vertex_count = len(front_vertices)
        self.front_triangle_count = len(front_triangles)


class AnimatedDrawingMeshCardboardExternal(AnimatedDrawingMeshCardboardBase):
    """ A cardboard mesh, but it will be deformed using ARAP and the rendering order will the influenced by the driving motion data """

    def __init__(self, mask: npt.NDArray[np.uint8], txtr_dict: Dict[str, npt.NDArray[np.uint8]], rig: AnimatedDrawingRig, char_cfg: CharacterConfig, flip=[]):

        self.char_cfg = char_cfg

        super().__init__(mask, txtr_dict, flip)  # what does flip do?

        self.joint_name_to_render_order_vertex_ids: Dict[str, npt.NDArray[np.int32]]
        self._initialize_joint_name_to_render_order(mask, rig)
        self.submesh_render_order: List[str] = None

    def set_active_txtr_name(self, txtr_name: str):
        self.active_txtr_name: str = txtr_name

    def process_motion_source_joint_depths(self, render_ordered_names: List[str]):
        self.submesh_render_order = []
        for name in render_ordered_names:
            if name in self.joint_name_to_render_order_vertex_ids.keys():
                self.submesh_render_order.append(name)

        # # set rendering order for the 'forward' side of character
        # front_ordered_indices, grouped_vertex_indices = self._get_ordered_vertex_indices_from_ordered_joint_names(render_ordered_names)
        # self.indices[3*self.front_triangles_start:3*self.front_triangles_end] = front_ordered_indices

        # # rendering order for 'back' side of character is same as forward, but with part order reversed and with intrapart order reversed, and offset by number of front vertices
        # back_ordered_indices = self._get_ordered_vertex_indices_from_ordered_joint_names(render_ordered_names[::-1])[0][::-1] + self.front_vertex_count
        # self.indices[3*self.back_triangles_start:3*self.back_triangles_end] = back_ordered_indices

        # self.grouped_front_indices = grouped_vertex_indices

    def _get_ordered_vertex_indices_from_ordered_joint_names(self, render_ordered_names: List[str]):

        vertex_indices = []
        grouped_vertex_indices = []

        for name in render_ordered_names:
            try:
                vertex_indices.append(self.joint_name_to_render_order_vertex_ids[name])
                grouped_vertex_indices.append(self.joint_name_to_render_order_vertex_ids[name])
            except KeyError:
                pass

        return np.concatenate(vertex_indices), grouped_vertex_indices

    def initialize_mesh_deformer(self, constraints: npt.NDArray[np.float32]):

        # for arap_igarashi, we only use the vertices of the front of the character to perform deformation
        # also, our vertices and constraints are 2D, not 3D
        self.deformer = ARAP_Igarashi(self.vertices[:self.front_vertex_count, :2], self.triangles[self.front_triangles_start:self.front_triangles_end], constraints[:, :2])

    def _initialize_joint_name_to_render_order(self, mask: npt.NDArray[np.uint8], rig: AnimatedDrawingRig) -> None:
        """ Given the rig, this methods determines which triangles 'belong' to each joint. Used for rendering later"""

        """
        Uses BFS to find and return the closest joint bone (line segment between joint and parent) to each triangle centroid.
        """
        shortest_distance = np.full(mask.shape, 1 << 12, dtype=np.int32)  # to nearest joint
        closest_joint_idx = np.full(mask.shape, -1, dtype=np.int8)  # track joint idx nearest each point

        img_dim = mask.shape[0]

        joints_d = {}
        for joint_name, xyz in zip(rig.root_joint.get_chain_joint_names(), np.array(rig.root_joint.get_chain_worldspace_positions()).reshape([-1, 3])):
            joints_d[joint_name] = xyz[:2]

        # store joint names and later reference by element location
        joint_name_to_idx: List[str] = [name for name in rig.root_joint.get_chain_joint_names()]

        # seed generation
        heap: List[Tuple[float, Tuple[int, Tuple[int, int]]]] = []  # [(dist, (joint_idx, (x, y))]
        for name, xy in joints_d.items():
            if name == 'root':
                continue

            parent_joint = rig.root_joint.get_transform_by_name(name).get_parent()
            assert isinstance(parent_joint, Joint)
            parent_joint_name = parent_joint.name

            joint_idx = joint_name_to_idx.index(name)
            dist_joint_xy: npt.NDArray[np.float32] = xy
            prox_joint_xy: npt.NDArray[np.float32] = joints_d[parent_joint_name]
            seeds_xy = (img_dim * np.linspace(dist_joint_xy, prox_joint_xy, num=20, endpoint=False)).round()
            heap.extend([(0, (joint_idx, tuple(seed_xy.astype(np.int32)))) for seed_xy in seeds_xy])

        # BFS search
        import time
        start_time: float = time.time()
        logging.info('Starting joint -> mask pixel BFS')
        print('starting')
        while heap:
            distance, (joint_idx, (x, y)) = heapq.heappop(heap)
            neighbors = [(x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
            n_dist = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]
            for (n_x, n_y), n_dist in zip(neighbors, n_dist):
                n_distance = distance + n_dist
                if not 0 <= n_x < img_dim or not 0 <= n_y < img_dim:
                    continue  # neighbor is outside image bounds- ignore

                if not mask[n_x, n_y]:
                    continue  # outside character mask

                if shortest_distance[n_x, n_y] <= n_distance:
                    continue  # a closer joint exists

                closest_joint_idx[n_x, n_y] = joint_idx
                shortest_distance[n_x, n_y] = n_distance
                heapq.heappush(heap, (n_distance, (joint_idx, (n_x, n_y))))
        logging.info(f'Finished joint -> mask pixel BFS in {time.time() - start_time} seconds')
        print(f'Finished joint -> mask pixel BFS in {time.time() - start_time} seconds')

        # create map between joint name and triangle centroids it is closest to
        joint_to_tri_v_idx_and_dist: DefaultDict[str, List[Tuple[npt.NDArray[np.int32], np.int32]]] = defaultdict(list)
        for tri_v_idx in self.triangles[self.front_triangles_start:self.front_triangles_end]:
            tri_verts = np.array([self.vertices[v_idx][:3] for v_idx in tri_v_idx])
            centroid_x, centroid_y, _ = list((tri_verts.mean(axis=0) * img_dim).round().astype(np.int32))
            tri_centroid_closest_joint_idx: np.int8 = closest_joint_idx[centroid_x, centroid_y]
            dist_from_tri_centroid_to_bone: np.int32 = shortest_distance[centroid_x, centroid_y]
            joint_to_tri_v_idx_and_dist[joint_name_to_idx[tri_centroid_closest_joint_idx]].append((tri_v_idx, dist_from_tri_centroid_to_bone))

        joint_to_tri_v_idx: Dict[str, npt.NDArray[np.int32]] = {}
        for key, val in joint_to_tri_v_idx_and_dist.items():
            # sort by distance, descending
            val.sort(key=lambda x: float(x[1]), reverse=True)

            # retain vertex indices, remove distance info
            val = [v[0] for v in val]

            # convert to np array and save in dictionary
            joint_to_tri_v_idx[key] = np.array(val).flatten()  # type: ignore

        self.joint_name_to_render_order_vertex_ids = joint_to_tri_v_idx

    def get_ordered_triangle_indices_and_submesh_info(self) -> List[int]:
        """ Builds and returns the information necessary to later recover submesh triangle indices from submesh_name. """
        render_triangles = []  # indices specifying triangles of the mesh

        submesh_names: List[str] = []  # human readable name that will be used identify submesh when specifying render order
        submesh_starting_indices = []  # index of render_triangles where each submesh starts
        submesh_lengths = []  # number of indices that define triangles for this submesh

        for key, val in self.joint_name_to_render_order_vertex_ids.items():
            submesh_names.append(key)
            submesh_starting_indices.append(len(render_triangles))
            submesh_lengths.append(val.shape[0])
            render_triangles.extend(val.tolist())

        return (render_triangles, submesh_names, submesh_starting_indices, submesh_lengths)

    def deform(self, constraint_positions: npt.NDArray[np.float32]):

        constraint_positions = constraint_positions[:, :2]  # expects 2D constraints, not 3D

        # use solve results directly for front mesh vertex coordinates
        self.vertices[:self.front_vertex_count, :2] = self.deformer.solve(constraint_positions)

        # use solve results to compute back mesh and boundary mesh vertex coordinates
        self.vertices[self.front_vertex_count:2*self.front_vertex_count, :2] = self.vertices[:self.front_vertex_count, :2]  # back mesh vertices
        self.vertices[list(self.frontmeshvdx_to_boundarymeshvdx_map.values()), :3] = self.vertices[list(self.frontmeshvdx_to_boundarymeshvdx_map.keys()), :3]  # frontmesh boundary vertices
        self.vertices[list(self.backmeshvdx_to_boundarymeshvdx_map.values()), :3] = self.vertices[list(self.backmeshvdx_to_boundarymeshvdx_map.keys()), :3]  # backmesh boundary vertices

        self._compute_boundary_normals()

    def _compute_boundary_normals(self) -> None:

        boundary_triangles = self.triangles[self.boundary_triangles_start:self.boundary_triangles_end]
        boundary_vertices = boundary_triangles.flatten()

        self.vertices[boundary_vertices][:, 8:11] = [0, 0, 0]  # set normals to zero

        # compute necessary per face edges
        v0xyz, v1xyz, v2xyz = np.swapaxes(self.vertices[boundary_triangles, 0:3], 0, 1)
        e0 = (v1xyz - v0xyz)
        e1 = (v2xyz - v0xyz)

        # cross to get per face normals, normalize
        normal = np.cross(e0, e1)
        normal /= np.linalg.norm(normal, axis=1).reshape([-1, 1])

        # add each per face normal to each vertex within that face
        self.vertices[boundary_triangles[:, 0], 8:11] += normal
        self.vertices[boundary_triangles[:, 1], 8:11] += normal
        self.vertices[boundary_triangles[:, 2], 8:11] += normal

        # normalize boundary vertex normals
        self.vertices[boundary_vertices, 8:11] /= np.linalg.norm(self.vertices[boundary_vertices, 8:11], axis=1).reshape([-1, 1])
