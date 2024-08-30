
from ad3d_server.transform import Transform
from ad3d_server.animated_drawing_rig import AnimatedDrawingRig
from ad3d_server.animated_drawing_part import AnimatedDrawingInternalPart
from ad3d_server.animated_drawing_mesh import AnimatedDrawingMesh, AnimatedDrawingMeshCardboardExternal
from ad3d_server.retargeter import RetargeterTwistedPerspective2D
from ad3d_server.config import CharacterViewConfig, CharacterConfig
import ad3d_server.animated_drawing_utils as ad_utils

import numpy.typing as npt
import numpy as np
from typing import Dict
from functools import reduce
import time

import io
import base64
import png


class ServerAnimatedDrawing(Transform):

    def __init__(self, char_cfg_fn):

        super().__init__()

        self.char_cfg: CharacterConfig = CharacterConfig(char_cfg_fn)

        self.views = {}
        self.views['left_view'] = ServerAnimatedDrawingView(self.char_cfg.left_view)
        self.add_child(self.views['left_view'])
        self.views['right_view'] = ServerAnimatedDrawingView(self.char_cfg.right_view)
        self.add_child(self.views['right_view'])

        self.active_view_name = 'right_view'  # default to right

        # variables to keep track of the motion source pose being applied to the character
        self.ms_joint_names = None
        self.ms_joint_positions = None
        self.ms_forward_vector = None

        # viewer position
        self.viewer = Transform()

        # view_theta
        self.view_theta = None

        # rotation plane
        self.plane_rotation = None

    def register_motion_source(self, motion_source_retarget_json):
        for view in self.views.values():
            view.retargeter.register_motion_source(motion_source_retarget_json)

    def set_viewer_position(self, pos: npt.NDArray[np.float32]) -> None:
        """ Sets the Animated_Drawing's viewer's world position """

        self.viewer.set_position(pos)

        for view in self.views.values():
            view.retargeter._viewer.set_position(pos)

    def set_motion_source_joint_positions(self, joint_names, joint_positions):
        self.ms_joint_names = joint_names
        self.ms_joint_positions = joint_positions

    def set_motion_source_root_offset(self, ms_root_offset):
        self.ms_root_offset = ms_root_offset

    def set_motion_source_forward_vector(self, ms_forward_vector):
        self.ms_forward_vector = ms_forward_vector

    def is_part_shared_with_client(self, part, view):
        """ helper function to determine if a particular's view's part must be sent to the client or can be safely ignored during visualization """
        # if part has no mesh, client doesn't need to know about it
        if part.mesh is None:
            return False

        # if part doesn't translate and none of it's parent parts translate, client doesn't need to know about it
        _part = part
        hide_from_client = False
        while True:
            if _part.does_rdtwp:  # if part/parent does move, we include it
                break
            elif _part.does_rdtwp == False and _part.parent_name in view.name_to_part.keys():  # progress up tree
                _part = view.name_to_part[_part.parent_name]
            else:
                hide_from_client = True
                break
        if hide_from_client:
            return False

        return True

    def get_meshes_for_client(self):
        meshes = []

        def txtr_to_base64(txtr):
            height = txtr.shape[0]
            width = txtr.shape[1]
            channel_num = txtr.shape[2]
            img = txtr.reshape([txtr.shape[0], -1])
            buffer = io.BytesIO()
            if channel_num == 3:
                w = png.Writer(width, height, greyscale=False)
            elif channel_num == 4:
                w = png.Writer(width, height, greyscale=False, alpha=True)
            else:
                assert False, f'unsupported number of channels in texture: {channel_num}'
            w.write(buffer, img)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        for view_name, view in self.views.items():

            print("base...")
            # get the base meshes for this view
            for mesh_name, mesh in view.meshes.items():
                mesh_vertices_xy = mesh.vertices[:mesh.front_vertex_count, :2].flatten().tolist()
                mesh_vertices_uv = mesh.vertices[:mesh.front_vertex_count, 6:8].flatten().tolist()
                mesh_vertices_normal = mesh.vertices[:mesh.front_vertex_count, 8:11].flatten().tolist()
                mesh_triangles, submesh_names, submesh_starting_indices, submesh_lengths = mesh.get_ordered_triangle_indices_and_submesh_info()

                mesh_txtr_names = []
                mesh_txtrs = []
                for name, txtr in mesh.txtr_dict.items():
                    mesh_txtr_names.append(name)
                    mesh_txtrs.append(txtr_to_base64(txtr))

                meshes.append({
                    'name': f'{view_name}|{mesh_name}',
                    'xy': mesh_vertices_xy,
                    'uv': mesh_vertices_uv,
                    'normal': mesh_vertices_normal,
                    'triangles': mesh_triangles,
                    'submesh_names': submesh_names,
                    'submesh_starting_indices': submesh_starting_indices,
                    'submesh_num_indices': submesh_lengths,
                    'txtr_names': mesh_txtr_names,
                    'txtrs': mesh_txtrs,
                    'hide_outside_of': ""
                })
                print(f'{view_name}|{mesh_name}')

            print("parts...")
            # get the part meshes for this view
            for part_name, part in view.name_to_part.items():
                mesh = part.mesh

                if not self.is_part_shared_with_client(part, view):
                    continue

                """ for meshes we are sending to client, get necessary info the send """
                mesh_vertices_xy = mesh.vertices[:mesh.front_vertex_count, :2].flatten().tolist()
                mesh_vertices_uv = mesh.vertices[:mesh.front_vertex_count, 6:8].flatten().tolist()
                mesh_vertices_normal = mesh.vertices[:mesh.front_vertex_count, 8:11].flatten().tolist()
                mesh_triangles = mesh.triangles[mesh.front_triangles_start:mesh.front_triangles_end].flatten().tolist()

                mesh_txtr_names = []
                mesh_txtrs = []
                for name, txtr in mesh.txtr_dict.items():
                    # if texture is None, then it's identical to view's original texture.
                    if txtr is None:
                        txtr = view.txtrs['original']

                    mesh_txtr_names.append(name)
                    mesh_txtrs.append(txtr_to_base64(txtr))

                # traverse up the part tree to see if it should be hidden outside of anything
                hide_outside_of = ""
                _part = part
                while _part is not None:
                    if _part.hide_outside_parent == True:  # save the first hide_outside_parent we find on the way up tree
                        hide_outside_of = _part.parent_name
                        break
                    if _part.parent_name not in view.name_to_part.keys():  # if parent is not a part, abort. otherwise progress up tree
                        break
                    _part = view.name_to_part[_part.parent_name]

                meshes.append({
                    'name': f'{view_name}|{part_name}',
                    'xy': mesh_vertices_xy,
                    'uv': mesh_vertices_uv,
                    'normal': mesh_vertices_normal,
                    'triangles': mesh_triangles,
                    'submesh_names': [],
                    'submesh_starting_indices': [],
                    'submesh_num_indices': [],
                    'txtr_names': mesh_txtr_names,
                    'txtrs': mesh_txtrs,
                    'hide_outside_of': hide_outside_of
                })
                print(f'{view_name}|{part_name}')
        return {'meshes': meshes}

    def get_plane_rotation_for_client(self) -> float:
        return self.plane_rotation

    def get_animated_drawing_translation_for_client(self):
        return self._translate_m[:-1, -1]

    def get_mesh_vertex_xy_updates_for_client(self) -> Dict[str, str]:
        """
        returns dictionary containing updated xy coordinates for any mesh that has been modified this frame
        key: name of mesh
        val: str of list of updated xy coordinates
        """
        ret = []

        active_view = self.views[self.active_view_name]
        mesh_name = active_view.active_mesh_name
        active_view_mesh = active_view.meshes[active_view.active_mesh_name]
        updated_vertex_xy = active_view_mesh.vertices[:active_view_mesh.front_vertex_count, :2].flatten().tolist()

        ret.append({"name": f'{self.active_view_name}|{mesh_name}', "xy": updated_vertex_xy})

        """ Add in code to get updated part vertices, in root local space, and add them to ret """
        for part_name, part in active_view.name_to_part.items():

            if not self.is_part_shared_with_client(part, active_view):
                continue

            local_transform_stack = [part._local_transform]
            part_ = part
            while part_.parent_name in active_view.name_to_part.keys():
                part_ = active_view.name_to_part[part_.parent_name]
                local_transform_stack.insert(0, part_._local_transform)
            root_relative_transform = reduce(np.matmul, local_transform_stack)

            part_mesh_xyz1 = np.hstack([part.mesh.vertices[:part.mesh.front_vertex_count, :3], np.ones([part.mesh.front_vertex_count, 1])]).T
            root_relative_vertex_xy = (root_relative_transform @ part_mesh_xyz1)[:2, :].T

            ret.append({"name": f'{self.active_view_name}|{part_name}', "xy": root_relative_vertex_xy.flatten().tolist()})

        return ret

    def get_mesh_active_texture_names_for_client(self):

        ret = []

        # get active texture for base mesh
        active_view = self.views[self.active_view_name]
        mesh_name = active_view.active_mesh_name
        active_view_mesh = active_view.meshes[active_view.active_mesh_name]
        active_base_texture = active_view_mesh.active_txtr_name
        ret.append({"name": f'{self.active_view_name}|{mesh_name}', "txtr": active_base_texture})

        # get active textures for visible parts
        for _, part in active_view.name_to_part.items():

            if not self.is_part_shared_with_client(part, active_view):
                continue

            part_is_visible = part.is_visible
            parent_name = part.parent_name
            while parent_name in active_view.name_to_part.keys() and part_is_visible:
                parent_part = active_view.name_to_part[parent_name]
                part_is_visible &= parent_part.is_visible
                parent_name = parent_part.parent_name
            if not part_is_visible:
                continue

            ret.append({"name": f'{self.active_view_name}|{part.name}',
                       "txtr": part.mesh.active_txtr_name})

        return ret

    def get_meshes_renderorders_for_client(self):
        """ returns a list of tuples, where [0] is name of mesh and [1] is list of indices for rendering. if [1] is empty, render all indices of mesh """
        active_view = self.views[self.active_view_name]
        mesh_name = active_view.active_mesh_name
        active_view_mesh = active_view.meshes[active_view.active_mesh_name]
        ret = []

        # motion not updated yet
        if active_view_mesh.submesh_render_order is None:
            return []

        # first do the different regions of the base mesh
        for submesh_name in active_view_mesh.submesh_render_order:
            name = f'{self.active_view_name}|{mesh_name}'
            ret.append({"name": name, "submesh": submesh_name})

        # next do the parts
        for _, part in active_view.name_to_part.items():
            # don't send mesh for part if it or any of it's parents is not visible for this frame
            part_is_visible = part.is_visible
            parent_name = part.parent_name
            while parent_name in active_view.name_to_part.keys() and part_is_visible:
                parent_part = active_view.name_to_part[parent_name]
                part_is_visible &= parent_part.is_visible
                parent_name = parent_part.parent_name
            if not part_is_visible:
                continue

            name = f'{self.active_view_name}|{part.name}'
            ret.append({"name": name, "submesh": ""})

        return ret

    def update_view_theta(self) -> None:
        # get xz components of view vector and motion source forward vector
        view_vec = (self.update_and_get_world_position() - self.viewer.update_and_get_world_position())[[0, 2]]
        fwd_vec = self.ms_forward_vector[[0, 2]]

        # normalize
        if np.isclose(0, np.linalg.norm(view_vec)):
            view_vec_n = np.array(0.0, 1.0)
        else:
            view_vec_n = view_vec / np.linalg.norm(view_vec)
        fwd_vec_n = fwd_vec / np.linalg.norm(fwd_vec)

        # compute view theta
        _y =  fwd_vec_n[1] * -view_vec_n[0] - fwd_vec_n[0] * -view_vec_n[1]
        _x = -fwd_vec_n[0] * -view_vec_n[0] + fwd_vec_n[1] * view_vec_n[1]
        theta = np.arctan2(_y, _x)
        self.view_theta = np.degrees(theta) % 360

    def update_active_view(self):
        if 0 <= self.view_theta < 180:
            self.active_view_name = 'left_view'
        elif 180 <= self.view_theta < 360:
            self.active_view_name = 'right_view'

    def update(self):

        # update view viewing angle
        self.update_view_theta()

        # determine which view to use
        self.update_active_view()
        active_view = self.views[self.active_view_name]

        # update the correct rig
        rig_orientations = active_view.retargeter.get_character_bone_orientations(self.ms_joint_names, self.ms_joint_positions, self.view_theta)
        active_view.rig.set_global_orientations(rig_orientations)
        active_view.update_transforms(update_ancestors=True, recurse_on_children=True)

        # set the correct active mesh, based on foot orientation
        active_view.active_mesh_name = active_view.retargeter.get_feet_orientation()
        active_mesh = active_view.meshes[active_view.active_mesh_name]

        # deform the base mesh
        active_mesh.deform(active_view.rig.get_joint_positions())

        # update each part based on viewing angle
        for part in active_view.root_parts:
            part.update(self.view_theta)

        # quick hack to implement blinking
        if True:
            if 'LEye' in active_view.name_to_part.keys() and 'REye' in active_view.name_to_part.keys():
                t = time.time() % 4.5
                if 1.0 < t < 1.3:
                    active_view.name_to_part['LEye'].set_scale([1, 0.1, 1])
                    active_view.name_to_part['REye'].set_scale([1, 0.1, 1])
                elif 2.5 < t < 2.7:
                    active_view.name_to_part['LEye'].set_scale([1, 0.1, 1])
                    active_view.name_to_part['REye'].set_scale([1, 0.1, 1])
                elif 2.9 < t < 3.2:
                    active_view.name_to_part['LEye'].set_scale([1, 0.1, 1])
                    active_view.name_to_part['REye'].set_scale([1, 0.1, 1])
                else:
                    active_view.name_to_part['LEye'].set_scale([1, 1, 1])
                    active_view.name_to_part['REye'].set_scale([1, 1, 1])

        # set the mesh triangle rendering order
        active_mesh.process_motion_source_joint_depths(active_view.retargeter.get_destination_joint_render_order(self.ms_joint_names, self.ms_joint_positions, self.view_theta))

        # set the active texture name
        active_txtr_name = active_view.retargeter.get_side_facing_viewer(self.view_theta)
        active_mesh.set_active_txtr_name(active_txtr_name)

        # set plane rotation
        v = self.viewer.update_and_get_world_position() - self.update_and_get_world_position()  # vector from AD to viewer
        self.plane_rotation = np.arctan2(v[0], v[2])

        # set_translation
        root_offset_scale = active_view.retargeter.get_root_offset_scaling_factor(self.ms_joint_names, self.ms_joint_positions)
        self.offset(root_offset_scale * self.ms_root_offset)


class ServerAnimatedDrawingView(Transform):

    def __init__(self, char_view_cfg: CharacterViewConfig):
        print('starting')
        super().__init__()

        self.char_view_cfg: CharacterViewConfig = char_view_cfg

        self.rig: AnimatedDrawingRig = AnimatedDrawingRig.get_rig_from_char_cfg(self.char_view_cfg)

        # materials needed to generate textured mesh
        self.mask: npt.NDArray[np.uint8] = ad_utils.load_mask(self.char_view_cfg.mask_p, self.char_view_cfg.img_dim, self.char_view_cfg.img_height, self.char_view_cfg.img_width)
        self.txtrs: Dict[str, npt.NDArray[np.uint8]] = self._load_txtrs()

        # generate the meshes
        self.meshes: Dict[str, AnimatedDrawingMesh] = self._create_meshes()

        self.active_mesh_name = 'footleft-footright'
        self.active_txtr_name = 'front'

        for mesh in self.meshes.values():
            self.add_child(mesh)

        self.add_child(self.rig)

        # position so root joint is at origin
        root_x, root_y, root_z = self.rig.root_joint.update_and_get_world_position()
        self.rig.root_joint.offset(-np.array([root_x, root_y, root_z]))

        # move mesh vertices by same amount
        for mesh in self.meshes.values():
            mesh.vertices[:, :3] += -np.array([root_x, root_y, root_z])

        # generate the parts
        # TODO: Make compatable with characters that don't have parts
        parts_p = self.char_view_cfg.txtr_p.parent / 'parts_info.yaml'
        self.root_parts, self.name_to_part = AnimatedDrawingInternalPart.generate_parts(f'{parts_p}', self.char_view_cfg, self)

        # # Add the submesh base parts to txtr dictionary
        # base_mesh_parts = AnimatedDrawingInternalPart.generate_base_mesh_names_and_masks(f'{parts_p}', self.char_view_cfg, self.name_to_part)
        # for key, val in base_mesh_parts.items():
        #     assert key not in self.txtrs.keys(), f'invalid base mesh part name: {key}'
        #     self.txtrs[key] = val

        for part in self.root_parts:
            self.add_child(part)

        # move parts so they have proper relation to root joint
        for part in self.root_parts:
            part.offset(-np.array([root_x, root_y, root_z]))

        for part in self.name_to_part.values():
            # if it translates around, it needs mesh attachment
            if part.does_rdtwp:
                part.set_attachment_mesh_and_compute_barycentric_coordinates()

        # set up mesh deformer with the initial constraint handle positions
        for mesh in self.meshes.values():
            mesh.initialize_mesh_deformer(self.rig.get_joint_positions())

        # initialize the retargeter
        self.retargeter: RetargeterTwistedPerspective2D = RetargeterTwistedPerspective2D(self.rig)
        self.retargeter.set_viewer(Transform())

        # set starting position of character so mesh is above groundplane
        mesh_min = 999
        for mesh in self.meshes.values():
            mesh_min = min(mesh_min, min(mesh.vertices[:, 1]))
        self.root_groundplane_offset = mesh_min

        self.rig.root_joint.offset(-np.array([0, mesh_min, 0]))
        self.rig.update_transforms()

    def get_plane_rotation_for_client(self):
        return self.retargeter.get_plane_rotation()

    def _create_meshes(self) -> Dict[str, AnimatedDrawingMesh]:
        # this function should be replaced with something that takes in masks and create a mesh per mask or something.

        # TODO: This function displeases me. Replace 'left'-'right' with bool like is_img_left or something.
        def str_flip(input):
            if input == 'right':
                return 'left'
            if input == 'left':
                return 'right'
            assert False
        if self.char_view_cfg.mesh_type != 'cardboard':
            raise NotImplementedError

        meshes: Dict[str, AnimatedDrawingMesh] = {}

        if not self.char_view_cfg.fo_right and not self.char_view_cfg.fo_left:
            meshes['footleft-footright'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=[])
            meshes['footright-footright'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=[])
            meshes['footright-footleft'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=[])
            meshes['footleft-footleft'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=[])
        else:
            fo_right = self.char_view_cfg.fo_right  # left or right
            fo_left = self.char_view_cfg.fo_left
            # TODO Speed this up so the solve is only done once, and we copy/flip each foot to create other versions
            meshes[f'foot{fo_left}-foot{fo_right}'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=[])
            meshes[f'foot{str_flip(fo_left)}-foot{fo_right}'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=['right_foot'])
            meshes[f'foot{str_flip(fo_left)}-foot{str_flip(fo_right)}'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=['right_foot', 'left_foot'])
            meshes[f'foot{fo_left}-foot{str_flip(fo_right)}'] = AnimatedDrawingMeshCardboardExternal(self.mask, self.txtrs, self.rig, self.char_view_cfg, flip=['left_foot'])

        return meshes

    def _load_txtrs(self) -> Dict[str, npt.NDArray[np.uint8]]:

        ret: Dict[str, npt.NDArray[np.uint8]] = {}

        # original texture
        txtr_orig_p = self.char_view_cfg.txtr_p.parent / ('texture_original.png')
        assert txtr_orig_p.exists(), f"Missing original texture: {txtr_orig_p}"
        ret['original'] = ad_utils.load_txtr(txtr_orig_p, self.char_view_cfg.img_dim, self.char_view_cfg.img_height, self.char_view_cfg.img_width)

        # front texture
        txtr_front_p = self.char_view_cfg.txtr_p.parent / ('texture_front.png')
        if not txtr_front_p.exists():
            print(f'{txtr_front_p} DNE. Using {txtr_orig_p} instead')
            txtr_front_p = txtr_orig_p
        ret['front'] = ad_utils.load_txtr(txtr_front_p, self.char_view_cfg.img_dim, self.char_view_cfg.img_height, self.char_view_cfg.img_width)

        # back texture
        txtr_back_p = self.char_view_cfg.txtr_p.parent / ('texture_back.png')
        if not txtr_back_p.exists():
            print(f'{txtr_back_p} DNE. Using {txtr_front_p} instead')
            txtr_back_p = self.char_view_cfg.txtr_p
        ret['back'] = ad_utils.load_txtr(txtr_back_p, self.char_view_cfg.img_dim, self.char_view_cfg.img_height, self.char_view_cfg.img_width)

        return ret

    def set_viewer(self, viewer: Transform) -> None:
        """ Changes the transform representing viewer position for view-dependend representation"""
        self.retargeter.set_viewer(viewer)

    def get_viewer_world_position(self) -> npt.NDArray:
        return self.retargeter.get_viewer_position()
