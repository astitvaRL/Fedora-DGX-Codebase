from animated_drawings.model.time_manager import TimeManager
from animated_drawings.model.renderable import Renderable
from animated_drawings.model.transform import Transform
from animated_drawings.model.quaternions import Quaternions
from animated_drawings.model.vectors import Vectors
from animated_drawings.model.motion_source import MotionSource
from animated_drawings.model.animated_drawing_mesh import AnimatedDrawingMeshClient
from animated_drawings.model.animated_drawing_client_server_interface import DevelopmentServerInterface, PyTelepathyServerInterface
import numpy as np
import numpy.typing as npt
import json
import yaml


class AnimatedDrawing(Transform, TimeManager, Renderable):

    def __init__(self, char_cfg_fn: str, retarget_cfg_fn: str, viewer: Transform):
        super().__init__()

        self.viewer: Transform = viewer

        self.motion_source: MotionSource = None

        motion_source_retarget_json = self.create_motion_source_retarget_json_from_yaml_fn(retarget_cfg_fn)

        if True:  # change to True during server development
            self.server_interface = DevelopmentServerInterface(char_cfg_fn)
            server_meshes = self.server_interface.clip_data_call()
            self.server_interface.register_motion_source_call(motion_source_retarget_json)
        else:
            self.server_interface = PyTelepathyServerInterface()
            clip_response = self.server_interface.clip_data_call(char_cfg_fn, retarget_cfg_fn)
            self.server_interface.register_motion_source_call(motion_source_retarget_json)
            server_meshes = json.loads(clip_response)

        # create meshes
        self.meshes = {}
        for mesh_ in server_meshes['meshes']:
            xy = np.array(mesh_['xy']).reshape(-1, 2).tolist()
            uv = np.array(mesh_['uv']).reshape(-1, 2).tolist()
            normal = np.array(mesh_['normal']).reshape(-1, 3).tolist()
            triangles = np.array(mesh_['triangles']).reshape(-1, 3).tolist()

            mesh = AnimatedDrawingMeshClient(
                vertices_xy=xy,
                vertices_uv=uv,
                vertices_normal=normal,
                triangles=triangles,
                submesh_names=mesh_['submesh_names'],
                submesh_starting_indices=mesh_['submesh_starting_indices'],
                txtr_names=mesh_['txtr_names'],
                txtrs=mesh_['txtrs'],
                hide_outside_of=mesh_['hide_outside_of']
            )

            self.meshes[mesh_["name"]] = mesh
            self.add_child(mesh)

        self.render_order = []

        self.ms_root_position_previous_frame = None

    def update(self):

        # get data that must be passed to server
        viewer_position = self.viewer.update_and_get_world_position()

        character_position = self.get_world_position()

        ms_joint_names = self.motion_source.bvh.get_joint_names()

        ms_joint_positions = np.array(self.motion_source.bvh.root_joint.get_chain_worldspace_positions()).reshape([-1, 3])

        ms_forward_vector = self.motion_source.get_fwd_vector()

        # get motion source root offset
        if self.ms_root_position_previous_frame is None:  # if not set yet
            self.ms_root_position_previous_frame = self.motion_source.bvh.root_joint.get_world_position()  # then use current value
        ms_root_offset = self.motion_source.bvh.root_joint.get_world_position() - self.ms_root_position_previous_frame  # compute the offset
        self.ms_root_position_previous_frame = self.motion_source.bvh.root_joint.get_world_position()  # update previous_frame val for use next time

        # query server for response
        frame_call_response = self.server_interface.frame_data_call(viewer_position, character_position, ms_joint_names, ms_joint_positions, ms_forward_vector, ms_root_offset)
        frame_call_response = json.loads(frame_call_response)

        # process the response
        theta = float(frame_call_response['theta'])

        position = np.array(json.loads(frame_call_response['position']))

        meshname_updated_xy_list = frame_call_response['meshname_updated_xy_list']
        for item in meshname_updated_xy_list:
            item['xy'] = np.array(item['xy'])

        meshname_renderorders = frame_call_response['meshes_renderorders']
        # for idx in range(len(meshname_renderorders)):
        #     if meshname_renderorders[idx]['indices'] is None:
        #         continue
        #     meshname_renderorders[idx]['indices'] = np.array(json.loads(meshname_renderorders[idx]['indices']))

        meshname_active_texturename_list = frame_call_response['meshname_active_texturename_list']

        # update the rotation of character plane
        y_axis: Vectors = Vectors([0, 1, 0])
        self.set_rotation(Quaternions.from_angle_axis(np.array([theta]), y_axis))

        # update the character position
        self.set_position(position)

        # update vertex xy positions
        for item in meshname_updated_xy_list:
            mesh_name, vert_xys = item['name'], item['xy']

            self.meshes[mesh_name].vertices[:, :2] = vert_xys.reshape([-1, 2])
            self.meshes[mesh_name]._rebuffer_vertex_data()

        # get active textures for each mesh
        for item in meshname_active_texturename_list:
            self.meshes[item['name']].active_txtr_name = item['txtr']

        # update render order
        self.render_order = meshname_renderorders

        self.update_transforms()

    def draw(self, **kwargs):

        for mesh_submesh_dict in self.render_order:
            if mesh_submesh_dict['name'] not in self.meshes.keys():
                continue
            mesh = self.meshes[mesh_submesh_dict['name']]
            submesh_name = mesh_submesh_dict['submesh']

            mesh.draw(submesh_name=submesh_name, **kwargs)

    def set_motion_source(self, motion_source: MotionSource):
        self.motion_source = motion_source

    def get_viewer_world_position(self) -> npt.NDArray:
        return self.viewer.update_and_get_world_position()

    def create_motion_source_retarget_json_from_yaml_fn(self, retarget_cfg_fn) -> dict:

        # reads in the yaml, and constructs new json object from the relevant data fields in yaml
        with open(str(retarget_cfg_fn), 'r') as f:
            retarget_yaml = yaml.load(f, Loader=yaml.FullLoader)
        motion_source_retarget_json = {}

        # mapping from motion source depth driver joint to character joints
        motion_source_retarget_json['ms_depth_driver_to_char_joints'] = {}
        for item in retarget_yaml['char_bodypart_groups']:
            bvh_depth_driver = item['bvh_depth_drivers'][0]
            char_joints = item['char_joints']
            motion_source_retarget_json['ms_depth_driver_to_char_joints'][bvh_depth_driver] = char_joints

        # joint chains of the character and the motion source used to determine root offset scaling
        motion_source_retarget_json['char_ms_root_offset_scaling_joint_chains'] = {}
        motion_source_retarget_json['char_ms_root_offset_scaling_joint_chains']['ms_joint_chains'] = retarget_yaml['char_bvh_root_offset']['bvh_joints']
        motion_source_retarget_json['char_ms_root_offset_scaling_joint_chains']['char_joint_chains'] = retarget_yaml['char_bvh_root_offset']['char_joints']

        # mappings from character joint to motion source joints for determing orientations
        motion_source_retarget_json['char_joint_to_ms_joints_orientation_mapping'] = {}
        for key, val in retarget_yaml['char_joint_bvh_joints_mapping'].items():
            motion_source_retarget_json['char_joint_to_ms_joints_orientation_mapping'][key] = val

        # a runtime check to ensure the character's head doesn't get flipped by retargeting. @Nicky the JSON passed by unity to server can just hardcode these values for now.
        motion_source_retarget_json['char_runtime_checks'] = retarget_yaml['char_runtime_checks']

        return motion_source_retarget_json
