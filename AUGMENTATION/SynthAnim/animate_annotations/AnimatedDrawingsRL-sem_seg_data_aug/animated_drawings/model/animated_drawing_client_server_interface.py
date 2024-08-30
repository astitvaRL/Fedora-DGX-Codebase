from abc import ABC, abstractmethod

import numpy.typing as npt
from typing import List
import json
from pathlib import Path
import pickle

import animated_drawings.streaming.pytelepathy as pytelepathy
from ad3d_server.animated_drawing import ServerAnimatedDrawing


class ServerInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def clip_data_call(self):
        pass

    @abstractmethod
    def frame_data_call(self):
        pass

    @abstractmethod
    def register_motion_source_call(self):
        pass


class DevelopmentServerInterface(ServerInterface):

    def __init__(self, char_cfg_fn):

        cached_file_loc = Path(char_cfg_fn).parent / f"{char_cfg_fn.replace('/','')}.pickle"
        # first, check if cached version of character exists
        if not cached_file_loc.exists():
            # create the animated drawing
            self.server_animated_drawing = ServerAnimatedDrawing(
                char_cfg_fn,
            )
            # then save it
            with open(f'{cached_file_loc}', 'wb') as f:
                pickle.dump(self.server_animated_drawing, f)
        else:
            # load the animated drawing from the file
            with open(f'{cached_file_loc}', 'rb') as f:
                self.server_animated_drawing = pickle.load(f)

    def clip_data_call(self):
        meshes = self.server_animated_drawing.get_meshes_for_client()
        return meshes

    def frame_data_call(self, viewer_world_position, character_position, ms_joint_names, ms_joint_positions, ms_forward_vector, ms_root_offset):

        # things that are sent to server
        self.server_animated_drawing.set_viewer_position(viewer_world_position)  # set viewer position
        self.server_animated_drawing.set_position(character_position)  # set the character's position
        self.server_animated_drawing.set_motion_source_joint_positions(ms_joint_names, ms_joint_positions)  # set motion source joint positions
        self.server_animated_drawing.set_motion_source_forward_vector(ms_forward_vector)  # set motion source forward vector
        self.server_animated_drawing.set_position(character_position)  # set the position of the character
        self.server_animated_drawing.set_motion_source_root_offset(ms_root_offset)  # set motion source's root offset

        # character position
        # ms_root_joint_offset

        # update stuff on server
        self.server_animated_drawing.update()

        # things that are returned from server
        theta = self.server_animated_drawing.get_plane_rotation_for_client()
        position = self.server_animated_drawing.get_animated_drawing_translation_for_client()
        meshname_updated_xy_list = self.server_animated_drawing.get_mesh_vertex_xy_updates_for_client()
        meshname_active_texturename_list = self.server_animated_drawing.get_mesh_active_texture_names_for_client()
        meshes_renderorders = self.server_animated_drawing.get_meshes_renderorders_for_client()  # a list of (mesh_name, render_indices) in the order in which they should be rendered.

        message = {
            "theta": str(theta),
            "position": str(position.tolist()),
            "meshname_updated_xy_list": meshname_updated_xy_list,
            "meshname_active_texturename_list": meshname_active_texturename_list,
            "meshes_renderorders": meshes_renderorders
        }

        return json.dumps(message)

    def register_motion_source_call(self, motion_source_retarget_json):
        self.server_animated_drawing.register_motion_source(motion_source_retarget_json)


class PyTelepathyServerInterface(ServerInterface):

    def __init__(self):
        self.start()

    def start(self):
        # Print program name and instructions
        print("[Telepathy] Mock AD3D Client")
        print("Press Ctrl+C to exit the program at any time.")

        host = "127.0.0.1"
        port = 8888

        # Create the clip data request channel
        clip_req_channel = pytelepathy.ReqChannelConfig()
        clip_req_channel.path = "/clip_data"

        clip_req_config = pytelepathy.WebSocketClientConfig()
        clip_req_config.host = host
        clip_req_config.port = port
        clip_req_config.max_read_limit_bytes = 1024 * 1024 * 1024
        clip_req_config.max_write_limit_bytes = 1024 * 1024 * 1024
        clip_req_config.channel_config = clip_req_channel

        self.clip_req_client = pytelepathy.WebSocketClient(clip_req_config)
        self.clip_req_client.start()

        # Create the frame data request channel
        frame_req_channel = pytelepathy.ReqChannelConfig()
        frame_req_channel.path = "/frame_data"

        frame_req_config = pytelepathy.WebSocketClientConfig()
        frame_req_config.host = host
        frame_req_config.port = port
        frame_req_config.channel_config = frame_req_channel

        self.frame_req_client = pytelepathy.WebSocketClient(frame_req_config)
        self.frame_req_client.start()

        # Create the register_motion_source
        register_motion_source_req_channel = pytelepathy.ReqChannelConfig()
        register_motion_source_req_channel.path = "/register_motion_source_data"

        register_motion_source_req_config = pytelepathy.WebSocketClientConfig()
        register_motion_source_req_config.host = host
        register_motion_source_req_config.port = port
        register_motion_source_req_config.channel_config = register_motion_source_req_channel

        self.register_motion_source_req_client = pytelepathy.WebSocketClient(register_motion_source_req_config)
        self.register_motion_source_req_client.start()

    clip_response_msg = None

    def clip_data_call(self, char_cfg_fn, retarget_cfg_fn):
        def response_function(response_msg_, success):
            global clip_response_msg
            clip_response_msg = response_msg_
        self.clip_req_client.request(f"{char_cfg_fn}", 1000, response_function)
        return clip_response_msg

    register_motion_source_response_msg = None

    def register_motion_source_call(self, motion_source_retarget_json):
        def response_function(response_msg_, success):
            global register_motion_source_response_msg
            register_motion_source_response_msg = response_msg_
        self.register_motion_source_req_client.request(json.dumps(motion_source_retarget_json), 1000, response_function)
        return register_motion_source_response_msg

    frame_response_msg = None

    def frame_data_call(
        self,
        viewer_position: npt.NDArray,
        character_position: npt.NDArray,
        ms_joint_names: List[str],
        ms_joint_positions: npt.NDArray,
        ms_forward_vector: npt.NDArray,
        ms_root_offset: npt.NDArray
    ):
        assert len(ms_joint_names) == ms_joint_positions.shape[0]

        viewer_position = viewer_position.tolist()
        character_position = character_position.tolist()
        ms_joint_positions = ms_joint_positions.flatten().tolist()
        ms_forward_vector = ms_forward_vector.tolist()
        ms_root_offset = ms_root_offset.tolist()

        def response_function(response_msg_, success):
            global frame_response_msg
            frame_response_msg = response_msg_

        request_json = {}
        request_json['viewer_position'] = viewer_position
        request_json['character_position'] = character_position
        request_json['ms_joint_names'] = ms_joint_names
        request_json['ms_joint_positions'] = ms_joint_positions
        request_json['ms_forward_vector'] = ms_forward_vector
        request_json['viewer_position'] = viewer_position
        request_json['ms_root_offset'] = ms_root_offset
        self.frame_req_client.request(json.dumps(request_json), 1000, response_function)

        return frame_response_msg
