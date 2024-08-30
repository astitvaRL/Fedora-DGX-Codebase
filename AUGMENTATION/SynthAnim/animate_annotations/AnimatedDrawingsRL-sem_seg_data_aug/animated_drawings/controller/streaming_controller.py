
from animated_drawings.controller.controller import Controller
from animated_drawings.config import ControllerConfig
from animated_drawings.model.scene import Scene
from animated_drawings.model.transform import Transform
from animated_drawings.model.animated_drawing import AnimatedDrawing
from animated_drawings.view.view import View
from animated_drawings.view.streaming_view import StreamingView
import animated_drawings.streaming.pytelepathy as pytelepathy
import numpy as np
import json
from typing import List, Dict
import logging
import asyncio


class StreamingController(Controller):
    def __init__(self, cfg: ControllerConfig, scene: Scene, view: View) -> None:
        super().__init__(cfg, scene)
        self.view: View = view

        self.scene_frames_stream: List[Dict]
        self._initialize_streaming("::", 8888)

        self.animated_drawings: List[AnimatedDrawing] = []
        self._set_animated_drawings()

    def _set_animated_drawings(self) -> None:

        for child in self.scene.get_children():
            if not isinstance(child, AnimatedDrawing):
                continue

            child.set_viewer(Transform())  # since per-character view positions are being streamed to AD, we create a new empty transform to serve as 'viewer' for each character

            self.animated_drawings.append(child)

    def _initialize_streaming(self, host, port, num_max_threads=4, reconnect_interval_s=5) -> None:
        config = pytelepathy.WebSocketServerConfig()
        config.host = host
        config.port = port
        config.num_max_threads = 1
        config.reconnect_interval_s = 5

        # ResChannel configuration
        clip_res_channel = pytelepathy.ResChannelConfig()
        clip_res_channel.path = "/clip_data"
        clip_res_channel.callback = self.clip_data_res_channel_callback
        config.res_channels.append(clip_res_channel)

        frame_res_channel = pytelepathy.ResChannelConfig()
        frame_res_channel.path = "/frame_data"
        frame_res_channel.callback = self.frame_data_res_channel_callback
        config.res_channels.append(frame_res_channel)

        logging.info("Creating server...")
        self.server = pytelepathy.WebSocketServer(config)
        self.server.start()
        logging.info("Server started.")

    def clip_data_res_channel_callback(self, request_msg: str) -> str:
        print(f"/clip_data({len(request_msg)} bytes): {request_msg[:16]}...")
        # TODO See if need better design for the request msg
        if request_msg != "QueryClipData":
            return "InvalidRequest"

        if not isinstance(self.view, StreamingView):
            return {}

        try:
            return self.view.get_clip_level_data(self.scene)
        except Exception as e:
            logging.error(f'{e}')
            return "Failed to get clip data"

    def is_valid_frame_req(self, request_json: json) -> bool:
        required_keys = ["Timestamp", "ViewPos", "ViewPosCharMapping", "JointPos", "JointPosCharMapping", "CharToReturn"]
        all_keys_exist = all(key in request_json for key in required_keys)
        if not all_keys_exist:
            logging.warn("Not all keys exist in the frame request message.")
            logging.debug(request_json)
            return False

        is_valid = True
        for key in required_keys:
            if not request_json[key]:
                logging.warn(f"Found empty property: {key}")
                is_valid = False

        if not is_valid:
            logging.warn(request_json)
        return is_valid

    def frame_data_res_channel_callback(self, request_msg: str) -> str:
        print(f"/frame_data({len(request_msg)} bytes): {request_msg[:16]}...")
        request_json = json.loads(request_msg)
        if not self.is_valid_frame_req(request_json):
            logging.warn("Invalid frame received.")
            return ""

        self._update_view_positions(request_json)
        # TODO: uncomment camera lookat
        # self.view.camera.look_at(-np.array(request_json["CameraLookAt"]))  # camera renders out of its 'back', hence reversing the vector here  # This may no longer make sense to keep

        self._update_motion_source_joint_positions(request_json)

        self._update(request_json)

        if not isinstance(self.view, StreamingView):
            return ""

        try:
            assert len(request_json['CharToReturn']) == len(self.animated_drawings), 'len(CharToReturn) and len(self.animated_drawings) not equal'
            return self.view.get_frame_level_data(self.scene, request_json['CharToReturn'])
        except Exception as e:
            logging.error(f'{e}')
            return ""

    def _update_view_positions(self, request_json: Dict) -> None:
        if not len(request_json["ViewPosCharMapping"]):
            return

        try:
            # ensure ViewPosCharMapping doesn't refer ViewPos that don't exist
            assert len(request_json["ViewPos"]) > max([x[1] for x in request_json["ViewPosCharMapping"]]), 'ViewPosCharMapping specified ViewPosID that is too large'
            assert 0 <= min([x[1] for x in request_json["ViewPosCharMapping"]]), 'ViewPosCharMapping specified ViewPosID that is too small'

            # ensure ViewPosCharMapping doesn't refer to CharacterIDs that don't exist
            assert len(self.animated_drawings) > max([x[0] for x in request_json["ViewPosCharMapping"]]), 'ViewPosCharMapping specified CharacterID that is too large'
            assert 0 <= min([x[0] for x in request_json["ViewPosCharMapping"]]), 'ViewPosCharMapping specified CharacterID that is too small'
        except Exception as e:
            logging.error(e)
            raise e

        for character_id, viewpos_id in request_json["ViewPosCharMapping"]:
            self.animated_drawings[character_id].set_viewer_position(np.array(request_json["ViewPos"][viewpos_id]).copy())

    def _update_motion_source_joint_positions(self, request_json: Dict) -> None:
        if not len(request_json["JointPosCharMapping"]):
            return

        try:
            # ensure JointPosCharMapping doesn't refer ViewPos that don't exist
            assert len(request_json["JointPos"]) > max([x[1] for x in request_json["JointPosCharMapping"]]), 'JointPosCharMapping specified JointPosID that is too large'
            assert 0 <= min([x[1] for x in request_json["JointPosCharMapping"]]), 'JointPosCharMapping specified JointPosID that is too small'

            # ensure JointPosCharMapping doesn't refer to CharacterIDs that don't exist
            assert len(self.animated_drawings) > max([x[0] for x in request_json["JointPosCharMapping"]]), 'JointPosCharMapping specified CharacterID that is too large'
            assert 0 <= min([x[0] for x in request_json["JointPosCharMapping"]]), 'JointPosCharMapping specified CharacterID that is too small'
        except Exception as e:
            logging.error(e)
            raise e

        for character_id, jointpos_id in request_json["JointPosCharMapping"]:
            self.animated_drawings[character_id].retargeter.motion_source.set_joint_positions(np.array(request_json['JointPos'][jointpos_id]).reshape([-1, 3]).copy())

    def _update(self, request_json: Dict) -> None:

        try:
            assert len(request_json['CharToReturn']) == len(self.animated_drawings), 'length of CharToReturn not equal to number of animated_drawings in scene'
        except Exception as e:
            logging.error(e)
            raise e

        self.scene.update_transforms()

        # only update the characters we will be returning this frame
        animated_drawings_in_scene = [child for child in self.scene.get_children() if isinstance(child, AnimatedDrawing)]
        for is_returned, ad in zip(request_json['CharToReturn'], animated_drawings_in_scene):
            if is_returned:
                ad.update()

    async def run(self):
        """ Don't need to call a real run loop when using streaming controller. """
        while True:
            await asyncio.sleep(1)
