# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations
import logging
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Tuple, Dict, TypedDict, Optional
import yaml
from pkg_resources import resource_filename
from animated_drawings.utils import resolve_ad_filepath


class Config():

    def __init__(self, user_mvc_cfg_fn: str) -> None:
        # get the base mvc config
        with open(resource_filename(__name__, "mvc_base_cfg.yaml"), 'r') as f:
            base_cfg = defaultdict(dict, yaml.load(f, Loader=yaml.FullLoader) or {})  # pyright: ignore[reportUnknownMemberType])

        # search for the user-specified mvc confing
        user_mvc_cfg_p: Path = resolve_ad_filepath(user_mvc_cfg_fn, 'user mvc config')
        logging.info(f'Using user-specified mvc config file located at {user_mvc_cfg_p.resolve()}')
        with open(str(user_mvc_cfg_p), 'r') as f:
            user_cfg = defaultdict(dict, yaml.load(f, Loader=yaml.FullLoader) or {})  # pyright: ignore[reportUnknownMemberType]

        # overlay user specified mvc options onto base mvc, use to generate subconfig classes
        self.view: ViewConfig = ViewConfig({**base_cfg['view'], **user_cfg['view']})
        self.scene: SceneConfig = SceneConfig({**base_cfg['scene'], **user_cfg['scene']}, user_mvc_cfg_p)
        self.controller: ControllerConfig = ControllerConfig({**base_cfg['controller'], **user_cfg['controller']})

        # cannot use an interactive controller with a headless mesa viewer
        if self.controller.mode == 'interact':
            try:
                assert self.view.view_type == 'window', 'can only use interactive controller with "window" VIEW_TYPE'
            except AssertionError as e:
                msg = f'Config error: {e}'
                logging.critical(msg)
                assert False, msg

        # can only use export_view and export_controller together
        if self.controller.mode == 'export' or self.view.view_type == 'export':
            try:
                assert self.controller.mode == 'export' and self.view.view_type == 'export', 'export controller and export view must be used together'
            except AssertionError as e:
                msg = f'Config error: {e}'
                logging.critical(msg)
                assert False, msg

        # output video path must be set for render controller
        if self.controller.mode == 'video_render':
            try:
                assert self.controller.output_video_path is not None, 'output_video_path must be set when using video_render controller'
            except AssertionError as e:
                msg = f'Config error: {e}'
                logging.critical(msg)
                assert False, msg

        # output video codec must be set for render controller with .mp4 output filetype
        if self.controller.mode == 'video_render' and self.controller.output_video_path is not None and self.controller.output_video_path.endswith('.mp4'):
            try:
                assert self.controller.output_video_codec is not None, 'output_video_codec must be set when using video_render controller'
            except AssertionError as e:
                msg = f'Config error: {e}'
                logging.critical(msg)
                assert False, msg


class SceneConfig():

    def __init__(self, scene_cfg: dict, user_mvc_cfg_p: Path) -> None:

        # show or hide visualization of BVH motion driving characters
        try:
            self.add_ad_retarget_bvh: bool = scene_cfg['ADD_AD_RETARGET_BVH']
            assert isinstance(self.add_ad_retarget_bvh, bool), 'is not bool'
        except (AssertionError, ValueError) as e:
            msg = f'Error in ADD_AD_RETARGET_BVH config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # config files for characters, driving motions, and retargeting
        self.animated_characters: List[Tuple[str, str, MotionConfig]] = []

        each: Dict[str, str]
        for each in scene_cfg['ANIMATED_CHARACTERS']:

            char_cfg_p = resolve_ad_filepath(each['character_cfg'], 'character config')
            assert char_cfg_p.exists(), f'could not locate {char_cfg_p}'

            motion_cfg_p = resolve_ad_filepath(each['motion_cfg'], 'motion config')
            assert motion_cfg_p.exists(), f'could not locate {motion_cfg_p}'

            retarget_cfg_p = resolve_ad_filepath(each['retarget_cfg'], 'retarget config')
            assert retarget_cfg_p.exists(), f'could not locate {retarget_cfg_p}'

            self.animated_characters.append((
                str(char_cfg_p),
                str(retarget_cfg_p),
                MotionConfig(str(motion_cfg_p))
            ))


class ViewConfig():

    def __init__(self, view_cfg: dict) -> None:  # noqa: C901

        # set color used to clear render buffer
        try:
            self.clear_color: list[Union[float, int]] = view_cfg["CLEAR_COLOR"]
            assert len(self.clear_color) == 4, 'length not four'
            for val in self.clear_color:
                assert isinstance(val, (float, int)), f'{val} not float or int'
                assert val <= 1.0, 'values must be <= 1.0'
                assert val >= 0.0, 'values must be >= 0.0'
        except (AssertionError, ValueError) as e:
            msg = f'Error in CLEAR_COLOR config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set an image to use for the background, if desired
        try:
            self.background_image_path: Union[None, str] = resolve_ad_filepath(view_cfg["BACKGROUND_IMAGE_PATH"], 'background_image')
            assert Path(self.background_image_path).exists(), f'BACKGROUND_IMAGE_PATH does not exist {self.background_image_path}'
            self.draw_background_image = True
        except AssertionError as e:
            self.draw_background_image = False
            self.background_image_path = None
            msg = f'BACKGROUND_IMAGE_PATH DNE: {e}'
            logging.info(msg)
        except ValueError as e:
            msg = f'Error in BACKGROUND_IMAGE_PATH: {e}'
            logging.critical(msg)
            assert False, msg

        # set the dimensions of the window or output video
        try:
            self.window_dimensions: tuple[int, int] = view_cfg["WINDOW_DIMENSIONS"]
            assert len(self.window_dimensions) == 2, 'length is not 2'
            for val in self.window_dimensions:
                assert val > 0, f'{val} must be > 0'
                assert isinstance(val, int), 'type not int'
        except (AssertionError, ValueError) as e:
            msg = f'Error in WINDOW_DIMENSIONS config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set whether we want the character rigs to be visible
        try:
            self.draw_ad_rig: bool = view_cfg['DRAW_AD_RIG']
            assert isinstance(self.draw_ad_rig, bool), 'value is not bool type'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_AD_RIG config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set whether we want the character textures to be visible
        try:
            self.draw_ad_txtr: bool = view_cfg['DRAW_AD_TXTR']
            assert isinstance(self.draw_ad_txtr, bool), 'value is not bool type'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_AD_TXTR config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set whether we want the character triangle->bone assignment colors to be visible
        try:
            self.draw_ad_color: bool = view_cfg['DRAW_AD_COLOR']
            assert isinstance(self.draw_ad_color, bool), 'value is not bool type'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_AD_COLOR config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # initialize character normal visualization
        try:
            self.draw_ad_normal: bool = view_cfg['DRAW_AD_NORMAL']
            assert isinstance(self.draw_ad_normal, bool), 'value is not bool type'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_AD_NORMAL config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set whether we want the character mesh lines to be visible
        try:
            self.draw_ad_mesh_lines: bool = view_cfg['DRAW_AD_MESH_LINES']
            assert isinstance(self.draw_ad_mesh_lines, bool), 'value is not bool type'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_AD_MESH_LINES config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # show or hide character reflection
        try:
            self.draw_character_reflection: bool = view_cfg['DRAW_CHARACTER_REFLECTION']
            assert isinstance(self.draw_character_reflection, bool), 'is not bool'
        except (AssertionError, ValueError) as e:
            msg = f'Error in SHOW_CHARACTER_REFLECTION config parameter: {e}'
            logging.critical(msg)
            assert False, msg        # show or hide character reflection

        try:
            self.draw_shadows: bool = view_cfg['DRAW_SHADOWS']
            assert isinstance(self.draw_shadows, bool), 'is not bool'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_SHADOWS config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # show or hide bvh
        try:
            self.draw_bvh: bool = view_cfg['DRAW_BVH']
            assert isinstance(self.draw_bvh, bool), 'is not bool'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_BVH config parameter: {e}'
            logging.critical(msg)
            assert False, msg
        except KeyError as e:
            msg = f'KeyError: {e}. Defaulting to True'
            logging.info(msg)
            self.draw_bvh = True

        # show or hide floor
        try:
            self.draw_floor: bool = view_cfg['DRAW_FLOOR']
            assert isinstance(self.draw_floor, bool), 'is not bool'
        except (AssertionError, ValueError) as e:
            msg = f'Error in DRAW_FLOOR config parameter: {e}'
            logging.critical(msg)
            assert False, msg
        except KeyError as e:
            msg = f'KeyError: {e}. Defaulting to True'
            logging.info(msg)
            self.draw_floor = True

        # set whether we want to use mesa on the back end (necessary for headless rendering)
        try:
            self.view_type: str = view_cfg['VIEW_TYPE']
            assert self.view_type in ['window', 'mesa', 'export', 'streaming'], f'insupported value: {self.view_type}'
        except (AssertionError, ValueError) as e:
            msg = f'Error in VIEW_TYPE config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set the position of the view camera
        try:
            self.camera_pos: list[Union[float, int]] = view_cfg['CAMERA_POS']
            assert len(self.camera_pos) == 3, 'length != 3'
            for val in self.camera_pos:
                assert isinstance(val, (float, int)), f' {val} is not float or int'
        except (AssertionError, ValueError) as e:
            msg = f'Error in CAMERA_POS config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set the forward vector of the view camera (but it renders out of it's rear)
        try:
            self.camera_fwd: list[Union[float, int]] = view_cfg['CAMERA_FWD']
            assert len(self.camera_fwd) == 3, 'length != 3'
            for val in self.camera_fwd:
                assert isinstance(val, (float, int)), f' {val} is not float or int'
        except (AssertionError, ValueError) as e:
            msg = f'Error in CAMERA_FWD config parameter: {e}'
            logging.critical(msg)
            assert False, msg


class ControllerConfig():

    def __init__(self, controller_cfg: dict) -> None:

        # set controller mode
        try:
            self.mode: str = controller_cfg["MODE"]
            assert self.mode in ('interactive', 'video_render', 'export', 'streaming', 'image_render'), f'unsupported mode: {self.mode}'
        except (AssertionError, ValueError) as e:
            msg = f'Error in MODE config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set timestep for user interactions in interactive mode
        try:
            self.keyboard_timestep: Union[float, int] = controller_cfg["KEYBOARD_TIMESTEP"]
            assert isinstance(self.keyboard_timestep, (float, int)), 'is not floar or int'
            assert self.keyboard_timestep > 0, 'timestep val must be > 0'
        except (AssertionError, ValueError) as e:
            msg = f'Error in KEYBOARD_TIMESTEP config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set output video path (only use in video_render mode)
        try:
            self.output_video_path: Union[None, str] = controller_cfg['OUTPUT_VIDEO_PATH']
            assert isinstance(self.output_video_path, (NoneType, str)), 'type is not None or str'
            if isinstance(self.output_video_path, str):
                assert Path(self.output_video_path).suffix in ('.gif', '.mp4'), 'output video extension not .gif or .mp4 '
        except (AssertionError, ValueError) as e:
            msg = f'Error in OUTPUT_VIDEO_PATH config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set output image dir (only use in image_render mode)
        try:
            self.output_image_dir: Union[None, str] = controller_cfg['OUTPUT_IMAGE_DIR']
            assert isinstance(self.output_image_dir, (NoneType, str)), 'type is not None or str'
        except (AssertionError, ValueError) as e:
            msg = f'Error in OUTPUT_IMAGE_DIR config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # set output video codec (only use in video_render mode with .mp4)
        try:
            self.output_video_codec: Union[None, str] = controller_cfg['OUTPUT_VIDEO_CODEC']
            assert isinstance(self.output_video_codec, (NoneType, str)), 'type is not None or str'
        except (AssertionError, ValueError) as e:
            msg = f'Error in OUTPUT_VIDEO_CODEC config parameter: {e}'
            logging.critical(msg)
            assert False, msg


class MotionConfig():

    def __init__(self, motion_cfg_fn: str) -> None:  # noqa: C901
        motion_cfg_p = resolve_ad_filepath(motion_cfg_fn, 'motion cfg')
        with open(str(motion_cfg_p), 'r') as f:
            motion_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # validate start_frame_idx
        try:
            self.start_frame_idx: int = motion_cfg.get('start_frame_idx', 0)
            assert isinstance(self.start_frame_idx, int), 'type not int'
            assert self.start_frame_idx >= 0, 'start_frame_idx must be > 0'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating start_frame_idx: {e}'
            logging.critical(msg)
            assert False, msg

        # validate end_frame_idx
        try:
            self.end_frame_idx: Optional[int] = motion_cfg.get('end_frame_idx', None)
            assert isinstance(self.end_frame_idx, (NoneType, int)), 'type not NoneType or int'
            if isinstance(self.end_frame_idx, int):
                assert self.end_frame_idx >= self.start_frame_idx, 'end_frame_idx must be > start_frame_idx'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating end_frame_idx: {e}'
            logging.critical(msg)
            assert False, msg

        # validate groundplane joint
        try:
            self.groundplane_joint: str = motion_cfg['groundplane_joint']
            assert isinstance(self.groundplane_joint, str), 'groundplane joint must be str'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating groundplane joint: {e}'
            logging.critical(msg)
            assert False, msg

        # validate forward_perp_joint_vectors
        try:
            self.forward_perp_joint_vectors: List[Tuple[str, str]] = motion_cfg['forward_perp_joint_vectors']
            assert len(self.forward_perp_joint_vectors) > 0, 'forward_perp_joint_vectors len must be > 0'
            for each in self.forward_perp_joint_vectors:
                assert len(each) == 2, 'each list in forrward_perp_joint_vectors must have len = 2'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating forward_perp_joint_vectors: {e}'
            logging.critical(msg)
            assert False, msg

        # validate scale
        try:
            self.scale: float = motion_cfg['scale']
            assert isinstance(self.scale, (int, float)), 'scale must be float or int'
            assert self.scale > 0, 'scale must be > 0'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating scale: {e}'
            logging.critical(msg)
            assert False, msg

        # validate up
        try:
            self.up: str = motion_cfg['up']
            assert self.up in ['+y', '+z'], 'up must be "+y" or "+z'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating up: {e}'
            logging.critical(msg)
            assert False, msg

        # validate source_type
        try:
            self.source_type: str = motion_cfg['source_type']
            assert self.source_type in ['file_bvh', 'stream_bvh', 'file_jointpositionarray', 'stream_jointpositionarray']
        except KeyError as e:
            msg = f'No source_type specified, default to type `file`: {e}'
            logging.info(msg)
            self.source_type = 'file_bvh'
        except AssertionError as e:
            msg = f'Error validating source_type: {e}'
            logging.critical(msg)
            assert False, msg

        # validate file inputs
        if self.source_type.startswith('file'):

            # validate file exists and has correct extension
            try:
                self.ms_p: Path = resolve_ad_filepath(motion_cfg['filepath'], 'bvh filepath')  # motion source path
                if not self.ms_p.exists():
                    self.ms_p = Path(motion_cfg['filepath'])
                    if not self.ms_p.exists():
                        self.ms_p = Path(motion_cfg_fn).parent / self.ms_p
                    assert self.ms_p.exists(), f'could not locate {self.ms_p}'

                assert self.ms_p.exists(), f'specified filepath DNE: {self.ms_p}'

                if self.source_type.endswith('jointpositionarray'):
                    assert self.ms_p.suffix == '.npy', 'source_type is file_jointpositionarray, but file suffix is not .npy'
                elif self.source_type.endswith('bvh'):
                    assert self.ms_p.suffix == '.bvh', 'source_type is file_bvh, but file suffix is not .bvh'
                else:
                    assert False, f'unsupported file extension: {self.ms_p.suffix}'

            except (AssertionError, ValueError) as e:
                msg = f'Error validating motionsource of type {self.source_type}: {e}'
                logging.critical(msg)
                assert False, msg
        else:
            try:
                assert motion_cfg.get('filepath', None) is None, 'non file_* motion_sources should not define filepath'
            except AssertionError as e:
                msg = f'Error validating motionsource of type {self.source_type}: {e}'
                logging.critical(msg)
                assert False, msg

        # validate stream inputs
        pass

        # validate *_jointpositionarray source
        if self.source_type.endswith('jointpositionarray'):
            msg_preamble = 'Error validating motionsource *jointpositionarray'

            # validate that jointpositionarray_jointnames is defined
            try:
                self.jointpositionarray_jointnames = motion_cfg.get('jointpositionarray_jointnames', None)
                assert self.jointpositionarray_jointnames is not None, 'file_jointpositionarray motion_sources must define jointpositionarray_jointnames'
            except AssertionError as e:
                msg = f'{msg_preamble}: {e}'
                logging.critical(msg)
                assert False, msg

            # validate that jointpositionarray_frametime is defined
            try:
                self.jointpositionarray_frametime = motion_cfg.get('jointpositionarray_frametime', None)
                assert self.jointpositionarray_frametime is not None, 'file_jointpositionarray motion_sources must define jointpositionarray_frametime'
                assert isinstance(self.jointpositionarray_frametime, (int, float)), 'jointpositionarray_frametime must be str'
            except AssertionError as e:
                msg = f'{msg_preamble}: {e}'
                logging.critical(msg)
                assert False, msg

            # validate that jointpositionarray_root_joint_name is defined
            try:
                self.jointpositionarray_root_joint_name = motion_cfg.get('jointpositionarray_root_joint_name', None)
                assert self.jointpositionarray_root_joint_name is not None, 'file_jointpositionarray motion_sources must define jointpositionarray_root_joint_name'
                assert isinstance(self.jointpositionarray_root_joint_name, str), 'jointpositionarray_root_joint_name must be str'
            except AssertionError as e:
                msg = f'{msg_preamble}: {e}'
                logging.critical(msg)
                assert False, msg

        else:  # to avoid confusion, ensure that jointpositionarray specific attributes are not present in other motion_source types

            try:
                assert motion_cfg.get('jointpositionarray_jointnames', None) is None, 'file_bvh motion_sources should not define jointpositionarray_jointnames'
                assert motion_cfg.get('jointpositionarray_frametime', None) is None, 'file_bvh motion_sources should not define jointpositionarray_frametime'
                assert motion_cfg.get('jointpositionarray_root_joint_name', None) is None, 'file_bvh motion_sources should not define jointpositionarray_root_joint_name'
            except AssertionError as e:
                msg = f'{msg_preamble}: {e}'
                logging.critical(msg)
                assert False, msg

        # validate *_bvh source
        pass

        # This is deprecated
        # # validate stream source
        # if self.source_type == 'stream':
        #     try:  # validate ip address
        #         self.stream_ipv4 = motion_cfg['stream_ipv4']
        #     except KeyError as e:
        #         msg = f'Error validating stream_ipv4: {e}'
        #         logging.critical(msg)
        #         assert False, msg

        #     try:  # validate  port
        #         self.stream_port = motion_cfg['stream_port']
        #     except KeyError as e:
        #         msg = f'Error validating stream_port: {e}'
        #         logging.critical(msg)
        #         assert False, msg

    def validate_joint_names(self, motionsource_jointnames: List[str]) -> None:
        """ Performs all the validation steps that depend upon knowing the BVH joint names. This should be called once the BVH had been leaded """

        # joints needed to compute forward vector
        try:
            for prox_joint_name, dist_joint_name in self.forward_perp_joint_vectors:
                assert prox_joint_name in motionsource_jointnames, f'invalid prox_joint name in motion_cfg.forward_perp_joint_vectors: {prox_joint_name}'
                assert dist_joint_name in motionsource_jointnames, f'invalid dist_joint name in motion_cfg.forward_perp_joint_vectors: {dist_joint_name}'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating forward_perp_joint_vector joints: {e}'
            logging.critical(msg)
            assert False, msg

        # groundplane joint
        try:
            assert self.groundplane_joint in motionsource_jointnames, f'{self.groundplane_joint} not valid motion source joint names'
        except AssertionError as e:
            msg = f'Error validating groundplane joint: {e}'
            logging.critical(msg)
            assert False, msg

NoneType = type(None)  # needed for type checking
