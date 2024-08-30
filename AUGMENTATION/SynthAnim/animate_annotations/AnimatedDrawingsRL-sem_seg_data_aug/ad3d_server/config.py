import logging
import yaml
from typing import TypedDict, List, Union
from pathlib import Path
from pkg_resources import resource_filename


class CharacterConfig():

    def __init__(self, char_cfg_fn: str) -> None:
        """ char_cfg_fn is the path to a directory containing subdirectories 'left' and 'right' with char_cfg.yaml, parts_info.yaml, masks and texture subdirs """

        left_view_fn = Path(char_cfg_fn) / "left" / "char_cfg.yaml"
        self.left_view = CharacterViewConfig(left_view_fn)

        right_view_fn = Path(char_cfg_fn) / "right" / "char_cfg.yaml"
        self.right_view = CharacterViewConfig(right_view_fn)


class CharacterViewConfig():

    class JointDict(TypedDict):
        loc: List[float]
        name: str
        parent: Union[None, str]

    def __init__(self, char_cfg_fn: str) -> None:  # noqa: C901
        character_cfg_p = resolve_ad_filepath(char_cfg_fn, 'character cfg')
        with open(str(character_cfg_p), 'r') as f:
            char_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # validate image height
        try:
            self.img_height: int = char_cfg['height']
            assert isinstance(self.img_height, int), 'type not int'
            assert self.img_height > 0, 'must be > 0'
        except (AssertionError, ValueError) as e:
            msg = f'Error in character height config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # validate image width
        try:
            self.img_width: int = char_cfg['width']
            assert isinstance(self.img_width, int), 'type not int'
            assert self.img_width > 0, 'must be > 0'
        except (AssertionError, ValueError) as e:
            msg = f'Error in character width config parameter: {e}'
            logging.critical(msg)
            assert False, msg

        # based on height and width, determine what final img dimension will be (post padding)
        self.img_dim: int = max(self.img_height, self.img_width)

        # validate skeleton
        try:
            self.skeleton: List[CharacterConfig.JointDict] = []
            for joint in char_cfg['skeleton']:

                # ensure loc input is valid...
                loc: List[int] = joint['loc']
                assert len(loc) == 2, 'joint loc must be of length 2'
                assert loc[0] >= 0, 'x val must be >= 0'
                assert loc[0] < self.img_width, 'x val must be < image width'
                assert loc[1] >= 0, 'y val must be >= 0'
                assert loc[1] < self.img_height, 'y val must be < image height'

                # ... then scale to between 0-1 based on img dim
                loc_x: float = loc[0] / self.img_dim  # width
                loc_y: float = loc[1] / self.img_dim + (1 - self.img_height / self.img_dim)  # height

                # validate joint name
                name: str = joint['name']
                assert isinstance(name, str), 'name must be str'

                # validate joint parent
                parent: Union[None, str] = joint['parent']
                assert isinstance(parent, (NoneType, str)), 'parent must be str or NoneType'

                self.skeleton.append({'loc': [loc_x, loc_y], 'name': name, 'parent': parent})
        except AssertionError as e:
            msg = f'Error in character skeleton: {e}'
            logging.critical(msg)
            assert False, msg

        # validate skeleton joint parents
        try:
            names: List[str] = [joint['name'] for joint in self.skeleton]
            for joint in self.skeleton:
                assert isinstance(joint['parent'], NoneType) or joint['parent'] in names, f'joint.parent not None and not valid joint name: {joint}'
        except AssertionError as e:
            msg = f'Error in character skeleton: {e}'
            logging.critical(msg)
            assert False, msg

        # validate mask and texture files
        try:
            self.mask_p: Path = character_cfg_p.parent / 'mask.png'
            self.txtr_p: Path = character_cfg_p.parent / 'texture_original.png'
            assert self.mask_p.exists(), f'cannot find character mask: {self.mask_p}'
            assert self.txtr_p.exists(), f'cannot find character texture: {self.txtr_p}'
        except AssertionError as e:
            msg = f'Error validating character files: {e}'
            logging.critical(msg)
            assert False, msg

        # validate the mesh type
        try:
            self.mesh_type: str = char_cfg['mesh_type']
            assert self.mesh_type in ['2D', 'cardboard'], 'unsupported mesh_type'
        except KeyError:
            msg = 'No mesh_type specified. Defaulting to \'cardboard\''
            self.mesh_type = 'cardboard'
        except AssertionError as e:
            msg = f'AssertionError: {e}'
            logging.critical(msg)
            assert False, msg

        # validate the rig type
        try:
            self.rig_type: str = char_cfg['rig_type']
            assert self.rig_type in ['2D', '3D'], 'unsupported rig_type'
        except KeyError:
            msg = 'No rig_type specified. Defaulting to \'2D\''
            self.rig_type = '2D'
        except AssertionError as e:
            msg = f'AssertionError: {e}'
            logging.critical(msg)
            assert False, msg

        # validate deformer type
        try:
            self.deformer_type : str = char_cfg['deformer_type']
            assert self.deformer_type in ['arap_igarashi'], 'unsupported deformer_type'
        except KeyError:
            msg = 'No deformer_type specified. Defaulting to \'arap_igarashi\''
            self.deformer_type = 'arap_igarashi'
        except AssertionError as e:
            msg = f'AssertionError: {e}'
            logging.critical(msg)

        # validate character *image right* foot orientation
        try:
            self.fo_right = char_cfg['footorientation_right']
            assert self.fo_right in ['left', 'right', None]
        except KeyError:
            msg = 'No right foot orientation specified. defaulting to left'
            self.fo_right = 'left'
        except AssertionError as e:
            msg = f'AssertionError: {e}'
            logging.critical(msg)

        # validate character *image left* foot orientation
        try:
            self.fo_left = char_cfg['footorientation_left']
            assert self.fo_left in ['left', 'right', None]
        except KeyError:
            msg = 'No left foot orientation specified. defaulting to right'
            self.fo_left = 'right'
        except AssertionError as e:
            msg = f'AssertionError: {e}'
            logging.critical(msg)


class RetargetConfig():

    class BvhProjectionBodypartGroup(TypedDict):
        bvh_joint_names: List[str]
        method: str
        name: str

    class CharBodypartGroup(TypedDict):
        bvh_depth_drivers: List[str]
        char_joints: List[str]

    class CharBvhRootOffset(TypedDict):
        bvh_projection_bodypart_group_for_offset: str
        bvh_joints: List[List[str]]
        char_joints: List[List[str]]

    def __init__(self, retarget_cfg_fn: str) -> None:  # noqa: C901
        retarget_cfg_p = resolve_ad_filepath(retarget_cfg_fn, 'retarget cfg')
        with open(str(retarget_cfg_p), 'r') as f:
            retarget_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # validate character starting location
        try:
            self.char_start_loc = retarget_cfg['char_starting_location']
            assert len(self.char_start_loc) == 3, 'char start loc must be of len 3'
            for val in self.char_start_loc:
                assert isinstance(val, (float, int)), 'type must be float or int'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating char start location: {e}'
            logging.critical(msg)
            assert False, msg

        # validate retargeting method
        try:
            self.retargeting_method = retarget_cfg['retargeting_method']
            # assert self.retargeting_method in ['twisted_perspective_2D', 'twisted_perspective_2D_clamped', 'twisted_perspective_2D_viewdependent'], 'unsupported retargeting_method specified'
            assert self.retargeting_method in ['twisted_perspective_2D_viewdependent'], 'unsupported retargeting_method specified'
        except (AssertionError, KeyError) as e:
            msg = f'setting retargeting_method to twisted_perspective_2D: {e}'
            logging.info(msg)
            self.retargeting_method = 'twisted_perspective_2D'

        # validate bvh project bodypart groups
        self.bvh_projection_bodypart_groups: List[RetargetConfig.BvhProjectionBodypartGroup]
        try:
            self.bvh_projection_bodypart_groups = retarget_cfg['bvh_projection_bodypart_groups']

            for group in self.bvh_projection_bodypart_groups:
                assert group['method'] in ['pca', 'saggital', 'frontal'], 'group method must be "pca", "saggital", or "frontal"'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating bvh_projection_bodypart_groups: {e}'
            logging.critical(msg)
            assert False, msg

        # Check that group names are unique
        try:
            group_names = [group['name'] for group in self.bvh_projection_bodypart_groups]
            assert len(group_names) == len(set(group_names)), 'group names are not unique'
        except AssertionError as e:
            msg = f'Error validating bvh_projection_bodypart_groups: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char bodypart groups
        self.char_bodypart_groups: List[RetargetConfig.CharBodypartGroup]
        try:
            self.char_bodypart_groups = retarget_cfg['char_bodypart_groups']
            for group in self.char_bodypart_groups:
                assert len(group['bvh_depth_drivers']) > 0, 'bvh_depth_drivers must have at least one joint specified'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating char_bodypart_groups: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char bvh root offset
        self.char_bvh_root_offset: RetargetConfig.CharBvhRootOffset
        try:
            self.char_bvh_root_offset = retarget_cfg['char_bvh_root_offset']
            assert len(self.char_bvh_root_offset['bvh_joints']) > 0, 'bvh_joints list must be greater than zero'
            for each in self.char_bvh_root_offset['bvh_joints']:
                assert len(each) > 0, 'each list in bvh_joints must have len > 0'

            assert len(self.char_bvh_root_offset['char_joints']) > 0, 'char_joints list must be greater than zero'
            for each in self.char_bvh_root_offset['char_joints']:
                assert len(each) > 0, 'each list in char_joints must have len > 0'

            assert isinstance(self.char_bvh_root_offset['bvh_projection_bodypart_group_for_offset'], str), 'bvh_projection_bodypart_group_for_offset must be str'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating char_bvh_root_offset: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char joint bvh joints mapping
        self.char_joint_bvh_joints_mapping: Dict[str, Tuple[str, str]]
        try:
            self.char_joint_bvh_joints_mapping = retarget_cfg['char_joint_bvh_joints_mapping']
            for key, val in self.char_joint_bvh_joints_mapping.items():
                assert isinstance(key, str), 'key must be str'
                assert isinstance(val, tuple), 'val must be tuple'
                assert len(val) == 2, 'val must be of len 2'
                assert isinstance(val[0], str) and isinstance(val[1], str), 'values must be str'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating char_bvh_root_offset: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char runtime checks
        self.char_runtime_checks: List[str]
        try:
            self.char_runtime_checks = retarget_cfg['char_runtime_checks']
            for check in self.char_runtime_checks:
                assert check[0] in ['above'], 'currently only above check is supported'
                if check[0] == 'above':
                    assert len(check) == 4, 'above check needs 3 additional parameters'
        except (AssertionError, ValueError) as e:
            msg = f'Error validating char_runtime_checks: {e}'
            logging.critical(msg)
            assert False, msg

    def validate_char_and_bvh_joint_names(self, char_joint_names: List[str], bvh_joint_names: List[str]) -> None:  # noqa: C901

        # validate bvh_projection_bodypart_groups
        try:
            for group in self.bvh_projection_bodypart_groups:
                for bvh_joint_name in group['bvh_joint_names']:
                    assert bvh_joint_name in bvh_joint_names, f'bvh_joint_name not valid: {bvh_joint_name}'
        except AssertionError as e:
            msg = f'Error validating bvh_projection_bodypart_groups: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char_bodypart_groups
        try:
            for group in self.char_bodypart_groups:
                # check that bvh joint drivers are valid bvh joints
                for bvh_joint_name in group['bvh_depth_drivers']:
                    assert bvh_joint_name in bvh_joint_names, f'bvh_depth_driver joint name invalid: {bvh_joint_name}'

                # check that all char_joints are valid character joints
                for char_joint_name in group['char_joints']:
                    assert char_joint_name in char_joint_names, f'char_joints joint name invalid: {char_joint_name}'
        except AssertionError as e:
            msg = f'Error validating char_bodypart_groups: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char_bvh_root_offset
        try:
            # check that bvh_projection_bodypart_group_for_offset matches a bvh_projection_bodypart_group name
            group_names = [group['name'] for group in self.bvh_projection_bodypart_groups]
            assert self.char_bvh_root_offset['bvh_projection_bodypart_group_for_offset'] in group_names, 'invalid bvh_projection_bodypart_group_for_offset'

            # check bvh_joints contains valid joints
            for bvh_joint_name_group in self.char_bvh_root_offset['bvh_joints']:
                for joint_name in bvh_joint_name_group:
                    assert joint_name in bvh_joint_names, f'invalid joint name in bvh_joints: {joint_name}'

            # check char_joints are valid joints
            for char_joint_name_group in self.char_bvh_root_offset['char_joints']:
                for joint_name in char_joint_name_group:
                    assert joint_name in char_joint_names, f'invalid joint name in char_joints: {joint_name}'
        except AssertionError as e:
            msg = f'Error validating char_bvh_root_offset: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char_joint_bvh_joints_mapping
        try:
            # check that dict keys correspond to valid character joints
            for char_joint_name in self.char_joint_bvh_joints_mapping.keys():
                assert char_joint_name in char_joint_names, f'invalid char_joint_name: {char_joint_name}'

            # check that dict values correspond to valid bvh joints
            for bvh_prox_joint_name, bvh_dist_joint_name in self.char_joint_bvh_joints_mapping.values():
                assert bvh_prox_joint_name in bvh_joint_names, f'invalid bvh_prox_joint_name: {bvh_prox_joint_name}'
                assert bvh_dist_joint_name in bvh_joint_names, f'invalid bvh_dist_joint_name: {bvh_dist_joint_name}'
        except AssertionError as e:
            msg = f'Error validating char_joint_bvh_joints_mapping: {e}'
            logging.critical(msg)
            assert False, msg

        # validate char runtime checks
        try:
            for check in self.char_runtime_checks:
                if check[0] == 'above':
                    # check that, if above test, following 3 params are valid character joint names
                    _, target_joint_name, joint1_name, joint2_name = check
                    assert target_joint_name in char_joint_names, f'above test target_joint_name invalid {target_joint_name}'
                    assert joint1_name in char_joint_names, f'above test joint1_name invalid {joint1_name}'
                    assert joint2_name in char_joint_names, f'above test joint2_name invalid {joint2_name}'
        except AssertionError as e:
            msg = f'Error validating char_runtime_checks: {e}'
            logging.critical(msg)
            assert False, msg


def resolve_ad_filepath(file_name: str, file_type: str) -> Path:
    """
    Given input filename, attempts to find the file, first by relative to cwd,
    then by absolute, the relative to animated_drawings root directory.
    If not found, prints error message indicating which file_type it is.
    """
    if Path(file_name).exists():
        return Path(file_name)
    elif Path.joinpath(Path.cwd(), file_name).exists():
        return Path.joinpath(Path.cwd(), file_name)
    elif Path(resource_filename(__name__, str(file_name))).exists():
        return Path(resource_filename(__name__, file_name))
    elif Path(resource_filename(__name__, str(Path('..', file_name)))).exists():
        return Path(resource_filename(__name__, str(Path('..', file_name))))

    msg = f'Could not find the {file_type} specified: {file_name}'
    logging.critical(msg)
    assert False, msg


NoneType = type(None)  # needed for type checking
