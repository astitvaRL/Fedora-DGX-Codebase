import numpy as np
from parts import BasePart, InternalPart, ExternalPart
from PIL import Image
import copy
from typing import Dict
import json
from keypoint import Keypoint
from pathlib import Path
import numpy.typing as npt
import cv2
from infiller import Infiller
import yaml
import shutil
from tkinter import filedialog
import fileinput
import sys


class View():
    """ A view is essentially a copy of the character. It has its own parts and keypoints. The major difference is that, during
    initialization, it goes through and checks that all the external parts and internal parts indicate a consistent forward orientation,
    either D(rawing)Right or D(rawing)Left.
    """
    def __init__(self, tool, forward_orientation: str):

        self.image_name = tool.image_name

        # create a copy of keypoints and parts for this view to own
        self.parts = copy.deepcopy(tool.parts)
        self.keypoints = copy.deepcopy(tool.character_joint_keypoints)
        for keypoint in self.keypoints:
            keypoint.y = int(keypoint.y)
            keypoint.x = int(keypoint.x)

        """ Create the data structures we will need to generate the view """

        # map name to part
        self.name_to_part: Dict[str, BasePart] = {}
        for part in self.parts:
            assert part.name not in self.name_to_part.keys(), f'error: duplicate name: {part.name}'
            self.name_to_part[part.name] = part

        # root part is the part without a parent. Throw error if none or more than one
        self.root_part = None
        assert [part.parent_name for part in self.parts].count(None) == 1, 'error: number of parts without parent != 1'
        self.root_part = [part for part in self.parts if part.parent_name is None][0]

        # construct tree
        for part in self.parts:
            if part.parent_name is None:
                continue
            assert part.parent_name in self.name_to_part.keys(), f'error: parent_name not in parts: {part.parent_name}'
            self.name_to_part[part.parent_name].add_child(part)

        # for each part, compute any geometry needed for later on
        self.compute_display_geometry_recursively(self.root_part)

        # as a stopgap measure, we're going to give each part a copy of the original image here
        for part in self.parts:
            part.texture = tool.image_cv2.copy()

        # ... and we're going to give it a mask indicating which pixels may need to be infilled
        for part in self.parts:
            part.infill_mask = np.full(part.texture.shape[:2], False)  # initialize to empty
            for c in part.get_children():  # for each child, add their infill mask, or their visible pixels if that part might transform
                part.infill_mask = np.logical_or(part.infill_mask, self.get_visible_pixels_of_transforming_part_and_children(c))

            # restrict part's infill_mask so it's only true within the part's mask
            part.infill_mask = np.logical_and(part.infill_mask, part.mask)

        # ... and do the actual infilling
        infiller = Infiller()
        for part in self.parts:
            if len(part.infill_mask[part.infill_mask == True]) == 0:  # if part has zero pixels needing infilling, skip
                continue
            infill_img = infiller.infill(prompt="blank", image_np=part.texture, mask_np=part.infill_mask)
            part.texture[part.infill_mask == True] = np.array(infill_img)[:, :, :3][part.infill_mask == True]

        # ... and we're going to set their transforms here
        for part in self.parts:
            part.transforms: Dict[str, npt.NDArray] = {
                'left': np.identity(3),
                'right': np.identity(3),
                'center': np.identity(3)
            }

        # ensure that the root joint for skeleton is within the root_mask
        assert self.keypoints[0].name == 'root'
        assert self.root_part.mask[self.keypoints[0].y, self.keypoints[0].x] == True, 'root joint not within root part'

        # We'll need to figure out which pixels must be infilled due to external part flips
        # Starting with root, we check each child. If child root points one direction and child points the other or no direct, we mark the split and the pixels above and below it for infill
        # We do this recursively
        # when the forward orientation is being set, we need to likewise flip the pixels to infill
        # TODO Add this functionality later, skip & print message for now
        print('Do not forget to infill between external parts')

        # flip character pieces around to create forward orientation
        self.forward_orientation = forward_orientation
        self.set_forward_orientation_as(self.root_part, self.forward_orientation)

        # since we've flipped things around when setting orientation, recompute geometry
        self.compute_display_geometry_recursively(self.root_part)

    def get_visible_pixels_of_transforming_part_and_children(self, part, include_all_internal=False):

        if type(part) is InternalPart:  # this is only for internal parts
            if part.does_rdtwp or part.flip_as_drawn != "" or include_all_internal:  # if an ancestor transforms around, we need all it's children, even those that don't transform themselves
                visible_pixels = part.mask.copy()
                include_all_internal = True
            else:
                visible_pixels = np.full(part.mask.shape[:2], False)
        else:
            visible_pixels = np.full(part.mask.shape[:2], False)

        for c in part.get_children():
            visible_pixels = np.logical_or(visible_pixels, self.get_visible_pixels_of_transforming_part_and_children(c, include_all_internal))

        return visible_pixels

    def compute_display_geometry_recursively(self, part: BasePart):
        for c in part.get_children():
            self.compute_display_geometry_recursively(c)

        # ExternalParts are just horizonal segmentations of the base mesh. They don't have their own geometry
        if type(part) is ExternalPart:
            return

        # compute bounding box
        if len(part.mask[part.mask == True]) != 0:  # if part has it's own visible pixels
            true_y, true_x = np.where(part.mask)
            x0, x1 = int(np.min(true_x)), int(np.max(true_x))
            y0, y1 = int(np.min(true_y)), int(np.max(true_y))
        else:  # else part has no visible pixels of its own. Use direct children to get bounding box
            x0 = int(np.min([c.bounding_box[0] for c in part.get_children()]))
            y0 = int(np.min([c.bounding_box[1] for c in part.get_children()]))
            x1 = int(np.max([c.bounding_box[2] for c in part.get_children()]))
            y1 = int(np.max([c.bounding_box[3] for c in part.get_children()]))
        part.bounding_box = [x0, y0, x1, y1]
        
        # compute point representation
        if part.flip_as_drawn == "":  # if part has no orientation, it's center of bounding box
            px = round((x0 + x1)/2)
            py = round((y0 + y1)/2)
        elif part.flip_as_drawn == 'Drawing Left':  # part points left. Use midpoint of bbox right side
            px = x1
            py = round((y0 + y1)/2)
        elif part.flip_as_drawn == 'Drawing Right':  # part points right. Use midpoint of boox left side
            px = x0
            py = round((y0 + y1)/2)
        
        part.point = [px, py]

    def get_external_parts_mask_and_txtr(self):
        mask = np.full(self.root_part.mask.shape, False)
        txtr = np.full([*self.root_part.texture.shape[:-1], 4], 0, dtype=np.uint8)
        for part in self.parts:
            if type(part) is not ExternalPart:
                continue
            mask[part.mask == True] = True
            txtr[:, :, :3][part.mask == True] = part.texture[part.mask == True]
            txtr[:, :, 3][part.mask == True] = 255

        return mask, txtr

    def show_mask(self):
        mask = np.full(self.parts[0].mask.shape, 0)
        for part in self.parts:
            mask[part.mask] = part.mask[part.mask]
        Image.fromarray(mask.astype(np.bool_)).show()

    def set_forward_orientation_as(self, part, view):
        """ recursively goes through part and each child, and calls function to flip it if needed based on view """

        flip_axis = self._get_flip_axis(part)

        if type(part) is ExternalPart:
            if view == 'DLeft' and part.forward_orientation == 'DRight':
                self._flip_part_and_children(part, flip_axis)
            elif view == 'DRight' and part.forward_orientation == 'DLeft':
                self._flip_part_and_children(part, flip_axis)

        elif type(part) is InternalPart:
            if view == 'DLeft' and part.flip_as_drawn == 'Drawing Right':
                self._flip_part_and_children(part, flip_axis)
            elif view == 'DRight' and part.flip_as_drawn == 'Drawing Left':
                self._flip_part_and_children(part, flip_axis)

        for child in part.get_children():
            self.set_forward_orientation_as(child, view)

    def _get_flip_axis(self, part) -> int:
        """ determines the x coordinate of the vertical axis to flip the part about """
        if type(part) is ExternalPart:

            if part == self.root_part:  # the root part is flipped about the root keypoint
                return self.keypoints[0].x

            # for other parts, find where it attaches to parent, and flip at midpoint
            if part.bottom_split and part.bottom_split[1] == self.name_to_part[part.parent_name].top_split[1]:  # is attached via bottom split
                return round((part.bottom_split[0] + part.bottom_split[2]) / 2)
            elif part.top_split and part.top_split[1] == self.name_to_part[part.parent_name].bottom_split[1]:  # is attached via top split
                return round((part.top_split[0] + part.top_split[2]) / 2)

        elif type(part) is InternalPart:  # if internal part, we flip it around it's midpoint
            return round((part.bounding_box[0] + part.bounding_box[2]) / 2)

        assert False, 'problem determining flip axis'

    def _flip_part_and_children(self, part, flip_axis):

        if type(part) is ExternalPart:
            self._flip_external_part(part, flip_axis)

        if type(part) is InternalPart:
            self._flip_internal_part(part, flip_axis)

        for child in part.get_children():
            self._flip_part_and_children(child, flip_axis)

    def _flip_external_part(self, part: ExternalPart, flip_axis: int):

        # flip keypoints within the part
        for kpt in self.keypoints:
            if part.mask[kpt.y, kpt.x] == True:
                if 'right' in kpt.name:
                    kpt.name = kpt.name.replace('right', 'left')
                elif 'left' in kpt.name:
                    kpt.name = kpt.name.replace('left', 'right')
                kpt.x = 2*flip_axis - kpt.x

        # flip the mask
        self._flip_part_mask_and_texture(part, flip_axis)

        # update it's orientation
        if part.forward_orientation == 'DRight':
            part.forward_orientation = 'DLeft'
        elif part.forward_orientation == 'DLeft':
            part.forward_orientation = 'DRight'

        # update values of top/bottom split
        if part.top_split:
            x0, y0, x1, y1 = part.top_split
            x01 = 2*flip_axis - x1
            x11 = 2*flip_axis - x0
            part.top_split = [x01, y0, x11, y1]
        if part.bottom_split:
            x0, y0, x1, y1 = part.bottom_split
            x01 = 2*flip_axis - x1
            x11 = 2*flip_axis - x0
            part.bottom_split = [x01, y0, x11, y1]

    def _flip_internal_part(self, part: InternalPart, flip_axis: int):

        # flip the part's mask and texture if it has visible parts
        if part.mask[part.mask == True].size != 0:
            self._flip_part_mask_and_texture(part, flip_axis)

        # update the part's forward orienation if necessary
        if part.flip_as_drawn == 'Drawing Left':
            part.flip_as_drawn = 'Drawing Right'
        elif part.flip_as_drawn == 'Drawing Right':
            part.flip_as_drawn = 'Drawing Left'

        # flip the mask's bounding box
        old_x0, y0, old_x1, y1 = part.bounding_box
        new_x0 = 2*flip_axis - old_x1
        new_x1 = 2*flip_axis - old_x0
        part.bounding_box = [new_x0, y0, new_x1, y1]

    def _flip_part_mask_and_texture(self, part: BasePart, flip_axis: int):

        true_indices = np.where(part.mask)
        min_row, max_row = np.min(true_indices[0]), np.max(true_indices[0])
        min_col, max_col = np.min(true_indices[1]), np.max(true_indices[1])

        try:
            # flip the mask around vertical flip axis
            flipped_mask = part.mask[min_row:max_row+1, min_col:max_col+1][:, ::-1].copy()  # flip and save mask in a copy
            part.mask[min_row:max_row+1, min_col:max_col+1] = False  # set the original bounding box to False
            part.mask[min_row:max_row+1, flip_axis-(max_col-flip_axis):flip_axis+(flip_axis-min_col)+1] = flipped_mask  # put flipped mask in

            # flip the texture around vertical flip axis
            flipped_texture = part.texture[min_row:max_row+1, min_col:max_col+1][:, ::-1, :].copy()
            part.texture[min_row:max_row+1, min_col:max_col+1, :] = 0  # set the original bounding box to False
            part.texture[min_row:max_row+1, flip_axis-(max_col-flip_axis):flip_axis+(flip_axis-min_col)+1] = flipped_texture  # put flipped mask in
        except Exception as e:
            print(f'error flipping mask: {e}')

    def generate_drawing_left_right_view(self):
        view = 'right'
        # this manipulates Internal Parts to make the character appear to be looking far to the right

        # for each part, if it needs to be translated when being viewed from the right, check how much
        for view, horizontal_offset in [('left', -1), ('right', 1)]:
            for part in self.parts:
                if type(part) is not InternalPart:
                    continue
                if part.does_rdtwp:  # if this thing should translate within its parent

                    # starting at the part's point approximation,  process towards drawing right and stop when we fall outside the mask
                    px, py = part.point
                    mask = self.name_to_part[part.parent_name].mask
                    assert mask[py, px], f'part point approximation does not fall within parent mask. part: {part.name}, parent: {part.parent_name}'

                    mask_bound_x = px
                    while mask[py, mask_bound_x + horizontal_offset] == True:
                        mask_bound_x += horizontal_offset

                    # update the transform for this part
                    part.transforms[view][0, -1] = mask_bound_x - px


class MockTool():
    def __init__(self, kpts):
        self.parts = []
        self.character_joint_keypoints = []
        for k in kpts:
            self.character_joint_keypoints.append(Keypoint([k[1], k[2]], 0, k[0], ""))

    def load_parts_from_export(self, file_path):

        with open(file_path, 'r') as f:
            parts_list = json.load(f)

        masks_dir = Path(file_path).parent / f'{Path(file_path).stem}_masks'
        txtrs_dir = Path(file_path).parent / f'{Path(file_path).stem}_txtrs'

        for part_json in parts_list:

            if part_json['Type'] == 'Internal':

                name = part_json['Name']
                mask: npt.NDArray[np.uint8] = cv2.imread(f'{masks_dir}/{name}.png')[:, :, 0] != 0  # rgb to binary
                does_rdtwp = part_json['RotationDrivesTranslationWithinParent']
                hide_on_backside = part_json['HideOnBackside']
                hide_outside_parent = part_json['HideOutsideParent']
                rotation_drives_flip = part_json['RotationDrivesFlip']
                flip_as_drawn = part_json['FlipAsDrawn']
                parent_name = part_json['ParentName']

                part = InternalPart(image=self.image_cv2, name=name)
                part.set_mask(mask)
                part.hide_on_backside = hide_on_backside
                part.hide_outside_parent = hide_outside_parent
                part.rotation_drives_flip = rotation_drives_flip
                part.does_rdtwp = does_rdtwp
                part.flip_as_drawn = flip_as_drawn
                part.parent_name = parent_name
            
            elif part_json['Type'] == 'External':
                name = part_json['Name']
                mask: npt.NDArray[np.uint8] = cv2.imread(f'{masks_dir}/{name}.png')[:, :, 0] != 0  # rgb to binary
                txtr: npt.NDArray[np.uint8] = cv2.imread(f'{txtrs_dir}/{name}.png')
                orientation = part_json['ForwardOrientation']
                parent_name = part_json['ParentName']
                top_split = part_json['TopSplit']
                bottom_split = part_json['BottomSplit']

                part = ExternalPart()
                part.set_name(name)
                part.set_mask(mask)
                part.set_texture(txtr)
                part.set_parent_name(parent_name)
                part.set_forward_orientation(orientation)
                part.set_top_split(top_split)
                part.set_bottom_split(bottom_split)
            else:
                assert False

            self.parts.append(part)


def export_ad3d_annotations(vl, vr):

    # ask user where to save the rig
    selected_directory = filedialog.askdirectory(title="Select Directory to Create New Folder")
    parent_outdir = Path(selected_directory)/Path(vl.image_name).stem
    try:
        shutil.rmtree(parent_outdir)
    except Exception:
        pass
    parent_outdir.mkdir(exist_ok=False, parents=True)

    # {child:parent} joint hierarchy
    parent_dict = {
        'root': None,
        'hip': 'root',
        'torso': 'hip',
        'neck': 'torso',
        'right_shoulder': 'torso',
        'right_elbow': 'right_shoulder',
        'right_hand': 'right_elbow',
        'left_shoulder': 'torso',
        'left_elbow': 'left_shoulder',
        'left_hand': 'left_elbow',
        'right_hip': 'root',
        'right_knee': 'right_hip',
        'right_foot': 'right_knee',
        'left_hip': 'root',
        'left_knee': 'left_hip',
        'left_foot': 'left_knee',
    }

    # create the left and right version of the character
    for view, view_str in [(vl, 'left'), (vr, 'right')]:

        # make the directory that will contain this version of the character
        outdir = parent_outdir / view_str
        outdir.mkdir(exist_ok=False, parents=True)

        # using parent dictionary, create skeleton
        skeleton = []
        for k in view.keypoints:
            skeleton.append({
                'loc': [k.x, k.y],
                'name': k.name,
                'parent': parent_dict[k.name]
            })

        # since the mesh that will be manipulated via keypoints and ARAP is only comprised of background parts, get mask and txtr that is only those pieces 
        external_parts_mask, external_parts_txtr = view.get_external_parts_mask_and_txtr()

        # the config file contents
        char_cfg = {
            'height': external_parts_mask.shape[0],
            'width': external_parts_mask.shape[1],
            'skeleton': skeleton,
            'mesh_type': 'cardboard',
            'rig_type': '2D',
            'deformer_type': 'arap_igarashi',
            'footorientation_right': None,
            'footorientation_left': None
        }

        # export char_cfg, texture, and mask
        char_cfg_fn = f'{outdir}/char_cfg.yaml'
        with open(char_cfg_fn, 'w') as f:
            yaml.dump(char_cfg, f)
        cv2.imwrite(f'{outdir}/texture_original.png', cv2.cvtColor(external_parts_txtr, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(f'{outdir}/mask.png', 255*external_parts_mask.astype(np.uint8))

        masks_dir = f'{outdir}/masks'
        Path(masks_dir).mkdir(exist_ok=False, parents=False)

        texture_dir = f'{outdir}/texture'
        Path(texture_dir).mkdir(exist_ok=False, parents=False)

        with open(f'{outdir}/parts_info.yaml', 'w') as f:
            parts = []
            for part in view.parts:
                if type(part) is InternalPart:
                    parts.append({
                        'type': 'internal',
                        'does_rdtwp': part.does_rdtwp,
                        # 'rotation_drives_flip': part.rotation_drives_flip,
                        'forward_orientation': part.flip_as_drawn,
                        'hide_on_backside': part.hide_on_backside,
                        'hide_outside_part': part.hide_outside_parent,
                        'name': part.name,
                        'parent_name': part.parent_name,
                        'view_left_transform': part.transforms['left'].tolist(),
                        'view_right_transform': part.transforms['right'].tolist(),
                        'bounding_box': part.bounding_box,
                        'point_approximation': part.point,
                    })
                if type(part) is ExternalPart:
                    parts.append({
                        'type': 'external',
                        'name': part.name,
                        'top_split': part.top_split,
                        'bottom_split': part.bottom_split,
                        'parent_name': part.parent_name,
                        'forward_orientation': part.forward_orientation,
                    })

                # assert no parts have visible textures on the very end of the image
                assert len(part.mask[ 0, :][part.mask[ 0, :] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                assert len(part.mask[-1, :][part.mask[-1, :] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                assert len(part.mask[:,  0][part.mask[:,  0] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                assert len(part.mask[:, -1][part.mask[:, -1] == True]) == 0, f'error: part has visible pixels on border: {part.name}'
                cv2.imwrite(f'{masks_dir}/{part.name}.png', 255 * part.mask.astype(np.uint8))

                # write out texture
                try:
                    cv2.imwrite(f'{texture_dir}/{part.name}.png', cv2.cvtColor(part.texture, cv2.COLOR_RGB2BGR))
                except Exception:
                    print(f'no texture_rgba for part: {part.name}')

            yaml.dump(parts, f)

    # create the mvc for this character
    src_mvc_config = 'template_mvc.yaml'
    dst_mvc_config = f'{parent_outdir}/mvc.yaml'
    shutil.copy(src_mvc_config, dst_mvc_config)
    with fileinput.FileInput(dst_mvc_config, inplace=True) as file:
        for line in file:
            line = line.replace('<<CHARACTER_CONFIG_PATH>>', str(parent_outdir))
            print(line, end='')

    # run command
    run_cmd = f'python render.py {dst_mvc_config}'
    with open(f'{outdir}/run_cmd.txt', 'w') as f:
        f.write(run_cmd)
    print(run_cmd)


if __name__ == '__main__':

    """ Load annotations and image needed for rig generation """
    export_kpts_json_fn = sys.argv[1]  # the *_kpt.json exported by tool.py
    with open(export_kpts_json_fn, 'r') as f:
        kpts = json.load(f)
    tool = MockTool(kpts)

    original_img_fn = sys.argv[3]  # original image selected at beginning of tool.py user flow
    tool.image_cv2 = cv2.cvtColor(cv2.imread(original_img_fn), cv2.COLOR_BGR2RGB)
    tool.image_name = Path(original_img_fn).stem

    export_parts_json_fn = sys.argv[2]  # path to parts json exported tool.py
    tool.load_parts_from_export(export_parts_json_fn)

    """ Generate the various 'views' of the character """
    # create the 'drawing right' facing version of the character
    vr = View(tool, 'DRight')
    vr.generate_drawing_left_right_view()

    # create the 'drawing left' facing version of the character
    vl = View(tool, 'DLeft')
    vl.generate_drawing_left_right_view()

    """ export both views as the complete rig """
    export_ad3d_annotations(vl, vr)
