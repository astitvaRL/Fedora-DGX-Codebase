import numpy as np
from parts import BasePart, InternalPart, ExternalPart
from PIL import Image
import copy
from typing import Dict, Tuple
import json
from keypoint import Keypoint
from pathlib import Path
import numpy.typing as npt
import cv2
# from infiller import Infiller
import yaml
import shutil
from tkinter import filedialog
import fileinput
import sys


class View():
    """ A view is essentially a copy of the character. It has its own parts and keypoints. The major difference is that, during
    initialization, it goes through a checks that all the external parts and internal parts indicate a consistent forward orientation,
    either D(rawing)Right or D(rawing)Left.
    """
    def __init__(self, tool, forward_orientation: str):

        self.image_name = tool.image_name

        self.parts = copy.deepcopy(tool.parts)
        self.keypoints = copy.deepcopy(tool.character_joint_keypoints)
        for keypoint in self.keypoints:
            keypoint.y = int(keypoint.y)  # TODO: fix annotation tool so it's converted to int there
            keypoint.x = int(keypoint.x)

        """ Create the data structures we will need to generate the different views """

        # map name to part
        self.name_to_part: Dict[str, BasePart] = {}
        for part in self.parts:
            assert part.name not in self.name_to_part.keys(), f'error: duplicate name: {part.name}'
            self.name_to_part[part.name] = part

        # root part
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
            part.infill_mask = np.full(part.texture.shape[:2], False)
            for c in part.get_children():
                part.infill_mask = np.logical_or(part.infill_mask, self.get_visible_pixels_of_transforming_part_and_children(c))

            # only need to infill these pixels if they are inside of the part's actual mask
            part.infill_mask = np.logical_and(part.infill_mask, part.mask)

        # # ... and do the actual infilling
        # infiller = Infiller()
        # for part in self.parts:
        #     if len(part.infill_mask[part.infill_mask == True]) == 0:
        #         continue
        #     infill_img = infiller.infill(prompt="blank", image_np=part.texture, mask_np=part.infill_mask)
        #     part.texture[part.infill_mask == True] = np.array(infill_img)[:, :, :3][part.infill_mask == True]

        # ... and we're going to set their transforms here
        for part in self.parts:
            part.transforms: Dict[str, npt.NDArray] = {
                'left': np.identity(3),
                'right': np.identity(3),
                'center': np.identity(3)
            }

        # ensure that the root joint for skeleton is within the root_mask
        assert self.keypoints[0].name == 'root'
        assert self.root_part.mask[self.keypoints[0].y, self.keypoints[0].x] == True, 'root joint not within root_part.mask'

        # We'll need to figure out which pixels must be infilled due to external part flips
        # Starting with root, we check each child. If child root points one direction and child points the other or no direct, we mark the split and the pixels above and below it for infill
        # We do this recursively
        # when the forward orientation is being set, we need to likewise flip the pixels to infill
        # print('Do not forget to infill between external parts')

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

        if type(part) is ExternalPart:
            return

        # compute bounding box
        if len(part.mask[part.mask == True]) != 0:  # part has it's own visible pixels
            true_y, true_x = np.where(part.mask)
            x0, x1 = int(np.min(true_x)), int(np.max(true_x))
            y0, y1 = int(np.min(true_y)), int(np.max(true_y))
        else:  # part has no visible pixels of its own. Use direct children to get bounding box
            x0 = int(np.min([c.bounding_box[0] for c in part.get_children()]))
            y0 = int(np.min([c.bounding_box[1] for c in part.get_children()]))
            x1 = int(np.max([c.bounding_box[2] for c in part.get_children()]))
            y1 = int(np.max([c.bounding_box[3] for c in part.get_children()]))
        part.bounding_box = [x0, y0, x1, y1]

        # compute point representation
        if part.flip_as_drawn == "":  # part has no orientation
            px = round((x0 + x1)/2)
            py = round((y0 + y1)/2)
        elif part.flip_as_drawn == 'Drawing Left':  # part points left. Use right side, midway up
            px = x1
            py = round((y0 + y1)/2)
        elif part.flip_as_drawn == 'Drawing Right':  # part points right. Use left side, midway up
            px = x0
            py = round((y0 + y1)/2)

        part.point = [px, py]

    def _compute_part_translation_for_view(self, part: InternalPart, view: int) -> Tuple[int, int]:
        """ view is an int ranging from 0 to 100. 0 in drawing left, 50 is center, 100 is drawing right """

        # translation depends upon translation of parent parts, so get them
        transforms = []
        _part = part
        while True:
            transform_left = _part.transforms['left']
            transform_right = _part.transforms['right']

            view_transform = np.identity(3)
            view_transform[0, -1] = round((1 - (view/100)) * transform_left[0, -1] + (view/100) * transform_right[0, -1])
            view_transform[1, -1] = round((1 - (view/100)) * transform_left[1, -1] + (view/100) * transform_right[1, -1])

            # transforms.append(_part.transforms[view])
            transforms.append(view_transform)
            if _part.parent_name is None:
                break
            _part = self.name_to_part[_part.parent_name]

        # matmul to get global transforms
        global_transform = np.identity(3)
        for trans in transforms[::-1]:
            global_transform = global_transform @ trans

        # pull out what's needed for display
        x_offset = int(global_transform[0][-1])
        y_offset = int(global_transform[1][-1])

        return (x_offset, y_offset)

    def _compute_translated_mask_for_view(self, part: InternalPart, view: int) -> npt.NDArray:
        """ view is an int ranging from 0 to 100. 0 in drawing left, 50 is center, 100 is drawing right """

        translated_mask = np.full(part.mask.shape, False)

        x0, y0, x1, y1 = part.bounding_box

        x_offset, y_offset = self._compute_part_translation_for_view(part, view)

        tx0 = x0 + x_offset
        tx1 = x1 + x_offset
        ty0 = y0 + y_offset
        ty1 = y1 + y_offset

        translated_mask[ty0:ty1+1, tx0:tx1+1] = part.mask[y0:y1+1, x0:x1+1]  # +1 to get True vals in last row and col

        return translated_mask

    def _compute_part_alpha_mask(self, part: InternalPart, view: int) -> npt.NDArray:
        """ view is an int ranging from 0 to 100. 0 in drawing left, 50 is center, 100 is drawing right """

        lineage = []
        _part = part
        while _part.parent_name is not None:
            lineage.insert(0, _part)
            _part = self.name_to_part[_part.parent_name]

        alpha_mask = np.full(part.mask.shape, True)
        for _part in lineage:
            if type(_part) is not InternalPart:
                continue
            if _part.hide_outside_parent == True:
                translated_parent_mask = self._compute_translated_mask_for_view(self.name_to_part[_part.parent_name], view)
                alpha_mask = np.logical_and(alpha_mask, translated_parent_mask)

        return alpha_mask

    def show_txtr(self, view: int = 50, show_in_window=True):
        """ view is an int ranging from 0 to 100. 0 in drawing left, 50 is center, 100 is drawing right """

        txtr = np.full(self.parts[0].texture.shape, 0)

        for part in self.parts:

            if type(part) is InternalPart:
                translated_part_mask = self._compute_translated_mask_for_view(part, view)
                translated_part_alpha_mask = self._compute_part_alpha_mask(part, view)

                hide_outside_parent = True
                if hide_outside_parent:
                    _tmp_txtr = np.full([*self.parts[0].texture.shape[:2], 4], 0)  # placehold texture of zeros
                    _tmp_txtr[translated_part_mask, :3] = part.texture[part.mask]  # copy in part pixels where they will be
                    _tmp_txtr[translated_part_mask, 3] = 255  # set alpha to visible for pixels belonging to part
                    _tmp_txtr[np.logical_not(translated_part_alpha_mask), 3] = 0  # set alpha to invisible for part pixels outside the alpha mask
                    txtr[_tmp_txtr[:, :, 3] == 255] = _tmp_txtr[:, :, :3][_tmp_txtr[:, :, 3] == 255]  # only copy over the pixels that are visible to final txtr
                else:
                    txtr[translated_part_mask] = part.texture[part.mask]
            else:
                txtr[part.mask] = part.texture[part.mask]

            show_bounding_boxes = False
            if show_bounding_boxes:
                if type(part) is InternalPart:
                    x_offset, y_offset = self._compute_part_translation_for_view(part, view)
                    x0, y0, x1, y1 = part.bounding_box
                    x0 += x_offset
                    x1 += x_offset
                    y0 += y_offset
                    y1 += y_offset
                    txtr[y0, x0:x1, :] = [255, 255, 255]
                    txtr[y1, x0:x1, :] = [255, 255, 255]
                    txtr[y0:y1, x0, :] = [255, 255, 255]
                    txtr[y0:y1, x1, :] = [255, 255, 255]
                else:
                    pass

            show_points = False
            if show_points:
                if type(part) is InternalPart:
                    x_offset, y_offset = self._compute_part_translation_for_view(part, view)
                    px, py = part.point
                    px += x_offset
                    py += y_offset
                    txtr[py-1:py+1, px-1:px+1, :] = [255, 0, 0]
                else:
                    pass

        for kpt in self.keypoints:
            txtr[kpt.y, kpt.x] = [255, 0, 0]

        if show_in_window:
            Image.fromarray(txtr.astype(np.uint8)).show()
        return txtr

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

        masks_dir = Path(file_path).parent / 'masks'
        txtrs_dir = Path(file_path).parent / 'txtrs'

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


def export_sequence(vl, vr):
    for view in range(0, 51):
        img = vl.show_txtr(view, show_in_window=False)
        cv2.imwrite(f'outdir/test_{view}.png', cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    for view in range(50, 101):
        img = vr.show_txtr(view, show_in_window=False)
        cv2.imwrite(f'outdir/test_{view}.png', cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))


def export_ad3d_annotations(vl, vr):
    # get directory where things are exported to

    selected_directory = filedialog.askdirectory(title="Select Directory to Create New Folder")
    parent_outdir = Path(selected_directory)/Path(vl.image_name).stem
    try:
        shutil.rmtree(parent_outdir)
    except Exception:
        pass
    parent_outdir.mkdir(exist_ok=False, parents=True)

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
        cv2.imwrite(f'{outdir}/texture.png', cv2.cvtColor(external_parts_txtr, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(f'{outdir}/mask.png', 255*external_parts_mask.astype(np.uint8))

        # create the mvc for this character
        template_path = 'template_mvc.yaml'
        copy_path = f'{outdir}/mvc.yaml'
        shutil.copy(template_path, copy_path)
        with fileinput.FileInput(copy_path, inplace=True) as file:
            for line in file:
                line = line.replace('<<CHARACTER_CONFIG_PATH>>', char_cfg_fn)
                print(line, end='')

        run_cmd = f'python render.py {copy_path}'
        with open(f'{outdir}/run_cmd.txt', 'w') as f:
            f.write(run_cmd)

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


if __name__ == '__main__':
    export_kpts_json_fn = sys.argv[1]
    # with open(f'/Users/hjessmith/Desktop/character_style_preference/user study 2/stickFigure/tk_output/part_export_kpts.json', 'r') as f:
    with open(export_kpts_json_fn, 'r') as f:
        kpts = json.load(f)
    tool = MockTool(kpts)

    original_img_fn = sys.argv[3]
    # image_fn = f'/Users/hjessmith/Desktop/character_style_preference/user study 2/stickFigure/8faf13d89edc47d7ae844ad98f71b1c5.png'
    tool.image_cv2 = cv2.cvtColor(cv2.imread(original_img_fn), cv2.COLOR_BGR2RGB)
    tool.image_name = Path(original_img_fn).stem

    export_parts_json_fn = sys.argv[2]
    tool.load_parts_from_export(export_parts_json_fn)

    vr = View(tool, 'DRight')
    vr.generate_drawing_left_right_view()
    # vr.show_txtr(view=100)

    vl = View(tool, 'DLeft')
    vl.generate_drawing_left_right_view()
    # vl.show_txtr(view=0)

    # export_sequence(vl, vr)

    export_ad3d_annotations(vl, vr)
