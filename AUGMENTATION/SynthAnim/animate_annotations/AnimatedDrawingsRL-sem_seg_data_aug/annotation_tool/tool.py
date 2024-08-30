from __future__ import annotations
from typing import List, Optional
import json
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import numpy.typing as npt

from pose_estimation.mmpose_handler import Handler
from segmentation.sam import SAM

from keypoint import Keypoint
from sam_point import SAMPoint
from stage_bar import StageBar

from pose_canvas import PoseCanvas
from pose_info import PoseInfo

from mask_split_canvas import MaskSplitCanvas
from mask_split_info import MaskSplitInfo

from fbseg_info import FBSegInfo
from fbseg_canvas import FBSegCanvas

from parts_info import PartsInfo
from parts_canvas import PartsCanvas
from parts import InternalPart, ExternalPart

from tkinter import filedialog
from pathlib import Path
import shutil


class AnnotationTool():

    def __init__(self, image_fn=None):

        # create the tk root
        self.root = tk.Tk()

        # get the image we are annotating
        if image_fn:
            self.image_fn = image_fn
        else:
            self.image_fn = filedialog.askopenfilename(title="Select Drawing")
        self.image_cv2 = cv2.cvtColor(cv2.imread(self.image_fn), cv2.COLOR_BGR2RGB)

        # add title
        self.root.title("AD3D Character Annotator")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Create a ttk frame to hold everything
        self.content = ttk.Frame(self.root, borderwidth=2, relief="ridge")
        self.content.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)
        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=1)

        # Create our stage bar
        self.current_stage_idx_var = tk.IntVar()  # current stage of tool
        self.current_stage_idx_var.trace('w', self._change_stage)
        self.stage_bar = StageBar(self, self.content, self.current_stage_idx_var)
        self.stage_bar.grid(column=0, row=1, columnspan=2, sticky=(tk.N, tk.W, tk.E, tk.S))

        # prep SAM
        self.sam = SAM()
        self.sam.set_image(self.image_cv2)

        """ Pose Stage """
        self.character_joint_keypoints: List[Keypoint] = []  # the skeletal keypoints used in Pose Stage
        self.active_oval_id_var = tk.IntVar()                # keeps track of the active oval in the pose stage
        self.active_oval_id_var.set(-1)

        # Pose annotation
        self.pose_info: Optional[ttk.Frame] = None
        self.pose_canvas: Optional[tk.Canvas] = None

        """ Foreground Background Stage """
        self.fbseg_info: Optional[FBSegInfo] =  None
        self.fbseg_canvas: Optional[FBSegCanvas] = None

        self.fb_sam_points: List[SAMPoint] = []  # the sam_points used for foreground background segmentation
        self.fb_mask = None  # foreground background segmentation mask

        """ Mask Split Stage """
        self.mask_split_info: Optional[MaskSplitInfo] = None
        self.mask_split_canvas: Optional[MaskSplitCanvas] = None

        """ Parts Stage """
        self.parts_info: Optional[PartsInfo] = None
        self.parts_canvas: Optional[PartsCanvas] = None

        self.parts: List[InternalPart] = []
        self.active_part: Optional[InternalPart] = None

        self.active_view: Optional[str] = None

        self.current_stage_idx_var.set(0)  # start at the beginning

    def _get_current_stage(self) -> str:
        stage_idx = self.current_stage_idx_var.get()
        if stage_idx == 0:
            return 'pose' 
        elif stage_idx == 1:
            return 'fg_seg'
        elif stage_idx == 2:
            return 'mask_split'
        elif stage_idx == 3:
            return 'parts'

    def _change_stage(self, *args):

        self.clear_content_frame()

        if self._get_current_stage() == 'pose':
            self.present_fresh_pose_stage()
        elif self._get_current_stage() == 'fg_seg':
            self.present_fresh_fbseg_stage()
        elif self._get_current_stage() == 'mask_split':
            self.present_fresh_mask_split_stage()
        elif self._get_current_stage() == 'parts':
            self.present_fresh_parts_stage()
        else:
            return

    def clear_content_frame(self):
        # remove pose
        if self.pose_info:
            self.pose_info.destroy()
        if self.pose_canvas:
            self.pose_canvas.destroy()

        # remove foreground background
        if self.fbseg_info:
            self.fbseg_info.destroy()
        if self.fbseg_canvas:
            self.fbseg_canvas.destroy()

        # remove mask split
        if self.mask_split_info:
            self.mask_split_info.destroy()
        if self.mask_split_canvas:
            self.mask_split_canvas.destroy()

        # remove parts
        if self.parts_info:
            self.parts_info.destroy()
        if self.parts_canvas:
            self.parts_canvas.destroy()

    def query_sam(self, sam_points: List[SAMPoint]) -> npt.NDArray[np.bool_]:
        coords = np.array([[k.x, k.y] for k in sam_points])
        labels = np.array([k.label for k in sam_points])
        return self.sam.predict_from_coords(coords, labels)

    def start(self):
        self.root.mainloop()

    """ Begin functions for handling parts """
    def delete_all_parts(self):

        treeview = self.get_visible_treeview()
        if treeview:
            treeview.clear_treeview()

        self.parts = []

    def create_new_part(self, name="", is_external=False, mask=None):
        if is_external:
            new_part = ExternalPart()
            new_part.set_mask(mask)
            new_part.set_name(name)
            new_part.set_texture(self.image_cv2.copy())
        else:
            new_part = InternalPart(name=name, editable=True, image=self.image_cv2.copy())  # create a new part

        if mask is None:  # initialize mask to empty
            mask = np.full(self.image_cv2.shape[:2], False, dtype=np.bool_)
        new_part.set_mask(mask)

        self.parts.append(new_part)

        self.active_part = new_part

        self.update_visible_treeview()

    def set_active_part(self, part: InternalPart):

        self._hide_part_widgets(self.active_part)

        self.active_part = part

        self._show_part_details(self.active_part)

    def _hide_part_widgets(self, part: Optional[InternalPart]):
        if part is None:
            return

        stage = self._get_current_stage()

        if stage == 'parts':
            if type(part) is ExternalPart:
                pass
            elif type(part) is InternalPart:
                # clean up sam_point ovals
                for sam_point in part.sam_points:
                    self.parts_canvas.delete(sam_point.oval_id)
                    sam_point.oval_id = None

    def _show_part_details(self, part: InternalPart):
        stage = self._get_current_stage()

        if stage == 'mask_split':
            if type(part) is InternalPart:
                print('warning: internl part found in mask_split stage')
            elif type(part) is ExternalPart:
                self.mask_split_info.external_part_frame.display_part_info(self.active_part)

        if stage == 'parts':
            if type(part) is ExternalPart:
                # set the new segmentation mask for this part
                self.parts_canvas.set_parts_seg_mask(self.active_part.mask)

                # hide the info frame
                self.parts_info.part_frame.grid_forget()

            elif type(part) is InternalPart:
                # clean up sam_point ovals
                for sam_point in part.sam_points:
                    self.parts_canvas.add_oval_from_sam_point(sam_point)

                # set the new segmentation mask for this part
                self.parts_canvas.set_parts_seg_mask(self.active_part.mask)

                # display information for part in part_info widget
                self.parts_info.part_frame.grid(column=0, row=3, columnspan=3)
                self.parts_info.part_frame.display_part_info(self.active_part)

    def export_parts(self):
        # create parts json
        parts_json = []
        for part in self.parts:
            parts_json.append(part.get_json_representation())

        # write parts json to file
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        with open(file_path, 'w') as f:
            json.dump(parts_json, f)

        # write out the keypoints
        kpt_file_path = Path(file_path).parent / f'{Path(file_path).stem}_kpts.json'
        with open(kpt_file_path, 'w') as f:
            json.dump([[k.name, k.x, k.y] for k in self.character_joint_keypoints], f)

        # create clean new dir for masks
        masks_dir = Path(file_path).parent / f'{Path(file_path).stem}_masks'
        if masks_dir.exists():
            shutil.rmtree(masks_dir)
        masks_dir.mkdir(exist_ok=True, parents=True)

        # write out masks
        for part in self.parts:
            mask_name = masks_dir / f'{part.name}'
            np.save(f'{mask_name}.npy', part.mask)
            cv2.imwrite(f'{mask_name}.png', 255 * part.mask)

        # create clean new dir for txtrs
        txtrs_dir = Path(file_path).parent / f'{Path(file_path).stem}_txtrs'
        if txtrs_dir.exists():
            shutil.rmtree(txtrs_dir)
        txtrs_dir.mkdir(exist_ok=True, parents=True)

        # write out txtrs
        for part in self.parts:
            if part.texture is not None:
                txtr_name = txtrs_dir / f'{part.name}'
                np.save(f'{txtr_name}.npy', part.texture)
                cv2.imwrite(f'{txtr_name}.png', 255 * part.texture)

    def load_parts_from_export(self, file_path=None):

        # delete existing parts
        self.delete_all_parts()

        # load parts from file
        if file_path is None:
            file_path = filedialog.askopenfilename(filetypes=[("AD3D Annotations", "*.json")])
        with open(file_path, 'r') as f:
            parts_list = json.load(f)

        masks_dir = Path(file_path).parent / f'{Path(file_path).stem}_masks'
        txtrs_dir = Path(file_path).parent / f'{Path(file_path).stem}_txtrs'

        for part_json in parts_list:

            if part_json['Type'] == 'Internal':

                name = part_json['Name']
                mask: npt.NDArray[np.uint8] = np.load(f'{masks_dir}/{name}.npy')
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
                mask: npt.NDArray[np.uint8] = np.load(f'{masks_dir}/{name}.npy')
                txtr: npt.NDArray[np.uint8] = np.load(f'{txtrs_dir}/{name}.npy')
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

        # update based on new parts
        self.update_visible_treeview()

    def get_visible_treeview(self):
        # get whatever treeview is currently visible
        stage = self._get_current_stage()

        if stage not in ['parts', 'mask_split']:
            return None

        if stage == 'parts' and self.parts_info:
            return self.parts_info.treeview
        elif stage == 'mask_split' and self.mask_split_info:
            return self.mask_split_info.treeview
        else:
            return None

    def update_visible_treeview(self):

        treeview = self.get_visible_treeview()
        if treeview:
            treeview.display_current_state_of_tool_parts()

    """ Begin Pose Specific Functions """
    def present_fresh_pose_stage(self):
        self.pose_info =  PoseInfo(self.content, active_oval_id_var=self.active_oval_id_var)
        self.pose_canvas = PoseCanvas(self.content, image_rgb=self.image_cv2, active_oval_id_var=self.active_oval_id_var)

        self.character_joint_keypoints = []
        self.query_ad_and_set_joint_keypoints(self.image_cv2)

        self.pose_info.set_listbox_items([x.name for x in self.character_joint_keypoints])

        self.pose_canvas.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E))
        self.pose_canvas.set_ad_keypoints(self.character_joint_keypoints)

        self.pose_info.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.S))

    def query_ad_and_set_joint_keypoints(self, image: npt.NDArray[np.uint8]) -> None:
        handler = Handler()
        pose_results = handler.predict_keypoints(image)
        if len(pose_results) != 1:
            raise Exception(f"AD pose detector did not detect exactly one figure in the scene: {pose_results}")

        kpts = np.array(pose_results[0]['keypoints'])

        self.character_joint_keypoints.append(Keypoint([round(x) for x in (kpts[11, :2]+kpts[12, :2])/2], c=(kpts[11, 2]+kpts[12, 2])/2, name='root', parent=None))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in (kpts[11, :2]+kpts[12, :2])/2], c=(kpts[11, 2]+kpts[12, 2])/2, name='hip', parent='root'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in (kpts[5, :2]+kpts[6, :2])/2  ], c=(kpts[5, 2]+kpts[6, 2])/2, name='torso', parent='hip'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[0, :2]             ], c=kpts[0, 2], name='neck'          , parent='torso'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[6, :2]             ], c=kpts[6, 2], name='right_shoulder', parent='torso'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[8, :2]             ], c=kpts[8, 2], name='right_elbow'   , parent='right_shoulder'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[10, :2]            ], c=kpts[10, 2], name='right_hand'    , parent='right_elbow'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[5, :2]             ], c=kpts[5, 2], name='left_shoulder' , parent='torso'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[7, :2]             ], c=kpts[7, 2], name='left_elbow'    , parent='left_shoulder'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[9, :2]             ], c=kpts[9, 2], name='left_hand'     , parent='left_elbow'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[12, :2]            ], c=kpts[12, 2], name='right_hip'     , parent='root'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[14, :2]            ], c=kpts[14, 2], name='right_knee'    , parent='right_hip'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[16, :2]            ], c=kpts[16, 2], name='right_foot'    , parent='right_knee'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[11, :2]            ], c=kpts[11, 2], name='left_hip'      , parent='root'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[13, :2]            ], c=kpts[13, 2], name='left_knee'     , parent='left_hip'))
        self.character_joint_keypoints.append(Keypoint([round(x) for x in kpts[15, :2]            ], c=kpts[15, 2], name='left_foot'     , parent='left_knee'))

        _dict = {}
        for k in self.character_joint_keypoints:
            _dict[k.name] = k

        # set parents and children
        for k in self.character_joint_keypoints:
            parent_str = k.parent
            if parent_str is not None:
                k.parent = _dict[parent_str]
                _dict[parent_str].children.append(k)

    """ Begin Foreground Background Specific Functionalities """
    def present_fresh_fbseg_stage(self):

        self.fbseg_info = FBSegInfo(self.content, self)
        self.fbseg_info.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.S))

        self.fbseg_canvas = FBSegCanvas(self.content, self, image_rgb=self.image_cv2)
        self.fbseg_canvas.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E))

        self.sam.reset()

        self.clear_fb_mask()

        self.clear_fb_sam_points()
        self.set_sam_points_to_ad_keypoints()

    def set_sam_points_to_ad_keypoints(self):
        self.fb_sam_points = []
        for kpt in self.character_joint_keypoints:
            self.add_sam_point(kpt.x, kpt.y, True)

    def add_sam_point(self, x, y, label: bool):
        sam_point = SAMPoint(x, y, label)
        self.fb_sam_points.append(sam_point)
        self.fbseg_canvas.add_oval_from_sam_point(sam_point)

    def clear_fb_sam_points(self):
        """ Removes all SAM points and instructs fbseg_canvas to remove SAM ovals """
        for sam_point in self.fb_sam_points:
            if sam_point.oval_id is not None:
                self.fbseg_canvas.delete(sam_point.oval_id)
            del sam_point
        self.fb_sam_points = []

    def clear_fb_mask(self):
        self.fb_mask = None
        self.fbseg_canvas.clear_mask()

    def set_fbseg_mask_from_sam(self):
        self.fb_mask = self.query_sam(self.fb_sam_points)
        self.fbseg_canvas.set_mask(self.fb_mask)

    """ Begin Mask Split Specific Functionalities """
    def present_fresh_mask_split_stage(self):

        self.delete_all_parts()
        self.create_new_part(name='Full Character', is_external=True, mask=self.fb_mask)

        self.mask_split_info = MaskSplitInfo(self.content, self)
        self.mask_split_info.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.S))

        self.mask_split_canvas = MaskSplitCanvas(self.content, self, image_rgb=self.image_cv2)
        self.mask_split_canvas.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E))

        self.update_visible_treeview()

    """ Begin Part Specific Functionalities """
    def present_fresh_parts_stage(self):
        self.sam.reset()

        self.parts_info = PartsInfo(self.content, self)
        self.parts_info.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.S))

        self.parts_canvas = PartsCanvas(self.content, self, image_rgb=self.image_cv2)
        self.parts_canvas.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E))

        self.parts_canvas.set_fbseg_mask(self.fb_mask)

        self.update_visible_treeview()

    def add_sam_point_to_active_part(self, x: int, y: int, label: bool):
        sam_point = SAMPoint(x, y, label)
        self.active_part.add_sam_point(sam_point)
        self.parts_canvas.add_oval_from_sam_point(sam_point)

    def update_active_part(self,
                           part_name: str,
                           parent_name: str,
                           does_rdtwp: bool,
                           hide_on_backside: bool,
                           hide_outside_parent: bool,
                           rotation_drives_flip: bool,
                           flip_as_drawn: str,
                           ):

        self.active_part.name = part_name
        self.active_part.parent_name = parent_name
        self.active_part.does_rdtwp = does_rdtwp
        self.active_part.hide_on_backside = hide_on_backside
        self.active_part.hide_outside_parent = hide_outside_parent
        self.active_part.rotation_drives_flip = rotation_drives_flip
        self.active_part.flip_as_drawn = flip_as_drawn

        self.parts_info.update_part(self.active_part)

        self.update_visible_treeview()

    def set_part_mask_from_sam(self):
        mask = self.query_sam(self.active_part.sam_points)
        self.active_part.set_mask(mask)

        self.parts_canvas.set_parts_seg_mask(self.active_part.mask)

        self.parts_info.update_part(self.active_part)

    def clear_active_part_sam_points(self):
        for sam_point in self.active_part.sam_points:
            self.parts_canvas.delete(sam_point.oval_id)
        self.active_part.sam_points = []


if __name__ == '__main__':
    AnnotationTool().start()
