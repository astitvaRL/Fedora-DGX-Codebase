from __future__ import annotations
import tkinter as tk
import cv2
from tkinter import ttk
from typing import Optional, Dict, List, Tuple
from PIL import Image, ImageTk
import numpy as np
import numpy.typing as npt
from skimage import measure  # for converstion from binary masks to vector points
import shapely  # for shape analysis
import rasterio.features


class BasePart():
    def __init__(self):
        self.name = None
        self.parent_name = None

        self.mask = None
        self.mask_thumbnail: Image = None

        self.texture: npt.NDArray = None

        self._children_parts: List[BasePart] = []

    def set_texture(self, image):
        self.texture = image

    def set_name(self, name):
        self.name = name

    def set_parent_name(self, parent_name):
        self.parent_name = parent_name

    def set_mask(self, mask: npt.NDArray[np.bool_]):
        self.mask = mask

        black_img_rgb = np.full([*self.mask.shape, 3], 0, dtype=np.uint8)
        alpha_channel = (~self.mask).astype(np.uint8) * 180
        black_img_rgba = np.dstack((black_img_rgb, alpha_channel))
        self.mask_thumbnail = Image.fromarray(black_img_rgba)
        self.mask_thumbnail.thumbnail((32, 32), Image.Resampling.LANCZOS)

    def add_child(self, part):
        self._children_parts.append(part)

    def get_children(self):
        return self._children_parts


class ExternalPart(BasePart):
    def __init__(self):
        super().__init__()

        self.top_split: Optional[Tuple[int, int, int, int]] = None  # the line which has split this external part from the external part above it
        self.bottom_split: Optional[Tuple[int, int, int, int]] = None  # the line which has split this external part from the external part below it

        self.forward_orientation = "None"  # ['None', 'DLeft', 'DRight']

    def set_forward_orientation(self, orientation: str):
        assert orientation in ['None', 'DLeft', 'DRight']
        self.forward_orientation = orientation

    def set_top_split(self, line: Tuple[int, int, int, int]):
        self.top_split = line

    def set_bottom_split(self, line: Tuple[int, int, int, int]):
        self.bottom_split = line

    def get_json_representation(self):
        return {
            'Type': 'External',
            'Name': self.name,
            'AreaMask': self.name,
            'ParentName': self.parent_name,
            'ForwardOrientation': self.forward_orientation,
            'BottomSplit': self.bottom_split,
            'TopSplit': self.top_split
        }


class InternalPart(BasePart):

    name_to_part: Dict[str, InternalPart] = {}  # mapping from name to Part. Populated in the 'process_for_review' method

    def __init__(self,  image: npt.NDArray[np.uint8], name: str = "", editable: bool = True, thumbnail: Optional[Image.Image] = None):
        super().__init__()

        self.name = name
        self.editable = editable

        self.full_image: npt.NDArray[np.uint8] = image

        self.mask: npt.NDArray[np.bool_] = np.full(self.full_image.shape[:2], True, dtype=np.bool_)

        self.thumbnail = thumbnail

        self.parent_name: Optional[str] = None

        self.does_rdtwp: bool = False  # does Rotation Drive Translation Within Parent?

        self.hide_on_backside: bool = False  # should this be hid when character is turned around?

        self.hide_outside_parent: bool = False  # if the bodypart is adjusted and part now falls outside its parent's area, should we hide that part?

        self.rotation_drives_flip: bool = False  # should this body part flip as a function of viewing angle?

        self.flip_as_drawn: str = ""  # is this drawn facing towards drawing left or drawing right?

        self.children_parts: List[InternalPart] = []

        # geometries needed to display the part
        self.polygons: List[shapely.Polygon]
        self.center_point: shapely.Point
        self.bounding_box: npt.NDArray[np.uint8]

        # the infill used to hide holes in ancestor parts
        self.infill_texture_rgba: npt.NDArray[np.uint8]

        self.texture_rgba_tk: Optional[ImageTk.PhotoImage] = None
        self.texture_rgba_tk_review_canvas_id: int = -1

        # view dependent transforms
        self.transforms: Dict[str, npt.NDArray] = {
            'left': np.diag([1, 1, 1]).astype(np.int16),
            'right': np.diag([1, 1, 1]).astype(np.int16),
            'center': np.diag([1, 1, 1]).astype(np.int16)
        }

    def has_own_mask(self):
        return len(self.mask[self.mask == True]) != 0  # noqa: E712

    def _create_display_geometry_from_mask(self):
        # using the mask, create contours
        contours: List[npt.NDArray[np.float64]] = measure.find_contours(255 * self.mask.astype(np.uint8), 128)

        self.polygons = []
        for contour in contours:
            self.polygons.append(shapely.Polygon(contour[:, [1, 0]]))  # swap x and y columns

        # calculate bounding box
        miny, minx = self.full_image.shape[:2]
        maxy, maxx = 0, 0
        for p in self.polygons:
            minx_, miny_, maxx_, maxy_ = p.bounds
            miny = min(miny, miny_)
            minx = min(minx, minx_)
            maxy = max(maxy, maxy_)
            maxx = max(maxx, maxx_)
        minx = round(max(0, minx))
        miny = round(max(0, miny))
        maxx = round(min(self.full_image.shape[1], maxx))
        maxy = round(min(self.full_image.shape[0], maxy))
        self.bounding_box = [minx, miny, maxx, maxy]

        if self.flip_as_drawn == "":
            # if part does not flip left/right as a function of viewing angle,
            # centroid is area-weighted average of all polygons
            weighted_centroids = []
            area_sum = sum([p.area for p in self.polygons])
            for p in self.polygons:
                weight = p.area / area_sum
                weighted_centroids.append(weight * np.array(p.centroid.xy))
            center_point = shapely.Point(np.sum(weighted_centroids, axis=0))
        elif self.flip_as_drawn == "Drawing Left":
            # if it points left, center point is on the right side of part, midway between top and bottom
            x = self.bounding_box[2]  # maxx
            y = round((self.bounding_box[1] + self.bounding_box[3]) / 2)  # maxy
            center_point = shapely.Point(x, y)
        elif self.flip_as_drawn == "Drawing Right":
            # if it points left, center point is on the right side of part, midway between top and bottom
            x = self.bounding_box[0]  # minx
            y = round((self.bounding_box[1] + self.bounding_box[3]) / 2)  # maxy
            center_point = shapely.Point(x, y)
        else:
            assert False, f'unknown flip_as_drawn value: {self.flip_as_drawn}'

        self.center_point = center_point

        self.part_to_be_infilled = np.full(self.full_image.shape[:2], False)

        # calculate the billboard part of the part that we will move around
        self.texture_rgba = np.zeros([maxy-miny, maxx-minx, 4], dtype=np.uint8)
        self.texture_rgba[:, :, :3] = self.full_image[miny:maxy, minx:maxx]
        self.texture_rgba[:, :, 3] = 255 * rasterio.features.rasterize(self.polygons, self.full_image.shape[:2])[miny:maxy, minx:maxx]
        self.texture_rgba_tk = ImageTk.PhotoImage(Image.fromarray(self.texture_rgba))

        if self.flip_as_drawn in ['Drawing Left', 'Drawing Right']:
            self.flip_texture_rgba = self.texture_rgba[:, ::-1, :].copy()

    def _create_display_geometry_from_children(self):
        assert self.has_own_mask() is False, 'if part has mask with area, that should be used to create geometry'

        for part in self.children_parts:
            part.create_display_geometry()

        x = np.mean([p.center_point.x for p in self.children_parts])
        y = np.mean([p.center_point.y for p in self.children_parts])
        self.center_point = shapely.Point(x, y)

        minx = int(np.min([p.bounding_box[0] for p in self.children_parts]))
        miny = int(np.min([p.bounding_box[1] for p in self.children_parts]))
        maxx = int(np.min([p.bounding_box[2] for p in self.children_parts]))
        maxy = int(np.min([p.bounding_box[3] for p in self.children_parts]))
        self.bounding_box = [minx, miny, maxx, maxy]

        self.texture_rgba = None
        self.texture_rgba_tk = None

    def create_display_geometry(self):
        if self.has_own_mask():
            self._create_display_geometry_from_mask()
        else:
            self._create_display_geometry_from_children()

    def compute_horizontal_intersection_to_parent_boundary_point(self, parent_part: InternalPart, left: bool = True):
        if left:
            p2x: int = 0
        else:
            p2x: int = self.full_image.shape[1]

        p1 = self.center_point
        p2 = shapely.Point(p2x, self.center_point.y)
        line_left = shapely.LineString([p1, p2])
        if len(parent_part.polygons) > 1:
            assert False, 'attempted intersection on parent_part with more than one polygon'
        intersection = line_left.intersection(parent_part.polygons[0].boundary)

        if intersection.is_empty:
            print("No intersection found")
            assert False
        else:
            # If there is an intersection, get the first point
            if intersection.geom_type == "Point":
                return intersection
            elif intersection.geom_type == "MultiPoint":
                intersection_points = shapely.MultiPoint(intersection)
                return intersection_points[0]

    def create_view_transform(self, view: str, parent_part: InternalPart):

        transform = np.diag([1, 1, 1]).astype(np.int16)  # identity transform

        if self.does_rdtwp is True:  # translate if rotation drives transflation within parent
            if view == 'left':
                intersection_point = self.compute_horizontal_intersection_to_parent_boundary_point(parent_part, left=True)
                transform[0, 2] = round(intersection_point.x - self.center_point.x)
            elif view == 'right':
                intersection_point = self.compute_horizontal_intersection_to_parent_boundary_point(parent_part, left=False)
                transform[0, 2] = round(intersection_point.x - self.center_point.x)
            elif view == 'center':
                assert 'left' in self.transforms.keys() and 'right' in self.transforms.keys(), 'left and right view not defined before center'
                transform[0, 2] = round((self.transforms['left'][0, 2] + self.transforms['right'][0, 2]) / 2)

        self.transforms[view] = transform

    def update_image_tk(self, view: str):
        print('skipping update image tk (alpha)')
        return
        if self.texture_rgba is None:
            return

        if view == 'left':
            self.texture_rgba[:, :, 3] = 255 * self.left_alpha.astype(np.uint8)
        elif view == 'right':
            self.texture_rgba[:, :, 3] = 255 * self.right_alpha.astype(np.uint8)
        elif view == 'center':
            self.texture_rgba[:, :, 3] = 255 * self.center_alpha.astype(np.uint8)
        elif view == 'as_drawn':
            minx, miny, maxx, maxy = self.bounding_box
            self.texture_rgba[:, :, 3] = 255 * self.mask[miny:maxy, minx:maxx]
        self.texture_rgba_tk = ImageTk.PhotoImage(Image.fromarray(self.texture_rgba))

    def get_json_representation(self):
        return {
            'Type': 'Internal',
            'Name': self.name,
            'RotationDrivesTranslationWithinParent': self.does_rdtwp,
            'AreaMask': self.name,
            'HideOnBackside': self.hide_on_backside,
            'HideOutsideParent': self.hide_outside_parent,
            'RotationDrivesFlip': self.rotation_drives_flip,
            'FlipAsDrawn': self.flip_as_drawn,
            'ParentName': self.parent_name
        }

    @staticmethod
    def preprocess_for_review(all_parts: List[InternalPart]):
        """ Takes in all of the parts from the Parts step and parses the hierarchy in preparation for Review step"""

        # create mapping from name to part
        InternalPart.name_to_part: Dict[str, InternalPart] = {}
        for part in all_parts:
            InternalPart.name_to_part[part.name] = part

        # ensure any children parts from a previous function call are removed:
        for part in all_parts:
            part.children_parts = []

        # parse the part tree
        root_part: InternalPart = None
        for part in all_parts:
            if part.parent_name is not None:
                InternalPart.name_to_part[part.parent_name].children_parts.append(part)
            else:
                if root_part is not None:
                    assert False, 'there is more than one part without a parent'
                root_part = part

        # create geometric data structures needed once mask is finalized
        for part in all_parts:
            part.create_display_geometry()

        # recurse on children
        InternalPart.recurse_create_views(root_part, None)

        # recurse use infillings to modify other parts textures
        InternalPart.determine_pixels_to_be_infilled(root_part)

    @staticmethod
    def determine_pixels_to_be_infilled(part):
        """ use part infillings to modify textures of parents so there's no weird hole when they move """

        # start with the most nested parts and work backwards to the full character
        for child_part in part.children_parts:
            InternalPart.determine_pixels_to_be_infilled(child_part)

        # if there's no area to this part, nothing to infill
        if not part.has_own_mask():
            return

        # get the boundaries of the part, in image-relative coordinates
        l, t, r, b = part.bounding_box

        # get the mask of the part, cropped with its bounding box
        local_mask = part.mask[t:b, l:r]

        # go up the ancestral path, modifying texture with infill as we go
        ancestor_part = part  # start with own part
        while ancestor_part.parent_name is not None:
            ancestor_part = InternalPart.name_to_part[ancestor_part.parent_name]  # get next ancestor

            if not ancestor_part.has_own_mask():  # no need to infill anything if ancestor has no visible pixles
                continue

            ancestor_part.part_to_be_infilled[t:b, l:r] = np.logical_or(ancestor_part.part_to_be_infilled[t:b, l:r], local_mask)

            if False:
                # get the pixels contained by ancestor part using child part's bounding box
                ancestor_local_mask = ancestor_part.mask[t:b, l:r]

                # find overlap between child and it's ancestor
                overlap = np.logical_and(local_mask, ancestor_local_mask)

                # figure out where  in ancestor's texture to insert fill
                al, at, ar, ab = ancestor_part.bounding_box
                arl = l - al  # ancestor relative left
                art = t - at  # ancestor relative top
                arr = r - al  # ancestor relative right
                arb = b - at  # ancestor relative bottom

                # update ancestor's texture using child part's infill, masked to only edit visible pixels shared by both
                ancestor_part.texture_rgba[art:arb, arl:arr][overlap] = part.infill_texture_rgba[overlap]

    @staticmethod
    def recurse_create_views(part: InternalPart, parent_part: Optional[InternalPart]):
        part.create_view_transform('left', parent_part)
        part.create_view_transform('right', parent_part)
        part.create_view_transform('center', parent_part)  # ensure left and right are calculated first

        for child_part in part.children_parts:

            child_part.create_view_transform('left', parent_part)
            child_part.create_view_transform('right', parent_part)
            child_part.create_view_transform('center', parent_part)  # ensure left and right are calculated first
            # add other calculations needed here

            InternalPart.recurse_create_views(child_part, part)


class InternalPartFrame(ttk.Frame):

    def __init__(self, parent_widget):

        super().__init__(parent_widget, style="PartFrame.TFrame")
        self['borderwidth'] = 2
        self['relief'] = 'sunken'

        self.pose_info_frame = parent_widget  # reference to parent AnnotationTool class

        self.name_label = ttk.Label(self, text='Name')
        self.name_label.grid(column=0, row=0)

        self.name_text = tk.Text(self, width=20, height=1)
        self.name_text.grid(column=1, row=0)

        self.is_editable_label = ttk.Label(self, text='Is Editable?')
        self.is_editable_label.grid(column=0, row=1)

        self.is_editable_value = ttk.Label(self, text='')
        self.is_editable_value.grid(column=1, row=1)

        self.parent_label = ttk.Label(self, text='Parent')
        self.parent_label.grid(column=0, row=2)

        self.parent_value = ttk.Combobox(self, state="readonly")
        self.parent_value.grid(column=1, row=2)

        self.does_rdtwp_label = ttk.Label(self, text='Rotation Drives Translation Within Parent?')
        self.does_rdtwp_label.grid(column=0, row=3)

        self.does_rdtwp_variable = tk.BooleanVar(False)
        self.does_rdtwp_value = ttk.Checkbutton(self, variable=self.does_rdtwp_variable)
        self.does_rdtwp_value.grid(column=1, row=3)

        query_sam_button = ttk.Button(self, text='Query SAM', command=self.query_sam)
        query_sam_button.grid(column=0, row=4)
        clear_points_button = ttk.Button(self, text='Clear Points', command=self.clear_points)
        clear_points_button.grid(column=1, row=4)
        dilate_mask_button = ttk.Button(self, text='Dilate Mask', command=self.dilate_mask)
        dilate_mask_button.grid(column=2, row=4)
        erode_mask_button = ttk.Button(self, text='Erode Mask', command=self.erode_mask)
        erode_mask_button.grid(column=3, row=4)

        self.hide_on_backside_label = ttk.Label(self, text='Hide on backside?')
        self.hide_on_backside_label.grid(column=0, row=5)
        self.hide_on_backside_variable = tk.BooleanVar(False)
        self.hide_on_backside_value = ttk.Checkbutton(self, variable=self.hide_on_backside_variable)
        self.hide_on_backside_value.grid(column=1, row=5)

        self.hide_outside_parent_label = ttk.Label(self, text='Hide when outside parent?')
        self.hide_outside_parent_label.grid(column=0, row=6)
        self.hide_outside_parent_variable = tk.BooleanVar(False)
        self.hide_outside_parent_value = ttk.Checkbutton(self, variable=self.hide_outside_parent_variable)
        self.hide_outside_parent_value.grid(column=1, row=6)

        self.rotation_drives_flip_label = ttk.Label(self, text='Rotation drive flip?')
        self.rotation_drives_flip_label.grid(column=0, row=7)
        self.rotation_drives_flip_variable = tk.BooleanVar(False)
        self.rotation_drives_flip_value = ttk.Checkbutton(self, variable=self.rotation_drives_flip_variable)
        self.rotation_drives_flip_value.grid(column=1, row=7)

        self.flip_direction_as_drawn_combobox = ttk.Combobox(self, state="readonly")
        self.flip_direction_as_drawn_combobox['values'] = ['', 'Drawing Left', 'Drawing Right']
        self.flip_direction_as_drawn_combobox.grid(column=2, row=7)

        save_changes_button = ttk.Button(self, text='Save Changes', command=self.save_changes)
        save_changes_button.grid(column=0, row=9, columnspan=2)

    def set_local_flip_axis(self):
        self.pose_info_frame.tool.active_part.rotation_drives_flip_axis = []
        self.pose_info_frame.tool.parts_canvas.state = 'flip_axis'
        print('set canvas state to flip_axis')

    def save_changes(self):
        if self.pose_info_frame.tool.active_part is None or self.pose_info_frame.tool.active_part.editable is False:
            return

        part_name: str = self.name_text.get("1.0", "end").strip()
        parent_name = None if self.parent_value.get() == "" else self.parent_value.get()
        does_rdtwp = self.does_rdtwp_variable.get()
        hide_on_backside = self.hide_on_backside_variable.get()
        hide_outside_parent = self.hide_outside_parent_variable.get()
        rotation_drives_flip = self.rotation_drives_flip_variable.get()
        flip_as_drawn = self.flip_direction_as_drawn_combobox.get()
        self.pose_info_frame.tool.update_active_part(
            part_name=part_name,
            parent_name=parent_name,
            does_rdtwp=does_rdtwp,
            hide_on_backside=hide_on_backside,
            hide_outside_parent=hide_outside_parent,
            rotation_drives_flip=rotation_drives_flip,
            flip_as_drawn=flip_as_drawn
        )

    def dilate_mask(self):
        kernel = np.ones((3, 3), np.uint8)
        tool = self.pose_info_frame.tool
        mask = cv2.dilate(tool.active_part.mask.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
        tool.active_part.set_mask(mask)

        tool.parts_canvas.set_parts_seg_mask(tool.active_part.mask)
        tool.parts_info.update_part(tool.active_part)

    def erode_mask(self):
        kernel = np.ones((3, 3), np.uint8)
        tool = self.pose_info_frame.tool
        mask = cv2.erode(tool.active_part.mask.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
        tool.active_part.set_mask(mask)

        tool.parts_canvas.set_parts_seg_mask(tool.active_part.mask)
        tool.parts_info.update_part(tool.active_part)

    def clear_points(self):
        if self.pose_info_frame.tool.active_part is None or self.pose_info_frame.tool.active_part.editable is False:
            return
        self.pose_info_frame.tool.clear_active_part_sam_points()

    def query_sam(self):
        if self.pose_info_frame.tool.active_part is None or self.pose_info_frame.tool.active_part.editable is False:
            return
        self.pose_info_frame.tool.set_part_mask_from_sam()

    def display_part_info(self, part: InternalPart):

        """ clear existing info """
        self.name_text.delete("1.0", "end")

        self.is_editable_value.config(text="")

        self.parent_value['values'] = []
        self.parent_value.set("")

        self.flip_direction_as_drawn_combobox.set("")

        self.does_rdtwp_variable.set(False)
        self.hide_on_backside_variable.set(False)
        self.hide_outside_parent_variable.set(False)
        self.rotation_drives_flip_variable.set(False)

        """ populate with info of new part """
        if part is not None:
            self.name_text.insert("1.0", part.name)
            self.is_editable_value.config(text=str(part.editable))

            self.parent_value['values'] = [p.name for p in self.pose_info_frame.tool.parts if p is not part]
            if part.parent_name is None:
                self.parent_value.set("")
            else:
                self.parent_value.set(part.parent_name)

            self.does_rdtwp_variable.set(part.does_rdtwp)
            self.hide_on_backside_variable.set(part.hide_on_backside)
            self.hide_outside_parent_variable.set(part.hide_outside_parent)
            self.rotation_drives_flip_variable.set(part.rotation_drives_flip)
            self.flip_direction_as_drawn_combobox.set(part.flip_as_drawn)
