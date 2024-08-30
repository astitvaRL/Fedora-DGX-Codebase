from typing import List
import tkinter as tk
from tkinter import ttk
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageTk
from parts import InternalPart


class ReviewCanvas(tk.Canvas):

    def __init__(self, parent_widget: ttk.Frame, tool, image_rgb: npt.NDArray[np.uint8]):

        super().__init__(parent_widget, width=image_rgb.shape[1], height=image_rgb.shape[0])

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)

        self.tool = tool

        self.image_rgb = image_rgb
        image_pil = Image.fromarray(self.image_rgb)
        self.image_tk = ImageTk.PhotoImage(image=image_pil)
        self.create_image(0, 0, image=self.image_tk, anchor='nw')

    def on_press(self, event):
        global start_x, start_y
        start_x = event.x
        start_y = event.y

    def on_drag(self, event):
        if self.tool.active_part is None:
            return

        global start_x, start_y
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        dx = x - start_x
        dy = y - start_y

        start_x, start_y = x, y

        # move this stuff to 'on press' where we can
        self.tool.active_part.transforms[self.tool.active_view][0, 2] += dx
        self.tool.active_part.transforms[self.tool.active_view][1, 2] += dy

        def _move_recursively(part, dx, dy):
            part_image_id = part.texture_rgba_tk_review_canvas_id
            self.move(part_image_id, dx, dy)
            for c_part in part.children_parts:
                _move_recursively(c_part, dx, dy)

        _move_recursively(self.tool.active_part, dx, dy)

    def show_view(self, view: str):

        self.create_image(0, 0, image=self.image_tk, anchor='nw')

        root_part =  self.tool.parts[0]
        if view == 'as_drawn':
            self.show_as_drawn_view_recurse(root_part)
        else:
            assert view in ['center', 'left', 'right']
            self.show_view_recurse(root_part, view, np.diag([1, 1, 1]), [], np.full(self.image_rgb.shape[:2], True, dtype=np.bool_))

    def show_view_recurse(self, part, view: str, parent_global_transform, parent_stack: List[InternalPart], part_visible_pixel_universe):  # , parent_part_stack: List[Part]):
        """
        Given a part, the view which is being shown, the global transform of the part's parent, and the visible pixel universe of the part, this creates the texture that should be shown
        and creates a tk image and inserts it into the canvas.
        """

        global_transform = parent_global_transform @ part.transforms[view]

        # if everything outside of the parent of this part must be hidden, then update part_visible_pixel_universe
        if part.hide_outside_parent is True:

            # get the bounds of the immediate parent within the current view
            parent_part = parent_stack[-1]
            parent_in_view_left: int = parent_part.bounding_box[0] + parent_global_transform[0, 2]
            parent_in_view_top = parent_part.bounding_box[1] + parent_global_transform[1, 2]
            parent_in_view_bottom = parent_in_view_top + parent_part.texture_rgba.shape[0]
            parent_in_view_right = parent_in_view_left + parent_part.texture_rgba.shape[1]

            # set visible_pixel_universe to True only where the parent part has visible pixels
            part_visible_pixel_universe = np.zeros(self.image_rgb.shape[:2])
            part_visible_pixel_universe[parent_in_view_top:parent_in_view_bottom, parent_in_view_left:parent_in_view_right] = parent_part.texture_rgba[:, :, -1]

        # create the infill and texture if needed
        if part.has_own_mask():

            as_drawn_left = part.bounding_box[0]
            as_drawn_top = part.bounding_box[1]
            as_drawn_right = as_drawn_left + part.texture_rgba.shape[1]
            as_drawn_bottom = as_drawn_top + part.texture_rgba.shape[0]

            parent_in_view_left = part.bounding_box[0] + global_transform[0, 2]
            parent_in_view_top = part.bounding_box[1] + global_transform[1, 2]
            parent_in_view_bottom = parent_in_view_top + part.texture_rgba.shape[0]
            parent_in_view_right = parent_in_view_left + part.texture_rgba.shape[1]

            # create the texture
            if part.texture_rgba_tk_review_canvas_id != -1:  # delete the old one if needed
                self.delete(part.texture_rgba_tk_review_canvas_id)

            # determines whether to present the flipped texture or not
            if (view == 'left' and part.flip_as_drawn == 'Drawing Right'):
                texture_rgba = part.flip_texture_rgba
                parent_in_view_left -= part.texture_rgba.shape[1]
                parent_in_view_right -= part.texture_rgba.shape[1]
                mask = part.mask[as_drawn_top:as_drawn_bottom, as_drawn_left:as_drawn_right][:, ::-1]
            elif (view == 'right' and part.flip_as_drawn == 'Drawing Left'):
                texture_rgba = part.flip_texture_rgba
                parent_in_view_left += part.texture_rgba.shape[1]
                parent_in_view_right += part.texture_rgba.shape[1]
                mask = part.mask[as_drawn_top:as_drawn_bottom, as_drawn_left:as_drawn_right][:, ::-1]
            else:
                texture_rgba = part.texture_rgba
                mask = part.mask[as_drawn_top:as_drawn_bottom, as_drawn_left:as_drawn_right]

            texture_rgba[:, :, -1] = 255 * np.logical_and(
                part_visible_pixel_universe[parent_in_view_top:parent_in_view_bottom, parent_in_view_left:parent_in_view_right],
                mask
            ).astype(np.uint8)
            part.texture_rgba_tk = ImageTk.PhotoImage(Image.fromarray(texture_rgba))
            part.texture_rgba_tk_review_canvas_id = self.create_image(parent_in_view_left, parent_in_view_top, image=part.texture_rgba_tk, anchor='nw')

        # recurse on children
        parent_stack.append(part)
        for child_part in part.children_parts:
            self.show_view_recurse(child_part, view, global_transform, parent_stack, part_visible_pixel_universe)
        parent_stack.pop(-1)

    def show_as_drawn_view_recurse(self, part):

        part.update_image_tk('as_drawn')

        left = part.bounding_box[0]
        top = part.bounding_box[1]
        if part.texture_rgba_tk_review_canvas_id != -1:  # if no image has been created on canvas yet
            self.delete(part.texture_rgba_tk_review_canvas_id)
        part.texture_rgba_tk_review_canvas_id = self.create_image(left, top, image=part.texture_rgba_tk, anchor='nw')

        for child_part in part.children_parts:
            self.show_as_drawn_view_recurse(child_part)
