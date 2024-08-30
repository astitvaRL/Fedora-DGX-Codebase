import tkinter as tk
from tkinter import ttk
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageTk
from sam_point import SAMPoint


class PartsCanvas(tk.Canvas):

    def __init__(self, parent_widget: ttk.Frame, tool, image_rgb: npt.NDArray[np.uint8]):

        super().__init__(parent_widget, width=image_rgb.shape[1], height=image_rgb.shape[0])

        self.bind("<Button-1>", self.left_click)
        self.bind("<Button-2>", self.right_click)

        self.state = None  # None means to add sam points, axis[0|1] means we're adding flip axes

        self.tool = tool

        image_pil = Image.fromarray(image_rgb)
        self.image_tk = ImageTk.PhotoImage(image=image_pil)
        self.create_image(0, 0, image=self.image_tk, anchor='nw')

        self.o_rad = 2

        self.fbseg_mask_id = None
        self.fbseg_mask_image_tk = None

        self.parts_seg_mask_id = None
        self.parts_seg_mask_image_tk = None

    def left_click(self, event):
        if self.state is None:
            self.add_sam_point(event, True)
        if self.state == 'flip_axis':
            self.tool.active_part.rotation_drives_flip_axis.append((event.x, event.y))

    def right_click(self, event):
        if self.state is None:
            self.add_sam_point(event, False)

    def add_sam_point(self, event, label):

        # if no active part is selected or we cannot modify the part, do nothing
        if self.tool.active_part is None or self.tool.active_part.editable is False:
            return

        self.tool.add_sam_point_to_active_part(event.x, event.y, label)

    def set_fbseg_mask(self, fbseg_mask: npt.NDArray[np.bool_]):

        self.delete(self.fbseg_mask_id)
        self.delete(self.fbseg_mask_image_tk)

        black_img_rgb = np.full([*fbseg_mask.shape, 3], 0, dtype=np.uint8)
        alpha_channel = (~fbseg_mask).astype(np.uint8) * 180
        black_img_rgba = np.dstack((black_img_rgb, alpha_channel))
        image_pil = Image.fromarray(black_img_rgba)
        image_tk = ImageTk.PhotoImage(image=image_pil)

        self.fbseg_mask_id = self.create_image(0, 0, image=image_tk, anchor='nw')
        self.fbseg_mask_image_tk = image_tk

    def set_parts_seg_mask(self, parts_seg_mask: npt.NDArray[np.bool_]):

        self.delete(self.parts_seg_mask_id)
        self.delete(self.parts_seg_mask_image_tk)

        black_img_rgb = np.full([*parts_seg_mask.shape, 3], 0, dtype=np.uint8)
        alpha_channel = (~parts_seg_mask).astype(np.uint8) * 180
        black_img_rgba = np.dstack((black_img_rgb, alpha_channel))
        image_pil = Image.fromarray(black_img_rgba)
        image_tk = ImageTk.PhotoImage(image=image_pil)

        self.parts_seg_mask_id = self.create_image(0, 0, image=image_tk, anchor='nw')
        self.parts_seg_mask_image_tk = image_tk

    def add_oval_from_sam_point(self, sam_point: SAMPoint):
        if sam_point.label:
            color = 'red'
        else:
            color = 'blue'
        oval_id = self.create_oval(sam_point.x-self.o_rad, sam_point.y-self.o_rad, sam_point.x+self.o_rad, sam_point.y+self.o_rad, fill=color, outline='black')
        sam_point.oval_id = oval_id
