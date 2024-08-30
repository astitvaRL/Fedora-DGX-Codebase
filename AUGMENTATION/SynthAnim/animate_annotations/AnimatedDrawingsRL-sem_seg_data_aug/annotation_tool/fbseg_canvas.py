import tkinter as tk
from tkinter import ttk
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageTk
from sam_point import SAMPoint


class FBSegCanvas(tk.Canvas):

    def __init__(self, parent_widget: ttk.Frame, tool, image_rgb: npt.NDArray[np.uint8]):

        super().__init__(parent_widget, width=image_rgb.shape[1], height=image_rgb.shape[0])

        self.tool = tool

        self.bind("<Button-1>", lambda e: self.add_sam_point(e, True))
        self.bind("<Button-2>", lambda e: self.add_sam_point(e, False))

        image_pil = Image.fromarray(image_rgb)
        self.image_tk = ImageTk.PhotoImage(image=image_pil)
        self.create_image(0, 0, image=self.image_tk, anchor='nw')

        self.o_rad = 2

        self.mask_id = None
        self.mask_image_tk = None

        self.point_ovals = []

    def add_sam_point(self, event, label: bool):
        self.tool.add_sam_point(event.x, event.y, label)

    def set_mask(self, fb_mask: npt.NDArray[np.bool_]):
        """ Takes in the foreground/background segmentation from the parent tool.
        Computes the new mask, deletes the previous one and applies it."""

        black_img_rgb = np.full([*fb_mask.shape, 3], 0, dtype=np.uint8)
        alpha_channel = (~fb_mask).astype(np.uint8) * 180
        black_img_rgba = np.dstack((black_img_rgb, alpha_channel))
        image_pil = Image.fromarray(black_img_rgba)
        image_tk = ImageTk.PhotoImage(image=image_pil)

        self.clear_mask()

        self.mask_id = self.create_image(0, 0, image=image_tk, anchor='nw')
        self.mask_image_tk = image_tk

    def clear_mask(self):
        self.delete(self.mask_id)
        self.delete(self.mask_image_tk)

    def reset(self):
        pass

    def add_oval_from_sam_point(self, sam_point: SAMPoint):
        if sam_point.label:
            color = 'red'
        else:
            color = 'blue'
        oval_id = self.create_oval(sam_point.x-self.o_rad, sam_point.y-self.o_rad, sam_point.x+self.o_rad, sam_point.y+self.o_rad, fill=color, outline='black')
        sam_point.oval_id = oval_id
