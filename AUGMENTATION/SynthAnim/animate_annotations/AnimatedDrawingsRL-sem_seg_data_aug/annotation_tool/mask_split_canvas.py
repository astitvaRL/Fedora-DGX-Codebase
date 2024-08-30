import tkinter as tk
from tkinter import ttk
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageTk


class MaskSplitCanvas(tk.Canvas):

    def __init__(self, parent_widget: ttk.Frame, tool, image_rgb: npt.NDArray[np.uint8]):

        super().__init__(parent_widget, width=image_rgb.shape[1], height=image_rgb.shape[0])

        self.tool = tool

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)

        image_pil = Image.fromarray(image_rgb)
        self.image_tk = ImageTk.PhotoImage(image=image_pil)
        self.create_image(0, 0, image=self.image_tk, anchor='nw')

    def on_press(self, event):
        global line_id
        global part
        for _part in self.tool.parts:
            if _part.mask[event.y, event.x]:
                part = _part
                x1 = np.where(part.mask[event.y, :event.x] == False)[0][-1]  # first pixel left of click that is not in mask
                x2 = event.x + np.where(part.mask[event.y, event.x:] == False)[0][0]  # first pixel right of click that is not in mask
                y = event.y
                line_id = self.create_line(x1, y, x2, y, width=1, fill='blue')

    def on_drag(self, event):
        global line_id
        global part
        if part.mask[event.y, event.x]:
            x1 = np.where(part.mask[event.y, :event.x] == False)[0][-1]  # first pixel left of click that is not in mask
            x2 = event.x + np.where(part.mask[event.y, event.x:] == False)[0][0]  # first pixel right of click that is not in mask
            y = event.y
            self.coords(line_id, x1, y, x2, y)

    def on_release(self, event):
        global line_id
        global part

        if part.mask[event.y, event.x]:

            line = self.coords(line_id)

            _mask = part.mask.copy()

            # mask_b, which is BELOW this split, will be given to the new part
            mask_b = np.full(part.mask.shape, False, dtype=np.bool_)
            mask_b[event.y:, :] = _mask[event.y:, :]
            self.tool.create_new_part("", is_external=True, mask=mask_b)

            # if previous part had a bottom split already, give it to this new part
            if part.bottom_split:
                self.tool.parts[-1].set_bottom_split(part.bottom_split)
            # its top split will be the line
            self.tool.parts[-1].set_top_split(line)

            # mask_a, which is ABOVE the split, will be applied to the pre-existing part
            mask_a = np.full(part.mask.shape, False, dtype=np.bool_)
            mask_a[:event.y, :] = _mask[:event.y, :]
            part.set_mask(mask_a)
            # since this is top part, top split won't change. Only bottom
            part.set_bottom_split(line)

            self.tool.update_visible_treeview()
