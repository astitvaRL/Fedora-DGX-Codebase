from collections import defaultdict
import tkinter as tk
from tkinter import ttk
from typing import List
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageTk
from keypoint import Keypoint


class PoseCanvas(tk.Canvas):

    class Oval():
        def __init__(self):
            pass

        outline = 'black'
        color_deactive = 'red'
        color_active = 'green'

    def remove_keypoint_widgets(self):
        for oval_id in self.ovals.values():
            self.delete(oval_id)
        self.ovals = {}
        self.active_oval_id_var.set(-1)

        self.tool_keypoints = None
        self.kpt_name2ovalid = {}
        self.ovalid2kpt_idx = {}

        for line_id in self.lines:
            self.delete(line_id)
        self.lines = []
        self.child_oval_to_line = {}
        self.parent_oval_to_lines = defaultdict(list)

    def __init__(self, parent_widget: ttk.Frame, image_rgb: npt.NDArray[np.uint8], active_oval_id_var: tk.IntVar):
        super().__init__(parent_widget, width=image_rgb.shape[1], height=image_rgb.shape[0])

        self.image_rgb = image_rgb
        self.image_pil = Image.fromarray(self.image_rgb)
        self.image_tk = ImageTk.PhotoImage(image=self.image_pil)
        self.create_image(0, 0, image=self.image_tk, anchor='nw')

        self.o_rad = 4  # oval radius
        self.active_oval_id_var: ttk.IntVar = active_oval_id_var
        self.active_oval_id_var.trace('w', self.update_active_oval)

        self.ovals = {}
        self.lines = []

        # keeps track for dragging these things
        self.dragging = False
        self.drag_data = {'last_x': 0, 'last_y': 0}

    def savePosn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def addLine(self, event):
        self.create_line((self.lastx, self.lasty, event.x, event.y))
        self.savePosn(event)

    def set_ad_keypoints(self, keypoints: List[Keypoint]):
        self.remove_keypoint_widgets()  # remove existing keypoints if exist
        self.tool_keypoints = keypoints  # pointer to where this is stored by the tool

        # create ovals
        for idx, k in enumerate(keypoints):

            oval_id = self.create_oval(k.x-4, k.y-4, k.x+4, k.y+4, fill=self.Oval.color_deactive, outline=self.Oval.outline)

            self.tag_bind(oval_id, "<ButtonPress-1>", lambda event, oval_id=oval_id: self.on_oval_press(event, oval_id))
            self.tag_bind(oval_id, "<ButtonRelease-1>", lambda event, oval_id=oval_id: self.on_oval_release(event, oval_id))
            self.tag_bind(oval_id, "<B1-Motion>", lambda event, oval_id=oval_id: self.on_oval_motion(event, oval_id))

            self.kpt_name2ovalid[k.name] = oval_id
            self.ovalid2kpt_idx[oval_id] = idx

            self.ovals[idx] = oval_id

        # create lines between ovals
        for idx, k in enumerate(keypoints):
            if k.parent is None:
                continue

            cx, cy = self.coords(self.kpt_name2ovalid[k.name])[:2]  # child (self) oval coordinates
            px, py = self.coords(self.kpt_name2ovalid[k.parent.name])[:2]  # parent oval coordinates

            line_id = self.create_line(cx+self.o_rad, cy+self.o_rad, px+self.o_rad, py+self.o_rad)

            self.lines.append(line_id)
            child_oval_id = self.kpt_name2ovalid[k.name]
            parent_oval_id = self.kpt_name2ovalid[k.parent.name]

            self.child_oval_to_line[child_oval_id] = line_id
            self.parent_oval_to_lines[parent_oval_id].append(line_id)

        # position the ovals on top of the lines
        for oval_id in self.ovals.values():
            self.lift(oval_id)

    def on_oval_press(self, event, oval_id):
        self.dragging = True

        for key, val in self.ovals.items():
            if oval_id == val:
                self.active_oval_id_var.set(key)
                break

        self.drag_data['last_x'], self.drag_data['last_y'] = event.x, event.y

    def on_oval_release(self, event, oval_id):
        self.dragging = False
        self.drag_data['last_x'], self.drag_data['last_y'] = 0, 0

        kpt_idx : Keypoint = self.ovalid2kpt_idx[oval_id]
        x, y = self.coords(oval_id)[:2]
        self.tool_keypoints[kpt_idx].x = x + self.o_rad
        self.tool_keypoints[kpt_idx].y = y + self.o_rad

    def on_oval_motion(self, event, oval_id):
        if not self.dragging:
            return
        # Calculate the distance the mouse has moved
        dx = event.x - self.drag_data['last_x']
        dy = event.y - self.drag_data['last_y']

        # move the oval
        self.move(oval_id, dx, dy)

        # move the child line
        if oval_id in self.child_oval_to_line.keys():  # evalutes false for root
            child_line_id = self.child_oval_to_line[oval_id]
            x1, y1, x2, y2 = self.coords(child_line_id)
            self.coords(child_line_id, x1+dx, y1+dy, x2, y2)

        # move the parent lines
        for parent_line_id in self.parent_oval_to_lines[oval_id]:
            x1, y1, x2, y2 = self.coords(parent_line_id)
            self.coords(parent_line_id, x1, y1, x2+dx, y2+dy)

        # Update the initial mouse position for the next motion event
        self.drag_data['last_x'], self.drag_data['last_y'] = event.x, event.y

    def update_active_oval(self, *args):
        for oval_id in self.ovals.values():
            self.itemconfigure(oval_id, fill=self.Oval.color_deactive)

        if self.active_oval_id_var.get() == -1:
            return

        self.itemconfigure(self.ovals[self.active_oval_id_var.get()], fill=self.Oval.color_active)
