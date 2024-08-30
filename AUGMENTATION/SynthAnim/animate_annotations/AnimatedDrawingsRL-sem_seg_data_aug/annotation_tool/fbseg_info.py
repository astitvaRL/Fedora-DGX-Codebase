from tkinter import ttk
import cv2
import numpy as np


class FBSegInfo(ttk.Frame):

    def __init__(self, parent_widget: ttk.Frame, tool):
        super().__init__(parent_widget)
        self.tool = tool  # reference to parent AnnotationTool class

        query_sam_button = ttk.Button(self, text='Query SAM', command=self.query_sam)
        query_sam_button.grid(column=0, row=0)

        clear_points_button = ttk.Button(self, text='Clear Points', command=self.clear_points)
        clear_points_button.grid(column=0, row=1)

        clear_mask_button = ttk.Button(self, text='Clear Mask', command=self.clear_mask)
        clear_mask_button.grid(column=0, row=2)

        export_mask_button = ttk.Button(self, text='Export Mask', command=self.export_mask)
        export_mask_button.grid(column=0, row=3)

    def export_mask(self):
        cv2.imwrite('dev_fb_seg.png', 255 * np.uint8(self.fb_mask))

    def query_sam(self):
        """ Directs tool to query SAM"""
        self.tool.set_fbseg_mask_from_sam()

    def clear_points(self):
        """ Directs tool to remove all SAM points"""
        self.tool.clear_fb_sam_points()

    def clear_mask(self):
        self.tool.clear_fb_mask()
