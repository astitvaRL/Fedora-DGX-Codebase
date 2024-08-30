from tkinter import ttk

from part_frame import ExternalPartFrame
from treeview import Treeview


class MaskSplitInfo(ttk.Frame):

    def __init__(self, parent_widget: ttk.Frame, tool):
        super().__init__(parent_widget)
        self.tool = tool  # reference to parent AnnotationTool class

        self.treeview = Treeview(self, tool)
        self.treeview.grid(column=0, row=1, columnspan=3)
        self.treeview.grid()

        self.external_part_frame = ExternalPartFrame(self)
        self.external_part_frame.grid(column=0, row=3, columnspan=3)

