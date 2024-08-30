from tkinter import ttk
from parts import InternalPart, InternalPartFrame
from treeview import Treeview


class PartsInfo(ttk.Frame):

    def __init__(self, parent_widget: ttk.Frame, tool):
        super().__init__(parent_widget)
        self.tool = tool  # reference to parent AnnotationTool class

        label = ttk.Label(self, text='Parts Stage')
        label.grid(column=0, row=0)

        # create treeview
        self.treeview = Treeview(self, tool)

        self.treeview.grid(column=0, row=1, columnspan=3)
        self.treeview.grid()

        # button to create a new body part
        new_part_button = ttk.Button(self, text='Create New Part', command=self.tool.create_new_part)
        new_part_button.grid(column=0, row=2)

        # button to export parts
        export_parts_button = ttk.Button(self, text='Export Parts', command=self.tool.export_parts)
        export_parts_button.grid(column=1, row=2)

        # button to load exported parts
        load_parts_export_button = ttk.Button(self, text='Load Parts Export', command=self.tool.load_parts_from_export)
        load_parts_export_button.grid(column=2, row=2)

        # part frame
        self.part_frame = InternalPartFrame(self)
        self.part_frame.grid(column=0, row=3, columnspan=3)

    def update_part(self, part_to_update: InternalPart):
        self.treeview.update_part(part_to_update)
