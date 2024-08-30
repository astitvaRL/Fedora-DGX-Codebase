from tkinter import ttk
from typing import Dict
from parts import InternalPart
from PIL import ImageTk


class Treeview(ttk.Treeview):

    def __init__(self, parent_widget, tool):

        self.tool = tool

        self.treeview_style = ttk.Style()
        self.treeview_style.configure("Custom.Treeview", rowheight=32)

        super().__init__(parent_widget, columns=("Name"), style="Custom.Treeview")

        self.heading("Name", text="Name")
        self.grid(column=0, row=1, columnspan=3)
        self.bind("<<TreeviewSelect>>", self.treeview_item_selected)
        self.items: Dict[str, InternalPart] = {}  # dictionary mapping item_ids to the Part it represents
        self.thumbnails = {}  # dictionary mapping item_ids to the thumbnail tk so it isn't garbage collected

        self.treeview_thumbnails = {}
        self.treeview_items = {}
        self.treeview_name_to_id = {}

    def display_current_state_of_tool_parts(self):

        # clear what is there now
        self.clear_treeview()

        # add all the parts to the treeview
        for part in self.tool.parts:
            self.add_part_to_treeview(part)

        for part in self.tool.parts:

            if not part.parent_name:  # root part
                continue

            parent_id = self.treeview_name_to_id[part.parent_name]
            child_id = self.treeview_name_to_id[part.name]
            self.move(child_id, parent_id, 'end')

    def add_part_to_treeview(self, part: InternalPart):
        image_tk = ImageTk.PhotoImage(image=part.mask_thumbnail)
        item_id = self.insert("", "end", values=(part.name,), image=image_tk)

        if part.name in self.treeview_name_to_id.keys():
            assert False, f'duplicate part name: {part.name}'

        self.treeview_name_to_id[part.name] = item_id

        self.treeview_thumbnails[item_id] = image_tk
        self.treeview_items[item_id] = part

        self.selection_set(item_id)

    def update_part(self, part_to_update: InternalPart):

        for item_id, part in self.treeview_items.items():
            if part is part_to_update:
                image_tk = ImageTk.PhotoImage(image=part.mask_thumbnail)
                self.treeview_thumbnails[item_id] = image_tk
                self.item(item_id, values=(part.name,), image=image_tk)
                return

    def treeview_item_selected(self, event):

        treeview_selection = self.selection()
        if len(treeview_selection) == 0:
            selected_part = None
        else:
            selected_part = self.treeview_items[treeview_selection[0]]

        self.tool.set_active_part(selected_part)

    def clear_treeview(self):

        for item_id in self.get_children():
            self.delete(item_id)
        self.treeview_items = {}

        self.treeview_thumbnails = {}

        self.treeview_name_to_id = {}
