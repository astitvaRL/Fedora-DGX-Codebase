from parts import ExternalPart
from tkinter import ttk
import tkinter as tk


class ExternalPartFrame(ttk.Frame):

    def __init__(self, parent_widget):
        super().__init__(parent_widget, style="PartFrame.TFrame")
        self['borderwidth'] = 2
        self['relief'] = 'sunken'

        self.parent_frame = parent_widget

        self.name_label = ttk.Label(self, text='Name')
        self.name_label.grid(column=0, row=0)

        self.name_text = tk.Text(self, width=20, height=1)
        self.name_text.grid(column=1, row=0)

        self.parent_label = ttk.Label(self, text='Parent')
        self.parent_label.grid(column=0, row=2)

        self.parent_value = ttk.Combobox(self, state="readonly")
        self.parent_value.grid(column=1, row=2)

        self.forward_orientation_label = ttk.Label(self, text='Part Forward Orientation')
        self.forward_orientation_label.grid(column=0, row=3)

        self.forward_orientation_value = ttk.Combobox(self, state="readonly")
        self.forward_orientation_value.grid(column=1, row=3)

        save_button = ttk.Button(self, text='save', command=self.save)
        save_button.grid(column=0, row=4)

    def display_part_info(self, part: ExternalPart):
        # clear existing info
        self.name_text.delete("1.0", "end")

        self.parent_value['values'] = []
        self.parent_value.set("")

        self.forward_orientation_value['values'] = ['None', 'DLeft', 'DRight']

        if part is None:
            return

        # add part's current info
        self.name_text.insert("1.0", part.name)

        self.parent_value['values'] = ["None (Root)"] + [p.name for p in self.parent_frame.tool.parts if p is not part]
        if part.parent_name is None:
            self.parent_value.set("None (Root)")
        else:
            self.parent_value.set(part.parent_name)

        self.forward_orientation_value.set(part.forward_orientation)

    def save(self):
        if self.parent_frame.tool.active_part is None:
            return

        self.parent_frame.tool.active_part.set_name(self.name_text.get("1.0", "end").strip())
        self.parent_frame.tool.update_visible_treeview()

        parent_name = None if self.parent_value.get() == "None (Root)" else self.parent_value.get()
        self.parent_frame.tool.active_part.set_parent_name(parent_name)

        self.parent_frame.tool.active_part.set_forward_orientation(self.forward_orientation_value.get())
        self.name_text.get("1.0", "end").strip()
