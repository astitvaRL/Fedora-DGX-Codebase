
import tkinter as tk
from tkinter import ttk
from typing import List
from keypoint import Keypoint


class PoseInfo(ttk.Frame):
    def __init__(self, parent_widget: ttk.Frame, active_oval_id_var: tk.IntVar):
        # get our geometry set up and ready to go
        super().__init__(parent_widget)
        self.listbox = tk.Listbox(self, height=16)
        self.listbox.grid(column=0, row=0)

        self.listbox.bind("<<ListboxSelect>>", lambda e: self.update_active_oval_id_var())

        self.active_oval_id_var: tk.IntVar = active_oval_id_var
        self.active_oval_id_var.trace('w', self.update_selected_item_in_list)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def set_listbox_items(self, items: List[Keypoint]):
        choices_var = tk.StringVar(value=items)
        self.listbox['listvariable'] = choices_var

    def update_active_oval_id_var(self):
        print('here')
        selection = self.listbox.curselection()[0]
        self.active_oval_id_var.set(selection)

    def update_selected_item_in_list(self, *args):
        self.listbox.selection_clear(0, self.listbox.size())

        if self.active_oval_id_var.get() == -1:
            return

        self.listbox.selection_set(self.active_oval_id_var.get())
        self.listbox.activate(self.active_oval_id_var.get())
