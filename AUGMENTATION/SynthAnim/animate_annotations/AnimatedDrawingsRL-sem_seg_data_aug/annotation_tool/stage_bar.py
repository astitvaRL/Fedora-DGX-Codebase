
import tkinter as tk
from tkinter import ttk


class StageBar(ttk.Frame):

    def __init__(self, tool, parent_widget: ttk.Frame, current_stage_idx_var: tk.IntVar):
        # get our geometry set up and ready to go
        super().__init__(parent_widget)

        self.tool = tool

        self.rowconfigure(0, weight=1)

        # keep track of the variables we need to interface with rest of tool
        self.current_stage_idx_var: tk.IntVar = current_stage_idx_var

        # far left label
        stage_label = ttk.Label(self, text='Annotation Stage:')
        stage_label.grid(column=0, row=0)
        self.columnconfigure(0, weight=1)

        """*****************"""

        # from stage text create labels
        self._stages_text = [
            'Pose',
            'Foreground/Background Segmentation',
            'Mask Splitting', 
            'Part Segmentation'
            # 'Review: Front',
            # 'Review: Left',
            # 'Review: Right',
            # 'Review: Back',
        ]
        self._stages_labels = []
        for idx, stage_text in enumerate(self._stages_text):
            stage_label = ttk.Label(self, text=stage_text, padding=(10, 10, 10, 10))
            stage_label.grid(column=idx+1, row=0)
            self._stages_labels.append(stage_label)
            self.columnconfigure(idx+1, weight=1)

        # styles for our labels
        self.style_active = ttk.Style()
        self.style_active.configure('Active.TLabel', background='Green', borderwidth=5, relief='raised')
        self.style_not_active = ttk.Style()
        self.style_not_active.configure('Non-Active.TLabel')

        """*****************"""
        # buttons for going forwards and backwards between poses
        prev_stage_button = ttk.Button(self, text='Prev Stage', command=self.decrement_stage)
        prev_stage_button.grid(column=len(self._stages_text)+2, row=0)
        next_stage_button = ttk.Button(self, text='Next Stage', command=self.increment_stage)
        next_stage_button.grid(column=len(self._stages_text)+3, row=0)

        # self.set_stage(self.current_stage_idx_var.get())

    def decrement_stage(self):
        if self.current_stage_idx_var.get() == 0:
            return
        self.set_stage(self.current_stage_idx_var.get()-1)

    def increment_stage(self):
        # self.tool.conclude_stage()

        if self.current_stage_idx_var.get() >= 3:
            return
        self.set_stage(self.current_stage_idx_var.get()+1)

    def set_stage(self, new_stage_id: int):
        self.current_stage_idx_var.set(new_stage_id)
        self.set_current_stage_label_style()

    def set_current_stage_label_style(self):

        for idx, stage_label in enumerate(self._stages_labels):
            if idx == self.current_stage_idx_var.get():
                stage_label['style'] = 'Active.TLabel'
            else:
                stage_label['style'] = 'Non-Active.TLabel'
