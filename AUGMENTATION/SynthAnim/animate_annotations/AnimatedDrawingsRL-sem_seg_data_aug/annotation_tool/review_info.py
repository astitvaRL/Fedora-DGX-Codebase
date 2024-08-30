from tkinter import ttk
from typing import Dict
from parts import InternalPart
from PIL import ImageTk


class ReviewInfo(ttk.Frame):

    def __init__(self, parent_widget: ttk.Frame, tool):
        super().__init__(parent_widget)
        self.tool = tool  # reference to parent AnnotationTool class

        label = ttk.Label(self, text='Review Stage')
        label.grid(column=0, row=0, columnspan=4)

        """ Treeview """
        self.treeview_style = ttk.Style()
        self.treeview_style.configure("Custom.Treeview", rowheight=32)

        self.treeview = ttk.Treeview(self, columns=("Name"), style="Custom.Treeview")
        self.treeview.heading("Name", text="Name")
        self.treeview.grid(column=0, row=1, columnspan=3)
        self.treeview_items: Dict[str, InternalPart] = {}  # dictionary mapping item_ids to the Part it represents
        self.treeview_thumbnails = {}  # dictionary mapping item_ids to the thumbnail tk so it isn't garbage collected
        self.treeview.grid()
        self.treeview.tag_configure("highlight", background="lightblue")

        self.treeview.bind("<ButtonRelease-1>", self.on_treeview_item_click)

        """ view buttons """
        button1 = ttk.Button(self, text='Left', command=lambda: self.switch_view('left'))
        button1.grid(column=0, row=2)

        button3 = ttk.Button(self, text='Center', command=lambda: self.switch_view('center'))
        button3.grid(column=1, row=2)

        button2 = ttk.Button(self, text='Right', command=lambda: self.switch_view('right'))
        button2.grid(column=2, row=2)

        button4 = ttk.Button(self, text='As Drawn', command=lambda: self.switch_view('as_drawn'))
        button4.grid(column=3, row=2)

        button4 = ttk.Button(self, text='Regenerate infill', command=self.regenerate_active_part_infill)
        button4.grid(column=0, row=3)

    def regenerate_active_part_infill(self):
        InternalPart.infill_parts(self.tool.active_part, recurse=False)
        self.tool.review_canvas.show_view(self.tool.active_view)

    def on_treeview_item_click(self, event):

        def _recurse_set_tag(item_id, tag: str):
            self.treeview.item(item_id, tags=(tag,))
            for c_item_id in self.treeview.get_children(item_id):
                _recurse_set_tag(c_item_id, tag)

        item_id = self.treeview.focus()
        self.tool.set_active_part(self.treeview_items[item_id])
        if item_id:
            _recurse_set_tag("", "")  # remove all tags
            _recurse_set_tag(item_id, "highlight")  # add highlight tag to selection

    def populate_treeview(self):

        self.treeview.delete(*self.treeview.get_children())  # clear existing items in treeview

        part_name_to_treeview_id: Dict[str, int] = {}
        for part in self.tool.parts:

            # insert into treeview as tree
            if part.parent_name:
                assert part.parent_name in part_name_to_treeview_id.keys(), 'parent_name part not in treeview yet'
                parent_item = part_name_to_treeview_id[part.parent_name]
            else:
                parent_item = ""

            image_tk = ImageTk.PhotoImage(image=part.mask_thumbnail)
            item_id = self.treeview.insert(parent_item, "end", values=(part.name,), image=image_tk)
            part_name_to_treeview_id[part.name] = item_id
            self.treeview_items[item_id] = part
            self.treeview_thumbnails[item_id] = image_tk

        # open them all
        def _recurse_open_items(item):
            self.treeview.item(item, open=True)
            for c_item in self.treeview.get_children(item):
                _recurse_open_items(c_item)
        _recurse_open_items("")

    def clear_treeview(self):
        for item_id in self.treeview_items.keys():
            self.treeview.delete(item_id)
        self.treeview_items = {}

    def switch_view(self, view: str):
        self.tool.active_view = view
        self.tool.review_canvas.show_view(self.tool.active_view)
