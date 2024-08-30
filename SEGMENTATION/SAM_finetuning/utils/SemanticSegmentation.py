import json
import os
import numpy as np



class SemanticSegmentation():
    def __init__(self, labels_definition_path):
        self.data = None
        with open(labels_definition_path) as json_file:
            self.data = json.load(json_file)
        self.data['label_name_to_id']['CONFLICT']=26
        self.data['label_name_to_id']['Conflicted']=26
        self.data['label_name_to_id']['Unlabeled']=26
        self.color_dict = {}
        for label_name in self.data['label_name_to_color']:
            self.color_dict[int(self.data['label_name_to_id'][label_name])] = self.data['label_name_to_color'][label_name]

    def labels_to_colors(self, img):
        w,h = img.shape[:2]
        img_rgb = np.zeros((w,h,3)).astype('uint8')
        for label_id in self.color_dict:
            img_rgb[img==label_id] = self.color_dict[label_id]
        return img_rgb
