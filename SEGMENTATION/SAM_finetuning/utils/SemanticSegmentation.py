import json
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

'''
{'Lower_leg': 1, 
'Eyebrows': 2, 
'Mouth': 3, 
'Pupils': 4, 
'Other_Facial_Accessory_': 5, 
'Head_Region': 6, 
'Hand': 7, 
'Lower_Torso': 8, 
'Lower_arm': 9, 
'Fingers': 10, 
'Skirt': 11, 
'Other_Head_accessory_': 12, 
'Nose': 13, 
'Upper_leg': 14, 
'Upper_Torso': 15, 
'Upper_arm': 16, 
'Teeth': 17, 
'Ears': 18, 
'Hand_accessory': 19, 
'Hair': 20, 
'Other_Body_accessory': 21, 
'Eyes': 22, 
'Tongue': 23, 
'Feet': 24, 
'Neck': 25, 
'BACKGROUND': 0, 
'CONFLICT': 255, 
'Unlabeled': 255, 
'Not_Annotated': 0, 
'Conflicted': 255}
'''

class SemanticSegmentation():
    def __init__(self, labels_definition_path, num_classes=18):
        self.num_classes = num_classes
        self.data = None
        with open(labels_definition_path) as json_file:
            self.data = json.load(json_file)
        self.data['label_name_to_id']['Unlabeled'] = 26 # handling id 255
        self.remap = {0:0, 1:1, 2:2, 3:2, 4:2, 5:2, 6:2, 13:2, 17:2, 18:2, 22:2, 23:2, 7:3, 8:4, 9:5, 10:6, 11:7, 12:8, 14:9, 15:10, 16:11, 19:12, 20:13, 21:14, 24:15, 25:16, 26:17}
        self.reverse_remap = {value: key for key, value in self.remap.items()}
        assert self.num_classes == 18 #highest id+1
        self.color_dict = {}
        for label_name in self.data['label_name_to_color']:
            self.color_dict[int(self.data['label_name_to_id'][label_name])] = self.data['label_name_to_color'][label_name]

    def remove_interpolation_artifacts(self, img):
        color_book = np.array([v for k,v in self.data['label_name_to_color'].items()])
        # Convert the image to HSV format
        # Reshape the image to a 2D array of pixels
        pixels = img.reshape((-1, 3))
        # Calculate the Euclidean distance between each pixel and each color in the color book
        distances = np.linalg.norm(pixels[:, None] - color_book, axis=2)
        # Get the index of the closest color for each pixel
        closest_colors = np.argmin(distances, axis=1)
        # Assign the closest color to each pixel
        result = color_book[closest_colors].reshape(img.shape)
        return result

    def labels_to_colors(self, img):
        w,h = img.shape[:2]
        img_rgb = np.zeros((w,h,3)).astype('uint8')
        for label_id in np.unique(img):
            img_rgb[img==label_id] = self.color_dict[self.reverse_remap[label_id]]
        return img_rgb
    
    def colors_to_labels(self, img):
        w,h = img.shape[:2]
        img = img.reshape(-1,3)
        r = img[:,0]
        g = img[:,1]
        b = img[:,2]
        labels = np.zeros_like(r)
        for classname in self.data['label_name_to_color']:
            ref_classname = classname + '' # copy precaution
            if classname in ['Eyebrows','Mouth','Pupils','Other_Facial_Accessory_','Nose','Teeth','Eyes', 'Ears','Tongue']:
                ref_classname = 'Head_Region'
            elif classname in ['Unlabeled', 'Conflicted', 'CONFLICT']: # handling id 255
                ref_classname = 'Unlabeled'
            elif classname in ['Not_Annotated', 'BACKGROUND']:
                ref_classname = 'BACKGROUND'
            color = self.data['label_name_to_color'][classname]
            mask = (r==color[0]) & (g==color[1]) & (b==color[2])
            remapped_id = self.remap[(int(self.data['label_name_to_id'][ref_classname]))]
            if remapped_id>=self.num_classes:
                print("Label ID overflows Number of Classes!")
                exit()
            labels[mask] = remapped_id
    
        labels = labels.reshape(w,h)
        return labels




class SemanticSegmentationFace():
    def __init__(self, labels_definition_path, num_classes=11):
        self.num_classes = num_classes
        self.data = None
        with open(labels_definition_path) as json_file:
            self.data = json.load(json_file)
        self.data['label_name_to_id']['Unlabeled'] = 26 # handling id 255
        self.remap = {0:0, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 13:6, 17:7, 18:8, 22:9, 23:10, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 14:0, 15:0, 16:0, 19:0, 20:0, 21:0, 24:0, 25:0, 26:0}
        self.reverse_remap = {value: key for key, value in self.remap.items()}
        assert self.num_classes == 11 #highest id+1
        self.color_dict = {}
        for label_name in self.data['label_name_to_color']:
            self.color_dict[int(self.data['label_name_to_id'][label_name])] = self.data['label_name_to_color'][label_name]

    def remove_interpolation_artifacts(self, img):
        color_book = np.array([v for k,v in self.data['label_name_to_color'].items()])
        # Convert the image to HSV format
        # Reshape the image to a 2D array of pixels
        pixels = img.reshape((-1, 3))
        # Calculate the Euclidean distance between each pixel and each color in the color book
        distances = np.linalg.norm(pixels[:, None] - color_book, axis=2)
        # Get the index of the closest color for each pixel
        closest_colors = np.argmin(distances, axis=1)
        # Assign the closest color to each pixel
        result = color_book[closest_colors].reshape(img.shape)
        return result

    def labels_to_colors(self, img):
        w,h = img.shape[:2]
        img_rgb = np.zeros((w,h,3)).astype('uint8')
        for label_id in np.unique(img):
            img_rgb[img==label_id] = self.color_dict[self.reverse_remap[label_id]]
        return img_rgb
    
    def colors_to_labels(self, img):
        w,h = img.shape[:2]
        img = img.reshape(-1,3)
        r = img[:,0]
        g = img[:,1]
        b = img[:,2]
        labels = np.zeros_like(r)
        for classname in self.data['label_name_to_color']:
            ref_classname = classname + '' # copy precaution
            if classname not in ['Head_Region','Eyebrows','Mouth','Pupils','Other_Facial_Accessory_','Nose','Teeth','Eyes', 'Ears','Tongue']:
                ref_classname = 'BACKGROUND'
            color = self.data['label_name_to_color'][classname]
            mask = (r==color[0]) & (g==color[1]) & (b==color[2])
            remapped_id = self.remap[(int(self.data['label_name_to_id'][ref_classname]))]
            if remapped_id>=self.num_classes:
                print("Label ID overflows Number of Classes!")
                exit()
            labels[mask] = remapped_id
    
        labels = labels.reshape(w,h)
        return labels
