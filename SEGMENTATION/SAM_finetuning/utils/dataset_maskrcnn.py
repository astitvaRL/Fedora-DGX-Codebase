import sys
sys.path.append('../')
sys.path.append('./')
import time
import os
import numpy as np
import ctypes
import multiprocessing as mp
import torch
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from collections.abc import Generator
from torchvision.ops import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from .strokes import create_stroke_prior


join = os.path.join

from segment_anything.utils.transforms import ResizeLongestSide
from .SemanticSegmentation import SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationCoarse, SemanticSegmentationAll


###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

class DrawingsDetectionDataset(Dataset): 
    def __init__(self, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', num_test_samples=100):
        self.num_classes = 27
        self.semantics = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.num_test_samples = num_test_samples
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.num_test_samples]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.num_test_samples:]
        # self.boxes = dict()
        # self.areas = dict()
        # self.names = dict()
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # input_image_tensor = self.cache[index][:3,:,:]
        # if not self.cache_available: 
        image_name = f"{self.files[index].split('_')[0]}.png"
        image = cv2.imread(join(self.data_root, self.image_dir_name, image_name))
        image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        # resize_img = sam_transform.apply_image(image)
        # resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        # input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        input_image_tensor = torch.tensor(image).float()/255
        input_image_tensor = torch.permute(input_image_tensor, (2,0,1))
        # populate cache entry during first-time access
        # self.cache[index][:3,:,:] = input_image_tensor

        # gt2D_labels = self.cache[index][3:,:,:]
        # if not self.cache_available: 
        # populate cache entry during first-time access
        gt2D = cv2.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
        gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
        gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
        gt2D_labels = self.semantics.colors_to_labels(gt2D)
        gt2D_labels = cv2.resize(gt2D_labels, (1024,1024), interpolation=cv2.INTER_NEAREST)
        gt2D_labels = torch.tensor(gt2D_labels[None, :,:]).float()
        # vis_gt2D_labels = self.semantics.labels_to_colors(gt2D_labels)
        # self.cache[index][3:,:,:] = gt2D_labels

        # estimate boxes, labels, areas
        obj_ids = torch.unique(gt2D_labels)
        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]
        # split the lebelled masks into a set of boolean masks.
        masks = gt2D_labels == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        labels = tv_tensors.Mask(masks)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(obj_ids),))
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(1024,1024))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels.to(torch.int64)
        target["image_id"] = index
        target["area"] = areas
        target["iscrowd"] = iscrowd
        # gt_mask = torch.nn.functional.one_hot(gt2D_labels.long(),self.num_classes)
        # gt_mask = gt_mask.squeeze(0)
        # gt_mask_tensor = torch.permute(gt_mask, (2,0,1))

        # # binary mask   
        # binmask = gt2D_labels.squeeze(0)>0
        # binmask = np.uint8(binmask)


        # convert image, gt, mask, bounding box to torch tensor
        # return input_image_tensor, gt2D_labels.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()

        return input_image_tensor, target

    def init_cache(self):
        print(f"Initializing {self.mode} cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (4, 1024, 1024) # shape of data (not including batch)
        shared_array_base = mp.Array(ctypes.c_float, cache_length * data_dims[0] * data_dims[1] * data_dims[2])
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(cache_length, *data_dims)
        self.cache = torch.from_numpy(shared_array)
        self.cache *=0
        print(f"{self.mode} dataset : {len(self.files)} --> {self.files[0]} -- {self.files[-1]}")
    
    def save_cache(self, path):
        torch.save(self.cache, path)

    def load_cache(self, path):
        self.cache = torch.load(path)

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################
