import sys
sys.path.append('../')
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
join = os.path.join

from segment_anything.utils.transforms import ResizeLongestSide
from .SemanticSegmentation import SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationCoarse


###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only body (no facial details)
class DrawingsDataset(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda'):
        self.sam_model = sam_model
        self.num_classes = 18
        self.semantics = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.num_test_samples = 100
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.num_test_samples]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.num_test_samples:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_image_tensor = self.cache[index][:3,:,:]
        if not self.cache_available: 
            image_name = f"{self.files[index].split('_')[0]}.png"
            # populate cache entry during first-time access
            image = cv2.imread(join(self.data_root, self.image_dir_name, image_name))
            image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
            input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            input_image_tensor = input_image_tensor.squeeze(0)
            self.cache[index][:3,:,:] = input_image_tensor

        gt2D_labels = self.cache[index][3:,:,:]
        if not self.cache_available: 
            # populate cache entry during first-time access
            gt2D = cv2.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
            gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
            gt2D_labels = self.semantics.colors_to_labels(gt2D)
            gt2D_labels = torch.tensor(gt2D_labels[None, :,:]).float()
            # vis_gt2D_labels = self.semantics.labels_to_colors(gt2D_labels)
            self.cache[index][3:,:,:] = gt2D_labels
        
        # binary mask   
        binmask = gt2D_labels.squeeze(0)>0
        binmask = np.uint8(binmask)

        bbox = np.array([0,0,1023,1023])
        if self.compute_bbox:
            Xs = np.where(binmask>0)[0]
            Ys = np.where(binmask>0)[1]
            bbox = np.array([min(Ys),min(Xs),max(Ys),max(Xs)])

        # convert image, gt, mask, bounding box to torch tensor
        return input_image_tensor, gt2D_labels.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()

    def init_cache(self):
        print(f"Initializing {self.mode} cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (4, 1024, 1024) # shape of data (not including batch)
        shared_array_base = mp.Array(ctypes.c_float, cache_length * 4 * 1024 * 1024)
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



###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only face
class DrawingsDatasetFace(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda'):
        self.sam_model = sam_model
        self.num_classes = 11
        self.semantics = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.num_test_samples = 100
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.num_test_samples]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.num_test_samples:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        face_present = True
        Xs = None
        Ys = None

        gt2D_labels = self.cache[index][3:,:,:]
        if not self.cache_available: 
            # populate cache entry during first-time access
            gt2D = cv2.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
            assert gt2D.shape[0]==1024 and gt2D.shape[1]==1024
            gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
            gt2D_labels = self.semantics.colors_to_labels(gt2D)
            face_binmask = gt2D_labels>0
            Xs = np.where(face_binmask>0)[0]
            Ys = np.where(face_binmask>0)[1]
            if len(Xs)==0 or len(Ys)==0:
                face_present = False
            if face_present:
                gt2D_labels_cropped = gt2D_labels[Xs.min():Xs.max(),Ys.min():Ys.max()] # crop to face (single channel)
                if gt2D_labels_cropped.shape[0]==0 or gt2D_labels_cropped.shape[1]==0 or face_binmask.sum()==0:
                    face_present = False
                    gt2D_labels = cv2.resize(gt2D_labels, (1024,1024), interpolation=cv2.INTER_NEAREST) # resize original image
                else:
                    gt2D_labels = cv2.resize(gt2D_labels_cropped, (1024,1024), interpolation=cv2.INTER_NEAREST) # resize cropped image
            gt2D_labels = torch.tensor(gt2D_labels[None, :,:]).float()
            self.cache[index][3:,:,:] = gt2D_labels

        input_image_tensor = self.cache[index][:3,:,:]
        if not self.cache_available: 
            image_name = f"{self.files[index].split('_')[0]}.png"
            image = cv2.imread(join(self.data_root, self.image_dir_name, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR) # face bbox is computed for 1024x1024
            if face_present:
                image_cropped = image[Xs.min():Xs.max(),Ys.min():Ys.max(),:] # crop to face (multi-channel)
                image = cv2.resize(image_cropped, (1024,1024), interpolation=cv2.INTER_LINEAR) # uncropped GT labels are already 1024x1024
            sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
            input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            input_image_tensor = input_image_tensor.squeeze(0)
            self.cache[index][:3,:,:] = input_image_tensor

        # binary mask   
        binmask = gt2D_labels.squeeze(0)>0
        binmask = np.uint8(binmask)

        bbox = np.array([0,0,1023,1023])
        if self.compute_bbox:
            Xs = np.where(binmask>0)[0]
            Ys = np.where(binmask>0)[1]
            bbox = np.array([min(Ys),min(Xs),max(Ys),max(Xs)])

        # convert image, gt, mask, bounding box to torch tensor
        return input_image_tensor, gt2D_labels.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()

    def init_cache(self):
        print(f"Initializing {self.mode} cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (4, 1024, 1024) # shape of data (not including batch)
        shared_array_base = mp.Array(ctypes.c_float, cache_length * 4 * 1024 * 1024)
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



###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only face
class DrawingsDatasetInference(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, device='cuda'):
        self.sam_model = sam_model
        self.num_classes = 11
        self.semantics = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.image_dir_name)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_image_tensor = self.cache[index]
        if not self.cache_available: 
            image_name = f"{self.files[index]}"
            image = cv2.imread(join(self.data_root, self.image_dir_name, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR) # face bbox is computed for 1024x1024
            sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
            input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            input_image_tensor = input_image_tensor.squeeze(0)
            self.cache[index] = input_image_tensor
        return input_image_tensor

    def init_cache(self):
        print(f"Initializing inference cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (3, 1024, 1024) # shape of data (not including batch)
        shared_array_base = mp.Array(ctypes.c_float, cache_length * data_dims[0] * data_dims[1] * data_dims[2])
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(cache_length, *data_dims)
        self.cache = torch.from_numpy(shared_array)
        self.cache *=0
        print(f"Inference dataset : {len(self.files)} --> {self.files[0]} -- {self.files[-1]}")
    
    def save_cache(self, path):
        torch.save(self.cache, path)

    def load_cache(self, path):
        self.cache = torch.load(path)

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################
