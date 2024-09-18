import sys
sys.path.append('../')
import time
import os
import numpy as np
import imageio as io # don't use, messes with orientation of the image
import ctypes
import multiprocessing as mp
import torch
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
join = os.path.join

from segment_anything.utils.transforms import ResizeLongestSide
from .SemanticSegmentation import SemanticSegmentation, SemanticSegmentationFace, SemanticSegmentationTernary


# dataset definition for only body (no facial details)
class Dataset_NoFace(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda'):
        self.sam_model = sam_model
        self.num_classes = 18
        self.semantics = SemanticSegmentation(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
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

################## --------------------------------------------------------------------------------------------------------------------------------------------




# dataset definition for only body (no facial details), three classes
class Dataset_body_real_ternary(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda', return_embeddings=False):
        self.sam_model = sam_model
        self.num_classes = 3
        self.semantics = SemanticSegmentationTernary(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.return_embeddings = return_embeddings
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
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


        img_embed = None
        if self.return_embeddings:
            img_embed = np.load(join(self.data_root, self.img_embed_dir_name, self.files[index][:-4]+'.npy')) #validate this logic in case filename convention changes
        
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

        # convert img embedding, gt, mask, bounding box to torch tensor
        if self.return_embeddings:
            return input_image_tensor, torch.tensor(img_embed).float(), gt2D_labels.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()
        else:
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

################## --------------------------------------------------------------------------------------------------------------------------------------------
