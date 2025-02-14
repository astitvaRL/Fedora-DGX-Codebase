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

from .strokes import create_stroke_prior


join = os.path.join

from segment_anything.utils.transforms import ResizeLongestSide
from .SemanticSegmentation import SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationCoarse, SemanticSegmentationAll


###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only body (no facial details)
class DrawingsDataset(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', sample_size=100):
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
        self.sample_size = sample_size
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.sample_size]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.sample_size:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = self.files[index]
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

        if self.mode=='eval': 
            return input_image_tensor, gt2D_labels.long(), image_name

        return input_image_tensor, gt2D_labels.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()
    

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

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only body (no facial details)
class DrawingsDatasetCoarsePoints(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', sample_size=100):
        self.sam_model = sam_model
        self.num_classes = 6
        self.num_random_points = 100
        self.semantics = SemanticSegmentationCoarse(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.sample_size]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.sample_size:]

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



###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only face
class DrawingsDatasetFace(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', sample_size=100):
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
        self.sample_size = sample_size
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.sample_size]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.sample_size:]

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
            gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
            # assert gt2D.shape[0]==1024 and gt2D.shape[1]==1024
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



###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only face
class DrawingsDatasetInference(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, max_num_samples, device='cuda'):
        self.sam_model = sam_model
        self.num_classes = 11
        self.semantics = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.max_num_samples = max_num_samples
        self.files = sorted(os.listdir(join(self.data_root, self.image_dir_name)))[:self.max_num_samples]

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
###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only body (no facial details) coarse-to-fine
class DrawingsDatasetStrokes(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', sample_size=100):
        self.sam_model = sam_model
        self.num_classes_coarse = 6
        self.num_classes = 18
        self.semantics_coarse = SemanticSegmentationCoarse(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes_coarse)
        self.semantics_fine = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.sample_size]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.sample_size:]

    def __len__(self):
        return len(self.files)
    
    def bezier(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Generator[np.ndarray, None, None]:
        def calc(t):
            return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3
        # get the approximate pixel count of the curve
        approx = cv2.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
        for t in np.linspace(0, 1, round(approx * 1.2)):
            yield np.round(calc(t)).astype(np.int32)
    def generate_scratch(self, img: np.ndarray, max_length: float, end_brush_range: tuple[float, float], mid_brush_range: tuple[float, float]) -> np.ndarray:
        H, W = img.shape
        # generate the 2 end points of the bezier curve
        x, y, rho1, theta1 = np.random.uniform([0] * 4, [W, H, max_length, np.pi * 2])
        p1 = np.array([x, y, 0])
        p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]
        # generate the second point, make sure that it cannot be too far away from the middle point of the 2 end points
        rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
        p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]
        # generate the brush sizes of the 3 points
        p1[2], p2[2], p3[2] = np.random.uniform(*np.transpose([end_brush_range, mid_brush_range, end_brush_range]))
        for x, y, brush in self.bezier(p1, p2, p3):
            cv2.circle(img, (x, y), brush, 255, -1)
        return img
    def create_stroke_prior(self, mask):
        MAX_LENGTH = 512  # maximum distance between two end points
        END_BRUSH_RANGE = (np.random.randint(2,30), np.random.randint(2,30))  # brush size range of the two end points
        MID_BRUSH_RANGE = (np.random.randint(2,30), np.random.randint(2,30))  # brush size range of the mid point
        SCRATCH_CNT = np.random.randint(20,60)
        strokes = np.zeros_like(mask)
        for _ in range(SCRATCH_CNT):
            self.generate_scratch(strokes, MAX_LENGTH, END_BRUSH_RANGE, MID_BRUSH_RANGE)
        exclude_class = 0
        # if mask.max()>1:
        #     exclude_class = np.random.randint(0,mask.max())
        strokes[strokes>0] = mask[strokes>0]
        # strokes[strokes==exclude_class] = 0
        return strokes


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

        gt2D_labels_coarse = self.cache[index][3:4,:,:]
        gt2D_labels_fine = self.cache[index][4:,:,:]
        if not self.cache_available: 
            # populate cache entry during first-time access
            gt2D = cv2.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
            gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
            gt2D_labels_coarse = self.semantics_coarse.colors_to_labels(gt2D)
            gt2D_labels_fine = self.semantics_fine.colors_to_labels(gt2D)
            gt2D_labels_coarse = torch.tensor(gt2D_labels_coarse[None, :,:]).float()
            gt2D_labels_fine = torch.tensor(gt2D_labels_fine[None, :,:]).float()
            # vis_gt2D_labels = self.semantics.labels_to_colors(gt2D_labels)
            self.cache[index][3:4,:,:] = gt2D_labels_coarse
            self.cache[index][4:5,:,:] = gt2D_labels_fine
        
        # binary mask   
        binmask = gt2D_labels_fine.squeeze(0)>0
        binmask = np.uint8(binmask)

        bbox = np.array([0,0,1023,1023])
        if self.compute_bbox:
            Xs = np.where(binmask>0)[0]
            Ys = np.where(binmask>0)[1]
            bbox = np.array([min(Ys),min(Xs),max(Ys),max(Xs)])
        
        # mask to strokes
        gt2D_labels_coarse = self.create_stroke_prior(gt2D_labels_coarse.squeeze(0).numpy().astype('uint8'))
        gt2D_labels_coarse = torch.tensor(gt2D_labels_coarse[None, :,:]).float()


        if self.semantics_coarse.exclude_neck:
            neck_mask = gt2D_labels_coarse==5
            gt2D_labels_coarse[neck_mask] = 4 #merge with torso
        
        # convert image, gt, mask, bounding box to torch tensor
        return input_image_tensor, gt2D_labels_coarse.long(), gt2D_labels_fine.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()

    def init_cache(self):
        print(f"Initializing {self.mode} cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (5, 1024, 1024) # shape of data (not including batch)
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
###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for only body (no facial details) coarse-to-fine
class DrawingsDatasetC2F(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', sample_size=100):
        self.sam_model = sam_model
        self.num_classes_coarse = 6
        self.num_classes = 18
        self.semantics_coarse = SemanticSegmentationCoarse(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes_coarse)
        self.semantics_fine = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.sample_size]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.sample_size:]

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

        gt2D_labels_coarse = self.cache[index][3:4,:,:]
        gt2D_labels_fine = self.cache[index][4:,:,:]
        if not self.cache_available: 
            # populate cache entry during first-time access
            gt2D = cv2.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
            gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
            gt2D_labels_coarse = self.semantics_coarse.colors_to_labels(gt2D)
            gt2D_labels_fine = self.semantics_fine.colors_to_labels(gt2D)
            gt2D_labels_coarse = torch.tensor(gt2D_labels_coarse[None, :,:]).float()
            gt2D_labels_fine = torch.tensor(gt2D_labels_fine[None, :,:]).float()
            # vis_gt2D_labels = self.semantics.labels_to_colors(gt2D_labels)
            self.cache[index][3:4,:,:] = gt2D_labels_coarse
            self.cache[index][4:5,:,:] = gt2D_labels_fine
        
        # binary mask   
        binmask = gt2D_labels_coarse.squeeze(0)>0
        binmask = np.uint8(binmask)

        bbox = np.array([0,0,1023,1023])
        if self.compute_bbox:
            Xs = np.where(binmask>0)[0]
            Ys = np.where(binmask>0)[1]
            bbox = np.array([min(Ys),min(Xs),max(Ys),max(Xs)])

        if self.semantics_coarse.exclude_neck:
            neck_mask = gt2D_labels_coarse==5
            gt2D_labels_coarse[neck_mask] = 4 #merge with torso

        # convert image, gt, mask, bounding box to torch tensor
        return input_image_tensor, gt2D_labels_coarse.long(), gt2D_labels_fine.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()

    def init_cache(self):
        print(f"Initializing {self.mode} cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (5, 1024, 1024) # shape of data (not including batch)
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

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition HD coarse-to-fine
class DrawingsDatasetC2FHD(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', sample_size=100):
        self.sam_model = sam_model
        self.num_classes_coarse = 6
        self.num_classes = 27
        self.semantics_coarse = SemanticSegmentationCoarse(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes_coarse)
        self.semantics_fine = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.sample_size]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.sample_size:]

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

        gt2D_labels_coarse = self.cache[index][3:4,:,:]
        gt2D_labels_fine = self.cache[index][4:,:,:]
        if not self.cache_available: 
            # populate cache entry during first-time access
            gt2D = cv2.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
            gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
            gt2D_labels_coarse = self.semantics_coarse.colors_to_labels(gt2D)
            gt2D_labels_fine = self.semantics_fine.colors_to_labels(gt2D)
            gt2D_labels_coarse = torch.tensor(gt2D_labels_coarse[None, :,:]).float()
            gt2D_labels_fine = torch.tensor(gt2D_labels_fine[None, :,:]).float()
            # vis_gt2D_labels = self.semantics.labels_to_colors(gt2D_labels)
            self.cache[index][3:4,:,:] = gt2D_labels_coarse
            self.cache[index][4:5,:,:] = gt2D_labels_fine
        
        # binary mask   
        binmask = gt2D_labels_coarse.squeeze(0)>0
        binmask = np.uint8(binmask)

        bbox = np.array([0,0,1023,1023])
        if self.compute_bbox:
            Xs = np.where(binmask>0)[0]
            Ys = np.where(binmask>0)[1]
            bbox = np.array([min(Ys),min(Xs),max(Ys),max(Xs)])

        if self.semantics_coarse.exclude_neck:
            neck_mask = gt2D_labels_coarse==5
            gt2D_labels_coarse[neck_mask] = 4 #merge with torso

        # convert image, gt, mask, bounding box to torch tensor
        return input_image_tensor, gt2D_labels_coarse.long(), gt2D_labels_fine.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(bbox).float()

    def init_cache(self):
        print(f"Initializing {self.mode} cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (5, 1024, 1024) # shape of data (not including batch)
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
###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

# dataset definition for all classes, coarse-to-fine
class DrawingsDatasetC2FAll(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, label_id_dir_name, mode='train', device='cuda', sample_size=100):
        self.sam_model = sam_model
        self.num_classes_coarse = 5
        self.num_classes_fine = 18
        self.num_classes_face = 11
        self.num_classes_all = 27
        self.semantics_coarse = SemanticSegmentationCoarse(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes_coarse, exclude_neck=True)
        self.semantics_fine = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes_fine)
        self.semantics_face = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes_face)
        self.semantics_all = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes_all)
        self.compute_bbox=False # slower data loading if True
        self.mode = mode
        self.cache_available=False
        self.cache = None
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[:-self.sample_size]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))[-self.sample_size:]

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

        gt2D_labels_coarse = self.cache[index][3:4,:,:]
        gt2D_labels_fine = self.cache[index][4:5,:,:]
        gt2D_labels_face = self.cache[index][5:6,:,:]
        gt2D_labels_all = self.cache[index][6:,:,:]
        if not self.cache_available: 
            # populate cache entry during first-time access
            gt2D = cv2.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
            gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
            gt2D_labels_coarse = self.semantics_coarse.colors_to_labels(gt2D)
            gt2D_labels_fine = self.semantics_fine.colors_to_labels(gt2D)
            gt2D_labels_face = self.semantics_face.colors_to_labels(gt2D)
            gt2D_labels_all = self.semantics_all.colors_to_labels(gt2D)
            gt2D_labels_coarse = torch.tensor(gt2D_labels_coarse[None, :,:]).float()
            gt2D_labels_fine = torch.tensor(gt2D_labels_fine[None, :,:]).float()
            gt2D_labels_face = torch.tensor(gt2D_labels_face[None, :,:]).float()
            gt2D_labels_all = torch.tensor(gt2D_labels_all[None, :,:]).float()
            # vis_gt2D_labels = self.semantics.labels_to_colors(gt2D_labels)
            self.cache[index][3:4,:,:] = gt2D_labels_coarse
            self.cache[index][4:5,:,:] = gt2D_labels_fine
            self.cache[index][5:6,:,:] = gt2D_labels_face
            self.cache[index][6:,:,:] = gt2D_labels_all

        
        if self.semantics_coarse.exclude_neck:
            neck_mask = gt2D_labels_coarse==5
            gt2D_labels_coarse[neck_mask] = 4 #merge with torso

        # convert image, gt, mask, bounding box to torch tensor
        return input_image_tensor, gt2D_labels_coarse.long(), gt2D_labels_fine.long(),  gt2D_labels_face.long(),  gt2D_labels_all.long(), image_name

    def init_cache(self):
        print(f"Initializing {self.mode} cache...")
        cache_length = len(self.files) # number of samples you want to cache
        data_dims = (7, 1024, 1024) # shape of data (not including batch)
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


###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################


# inference with coarse prior dataset definition for all classes, coarse-to-fine + face
class DrawingsDatasetInferFullWithCoarsePrior(Dataset): 
    def __init__(self, sam_model, img_dir_name, label_id_dir_name, semantics, coarse_mask_format='CartoonDogs', bg_color=[255,255,255], mode='test', device='cuda', sample_size=-1):
        self.sam_model = sam_model
        self.num_classes_coarse = 5
        self.coarse_mask_format = coarse_mask_format
        self.num_classes_fine = 18
        self.num_classes_face = 11
        self.num_classes_all = 27
        self.bg_color = bg_color
        self.mode = mode
        self.device = device
        self.semantics = semantics
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(self.image_dir_name))
        if self.sample_size>-1:
            self.files = files[:self.sample_size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = self.files[index]

        image = cv2.imread(join(self.image_dir_name, image_name),-1)
        if image.shape[-1]==4:
            alpha = image[:,:,3]
            image = image[:,:,:3]
            image[alpha==0] = self.bg_color
        image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        input_image_tensor = input_image_tensor.squeeze(0)

        prior_name = image_name.split('.')[0] + '.png'
        if self.coarse_mask_format == "Manual":
            prior_name = image_name.split('.')[0] + '_coarse.png'


        gt2D = cv2.imread(join(self.label_id_dir_name, prior_name))
        gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)

        if self.coarse_mask_format == 'CartoonDogs': 
            #remapping coarse prior to DrawingsDataset annotation format (only valid for Dog dataset)
            gt2D = gt2D[:,:,0]
            bg = gt2D==0
            head = gt2D==1
            arms = (gt2D==3) | (gt2D==4)
            legs = (gt2D==5) | (gt2D==6)
            torso = (gt2D==2) | (gt2D==7)
            gt2D[bg]=0
            gt2D[head] = 2
            gt2D[arms] = 3
            gt2D[legs] = 1
            gt2D[torso] = 4
            gt2D[gt2D>4] = 0
        elif self.coarse_mask_format=="Manual":
            assert gt2D.shape[-1]==3
            b,g,r = cv2.split(gt2D)
            bg = (r==0) & (g==0) & (b==0)
            head = (r>200) & (g>200) & (b<50)
            torso = (r<50) & (g>200) & (b<50)
            arms = (r>200) & (g<50) & (b<50)
            legs = (r<50) & (g<50) & (b>200)
            gt2D = np.zeros((1024,1024))
            gt2D[bg]=0
            gt2D[legs] = 1
            gt2D[head] = 2
            gt2D[arms] = 3
            gt2D[torso] = 4

        gt2D_label_id_coarse = torch.tensor(gt2D[None, :,:]).long()

        return image_name,input_image_tensor, gt2D_label_id_coarse

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################


# inference with coarse prior dataset definition for all classes, coarse-to-fine + face
class DrawingsDatasetInferFullWithStrokes(Dataset): 
    def __init__(self, sam_model, img_dir_name, label_id_dir_name, semantics, bg_color=[255,255,255], mode='test', device='cuda', sample_size=-1):
        self.sam_model = sam_model
        self.num_classes_coarse = 5
        self.num_classes_fine = 18
        self.num_classes_face = 11
        self.num_classes_all = 27
        self.bg_color = bg_color
        self.mode = mode
        self.device = device
        self.semantics = semantics
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(self.image_dir_name))[:self.sample_size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = self.files[index]

        # image = cv2.imread(join(self.image_dir_name, image_name),-1)
        image = cv2.imread('dragy.png')
        if image.shape[-1]==4:
            alpha = image[:,:,3]
            image = image[:,:,:3]
            image[alpha==0] = self.bg_color
        image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        input_image_tensor = input_image_tensor.squeeze(0)

        prior_name = image_name.split('.')[0] + '.png'
        # gt2D = cv2.imread(join(self.label_id_dir_name, prior_name))
        gt2D = cv2.imread('dragy_prior.png')
        gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)

        # #remapping coarse prior to DrawingsDataset annotation format (only valid for Dog dataset)
        # gt2D = gt2D[:,:,0]
        # bg = gt2D==0
        # head = gt2D==1
        # arms = (gt2D==3) | (gt2D==4)
        # legs = (gt2D==5) | (gt2D==6)
        # torso = (gt2D==2) | (gt2D==7)


        # remapping for manual strokes
        b,g,r = cv2.split(gt2D)
        gt2D = np.zeros_like(gt2D[:,:,0])
        bg = (r==0) & (g==0) & (b==0)
        head = (r==255) & (g==255) & (b==0)
        arms = (r==255) & (g==0) & (b==0)
        legs = (r==0) & (g==0) & (b==255)
        torso = (r==0) & (g==255) & (b==0)
        
        # convert to DrawingsDataset format
        gt2D[bg]=0
        gt2D[head] = 2
        gt2D[arms] = 3
        gt2D[legs] = 1
        gt2D[torso] = 4
        gt2D[gt2D>4] = 0

        # #mask to strokes
        # gt2D = create_stroke_prior(gt2D, exclude_random_class=False)

        gt2D_label_id_coarse = torch.tensor(gt2D[None, :,:]).long()

        return image_name,input_image_tensor, gt2D_label_id_coarse

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################


# inference dataset definition for all classes, coarse-to-fine + face
class DrawingsDatasetInferFull(Dataset): 
    def __init__(self, sam_model, img_dir_name, label_id_dir_name=None, bg_color=[255,255,255], mode='test', device='cuda', sample_size=-1):
        self.sam_model = sam_model
        self.num_classes_coarse = 5
        self.num_classes_fine = 18
        self.num_classes_face = 11
        self.num_classes_all = 27
        self.bg_color = bg_color
        self.mode = mode
        self.device = device
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.sample_size = sample_size
        self.files = sorted(os.listdir(self.image_dir_name))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image_name = self.files[index]

        image = cv2.imread(join(self.image_dir_name, image_name),-1)
        if image.shape[-1]==4:
            alpha = image[:,:,3]
            image = image[:,:,:3]
            image[alpha==0] = self.bg_color
        dims = image.shape
        image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        input_image_tensor = input_image_tensor.squeeze(0)

        return image_name, input_image_tensor, dims

###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################
