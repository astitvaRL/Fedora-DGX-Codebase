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
from skimage import io
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
join = os.path.join

from segment_anything.utils.transforms import ResizeLongestSide
from .SemanticSegmentation import SemanticSegmentation, SemanticSegmentationFace

# dataset definition for only body (no facial details) synthetic data
class Dataset_body(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda', return_embeddings=False):
        self.sam_model = sam_model
        self.num_classes = 18
        self.semantics = SemanticSegmentation(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.mode = mode
        self.device = device
        self.return_embeddings = return_embeddings
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[:-1000][:320]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[-1000:][:320]

     def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_image_tensor = self.cache[index][:3,:,:]
        if input_image_tensor.sum() == 0:
            image = io.imread(join(self.data_root, self.image_dir_name, self.files[index][:-4]+'.png'))
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
        if gt2D_labels.sum() == 0:
            gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index][:-6]+'_1024.png'))
            gt2D_labels = self.semantics.colors_to_labels(gt2D)
            gt2D_labels = torch.tensor(gt2D_labels[None, :,:]).float()
            # vis_gt2D_labels = self.semantics.labels_to_colors(gt2D_labels)
            self.cache[index][3:,:,:] = gt2D_labels
        
        # binary mask   
        binmask = gt2D_labels.squeeze(0)>0
        binmask = np.uint8(binmask)

        # for bounding box
        # Xs = np.where(binmask>0)[0]
        # Ys = np.where(binmask>0)[1]

        # convert img embedding, gt, mask, bounding box to torch tensor
        if self.return_embeddings:
            return input_image_tensor, torch.tensor(img_embed).float(), gt2D_labels.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([0,0,1023,1023])).float()
        else:
            return input_image_tensor, gt2D_labels.long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([0,0,1023,1023])).float()

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



# dataset definition for only body (no facial details) real data
class Dataset_real_body(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda', return_embeddings=False):
        self.sam_model = sam_model
        self.num_classes = 18
        self.semantics = SemanticSegmentation(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.mode = mode
        self.device = device
        self.return_embeddings = return_embeddings
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))
        self.files = [f for f in self.files if f.endswith('_1024.png')]
        if self.mode == 'train':
            self.files = self.files[:-100][:20]
        if self.mode == 'test':
            self.files = self.files[-100:][:4]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = f"{self.files[index].split('_')[0]}.png"
        image = io.imread(join(self.data_root, self.image_dir_name, image_name))
        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        input_image_tensor = input_image_tensor.squeeze(0)

        # load GT semantic segmentation map
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))

        # binary mask from GT, at inference can be replaced by an off-the-shelf model prediction (e.g. SAM)
        r,g,b = cv2.split(gt2D)
        binmask = ~((r==0) & (g==0) & (b==0))
        binmask = np.uint8(binmask)

        # colors to labels
        gt2D_labels = self.semantics.colors_to_labels(gt2D)
        # gt2D_labels_vis = self.semantics.labels_to_colors(gt2D_labels)

        # for bounding box
        Xs = np.where(binmask>0)[0]
        Ys = np.where(binmask>0)[1]
        
        # return image, gt, mask, bounding box as torch tensors
        return input_image_tensor, torch.tensor(gt2D_labels[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()







# dataset definition for only body (no facial details) synthetic data
class Dataset_precomputed_body(Dataset): 
    def __init__(self, data_root, labels_definition_file_path, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda'):
        self.num_classes = 18
        self.semantics = SemanticSegmentation(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.mode = mode
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[:-1000]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[-1000:]
        print(f"{mode} dataset : {len(self.files)} --> {self.files[0]} -- {self.files[-1]}") 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_embed = np.load(join(self.data_root, self.img_embed_dir_name, self.files[index][:-4]+'.npy')) #validate this logic in case filename convention changes
        
        # load GT semantic segmentation map
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index][:-6]+'_1024.png'))

        # binary mask from GT, at inference can be replaced by an off-the-shelf model prediction (e.g. SAM)
        r,g,b = cv2.split(gt2D)
        binmask = ~((r==0) & (g==0) & (b==0))
        binmask = np.uint8(binmask)

        # colors to labels
        gt2D_labels = self.semantics.colors_to_labels(gt2D)
        # gt2D_labels_vis = self.semantics.labels_to_colors(gt2D_labels)

        # for bounding box
        Xs = np.where(binmask>0)[0]
        Ys = np.where(binmask>0)[1]
        
        # convert img embedding, gt, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D_labels[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()



# dataset definition for only body (no facial details) real data
class Dataset_precomputed_body_real(Dataset): 
    def __init__(self, data_root, labels_definition_file_path, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda'):
        self.num_classes = 18
        self.semantics = SemanticSegmentation(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.mode = mode
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name)))
        self.files = [f for f in self.files if f.endswith('_1024.png')]
        if self.mode == 'train':
            self.files = self.files[:-100]
        if self.mode == 'test':
            self.files = self.files[-100:]
        print(f"{mode} dataset : {len(self.files)} --> {self.files[0]} -- {self.files[-1]}") 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        embedding_name = f"{self.files[index].split('_')[0]}.npy"
        img_embed = np.load(join(self.data_root, self.img_embed_dir_name, embedding_name)) #validate this logic in case filename convention changes
        
        # load GT semantic segmentation map
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))

        # binary mask from GT, at inference can be replaced by an off-the-shelf model prediction (e.g. SAM)
        r,g,b = cv2.split(gt2D)
        binmask = ~((r==0) & (g==0) & (b==0))
        binmask = np.uint8(binmask)

        # colors to labels
        gt2D_labels = self.semantics.colors_to_labels(gt2D)
        # gt2D_labels_vis = self.semantics.labels_to_colors(gt2D_labels)

        # for bounding box
        Xs = np.where(binmask>0)[0]
        Ys = np.where(binmask>0)[1]

        # convert img embedding, gt, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D_labels[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()



# dataset definition for only face, synthetic data
class Dataset_precomputed_face(Dataset): 
    def __init__(self, data_root, labels_definition_file_path, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda'):
        self.num_classes = 11
        self.semantics = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=self.num_classes)
        self.mode = mode
        self.device = device
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[:-1000]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[-1000:]
        print(f"{mode} dataset : {len(self.files)} --> {self.files[0]} -- {self.files[-1]}") 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_embed = np.load(join(self.data_root, self.img_embed_dir_name, self.files[index][:-4]+'.npy')) #validate this logic in case filename convention changes
        
        # load GT semantic segmentation map
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index][:-6]+'_1024.png'))

        # colors to labels
        gt2D_labels = self.semantics.colors_to_labels(gt2D)
        # gt2D_labels_vis = self.semantics.labels_to_colors(gt2D_labels)

        # binary mask from GT, at inference can be replaced by an off-the-shelf model prediction (e.g. SAM)
        binmask = gt2D_labels>0

        # for bbox face crop
        Xs = np.where(binmask>0)[0]
        Ys = np.where(binmask>0)[1]

        # crop GT to face
        gt2D_labels = gt2D_labels[Xs.min():Xs.max(),Ys.min():Ys.max()]
        gt2D_labels = cv2.resize(gt2D_labels, (1024,1024), interpolation=cv2.INTER_NEAREST)
        binmask_face = gt2D_labels>0

        # convert img embedding, gt, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D_labels[None, :,:]).long(), torch.tensor(binmask_face[None, :,:]).float(), torch.from_numpy(np.array([0,0,1023,1023])).float()
