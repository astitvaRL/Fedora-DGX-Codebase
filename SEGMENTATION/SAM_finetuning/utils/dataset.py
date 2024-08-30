import sys
sys.path.append('../')

import os
import numpy as np
import torch
import cv2
from skimage import io
from torch.utils.data import Dataset
join = os.path.join

from segment_anything.utils.transforms import ResizeLongestSide

# dataset definition for full character segmentation
class Dataset_multiclass(Dataset): 
    def __init__(self, data_root, img_embed_dir_name, label_id_dir_name, mode='train'):
        self.mode = mode
        self.data_root = data_root
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[:-100]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[-100:]
        print(f"{mode} dataset : {len(self.files)} --> {self.files[0]} -- {self.files[-1]}") 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_embed = np.load(join(self.data_root, self.img_embed_dir_name, self.files[index][:-4]+'.npy')) #validate this logic in case filename convention changes
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index][:-6]+'.png'))
        # binary mask from GT, at inference can be replaced by an off
        img_embed = np.load(join(self.data_root, self.img_embed_dir_name, self.files[index][:-4]+'.npy')) #validate this logic in case filename convention changes
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index][:-6]+'.png'))

        # binary mask from GT, at inference can be replaced by an off-the-shelf model prediction (e.g. SAM)
        binmask = gt2D>0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        binmask = cv2.morphologyEx(binmask.astype('uint8'),cv2.MORPH_CLOSE,kernel)

        # for bounding box
        Xs = np.where(binmask>0)[0]
        Ys = np.where(binmask>0)[1]
        
        # convert img embedding, gt, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()


# dataset definition for face segmentation
class Dataset_multiclass_face(Dataset): 
    def __init__(self, data_root, img_dir_name, img_embed_dir_name, label_id_dir_name):
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.label_id_dir_name))) 
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_embed = np.load(join(self.data_root, self.img_embed_dir_name, self.files[index][:-4]+'.npy')) #validate this logic in case filename convention changes
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index]))
        # convert img embedding, gt, bounding box of entire image to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.from_numpy(np.array([0,0,1023,1023])).float()



# dataset definition for hybrid training
class Dataset_hybrid(Dataset): 
    def __init__(self, sam_model, data_root, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda', return_embeddings=True):
        self.sam_model = sam_model
        self.mode = mode
        self.device = device
        self.return_embeddings = return_embeddings
        self.data_root = data_root
        self.image_dir_name = img_dir_name
        self.label_id_dir_name = label_id_dir_name
        self.img_embed_dir_name = img_embed_dir_name
        self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[:-100]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[-100:]
        print(f"{mode} dataset : {len(self.files)} --> {self.files[0]} -- {self.files[-1]}") 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = io.imread(join(self.data_root, self.image_dir_name, self.files[index][:-4]+'.png'))
        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        input_image_tensor = input_image_tensor.squeeze(0)


        img_embed = np.load(join(self.data_root, self.img_embed_dir_name, self.files[index][:-4]+'.npy')) #validate this logic in case filename convention changes
        gt2D = io.imread(join(self.data_root, self.label_id_dir_name, self.files[index][:-6]+'.png'))


        # binary mask from GT, at inference can be replaced by an off-the-shelf model prediction (e.g. SAM)
        binmask = gt2D>0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        binmask = cv2.morphologyEx(binmask.astype('uint8'),cv2.MORPH_CLOSE,kernel)

        # for bounding box
        Xs = np.where(binmask>0)[0]
        Ys = np.where(binmask>0)[1]
        
        # convert img embedding, gt, mask, bounding box to torch tensor
        if self.return_embeddings:
            return input_image_tensor, torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()
        else:
            return input_image_tensor, None, torch.tensor(gt2D[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()
