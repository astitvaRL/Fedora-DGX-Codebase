import sys
sys.path.append('../')

import os
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from skimage import io
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
join = os.path.join

from segment_anything.utils.transforms import ResizeLongestSide
from .SemanticSegmentation import SemanticSegmentation


# dataset definition for only body (no facial details)
class Dataset_body(Dataset): 
    def __init__(self, sam_model, data_root, labels_definition_file_path, img_dir_name, img_embed_dir_name, label_id_dir_name, mode='train', device='cuda', return_embeddings=True):
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
        self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[:-2000][:20]
        if self.mode == 'test':
            self.files = sorted(os.listdir(join(self.data_root, self.img_embed_dir_name)))[-2000:][:20]
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
        r,g,b = cv2.split(gt2D)
        binmask = ~((r==0) & (g==0) & (b==0))
        binmask = np.uint8(binmask)

        # colors to labels
        gt2D = self.semantics.remove_interpolation_artifacts(gt2D)
        gtID = self.semantics.colors_to_labels(gt2D)
        gtID_vis = self.semantics.labels_to_colors(gtID)

        # for bounding box
        Xs = np.where(binmask>0)[0]
        Ys = np.where(binmask>0)[1]
        
        # convert img embedding, gt, mask, bounding box to torch tensor
        if self.return_embeddings:
            return input_image_tensor, torch.tensor(img_embed).float(), torch.tensor(gtID[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()
        else:
            return input_image_tensor, None, torch.tensor(gtID[None, :,:]).long(), torch.tensor(binmask[None, :,:]).float(), torch.from_numpy(np.array([min(Ys),min(Xs),max(Ys),max(Xs)])).float()
