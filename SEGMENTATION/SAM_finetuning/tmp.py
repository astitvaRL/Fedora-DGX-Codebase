import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import io
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision
from torchvision import transforms
import monai
import json
from monai.networks import one_hot
import shutil

from utils.SemanticSegmentation import SemanticSegmentationNoFace

join = os.path.join


if __name__ == '__main__':

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    image_dir_name = 'MANIFOLD/EVAL400'
    label_id_dir_name = 'AD_SegMaps/labels_EVAL400' 

    save_dir = join(data_root, 'LIP_drawings/')
    os.makedirs(save_dir, exist_ok=True)
    # train_seg_dir =  join(save_dir, 'segmentations/train_segmentations')
    # train_img_dir = join(save_dir, 'images/train_images')
    # os.makedirs(train_seg_dir, exist_ok=True)
    # os.makedirs(train_img_dir, exist_ok=True)
    val_seg_dir = join(save_dir, 'segmentations/val_segmentations')
    val_img_dir = join(save_dir, 'images/val_images')
    os.makedirs(val_seg_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    img_names = sorted(os.listdir(join(data_root, image_dir_name)))
    label_names = sorted(os.listdir(join(data_root, label_id_dir_name)))

    labels_definition_file_path = 'label_definition.json'
    semantic = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=18)

    for name in tqdm(label_names):
        imgname = name.split('_')[0] + '.png'
        imgpath = join(data_root, image_dir_name, imgname)
        save_imagepath = join(val_img_dir, name)
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_imagepath, img)
        labelpath = join(data_root, label_id_dir_name, name)
        save_labelpath = join(val_seg_dir, name)
        gt2D = cv2.imread(labelpath)
        gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
        gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)
        gt2D_labels = semantic.colors_to_labels(gt2D)
        cv2.imwrite(save_labelpath, gt2D_labels)
    