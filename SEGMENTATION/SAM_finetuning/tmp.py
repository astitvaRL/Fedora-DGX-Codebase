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

from utils.SemanticSegmentation import SemanticSegmentationAll

join = os.path.join


if __name__ == '__main__':

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    label_id_dir_name = 'AD_SegMaps/labels_7k'
    syn_image_dir_name = 'AD_SegMaps/drawings_synth_20k'
    save_dir = join(data_root, 'MANIFOLD/animated_drawings_images_prior_april22/unlabelled_images/')
    os.makedirs(save_dir, exist_ok=True)

    label_names_dict = {}
    label_names = sorted(os.listdir(join(data_root, label_id_dir_name)))
    for name in tqdm(label_names):
        key = name.split('_')[0]
        label_names_dict[key] = True

    LABELLED_COUNT = 0
    UNLABELLED_COUNT = 0
    img_names = sorted(os.listdir(join(data_root, image_dir_name)))
    for name in tqdm(img_names):
        key = name.split('.')[0]
        try:
            if label_names_dict[key]:
                LABELLED_COUNT += 1
                continue
        except:
            UNLABELLED_COUNT += 1
            shutil.copyfile(join(data_root, image_dir_name, name), join(save_dir, name))
    

    print(f'LABELLED_COUNT: {LABELLED_COUNT}')
    print(f'UNLABELLED_COUNT: {UNLABELLED_COUNT}')
