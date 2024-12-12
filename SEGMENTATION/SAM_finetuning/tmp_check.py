import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
# from skimage import io
import imageio as io
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision
from torchvision import transforms
import monai
import json
from monai.networks import one_hot
import torchmetrics
import segmentation_refinement as segref

from segment_anything_parallel import sam_model_registry
from segment_anything_parallel_fine_infer import sam_model_registry as sam_model_registry_fine

from segment_anything_parallel_fine_infer.utils.transforms import ResizeLongestSide

from utils.dataset import DrawingsDatasetInferFull
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationCoarse, SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationAll
from utils.augment import RandomAug

join = os.path.join


if __name__ == '__main__':

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    labels_dir_path = join(data_root, 'AD_SegMaps/labels_7k_1024/')
    remaining_label_dir_path = join(data_root, 'AD_SegMaps/labels_remaining/')
    images_dir_path = join(data_root, '/mnt/users_scratch/astitva/DATA/MANIFOLD/animated_drawings_images_prior_april22/cropped_image/')

    old_labels = sorted(os.listdir(labels_dir_path))
    labels = sorted(os.listdir(remaining_label_dir_path))

    all_labels = old_labels + labels

    COUNT = 0
    for label in tqdm(labels):
        img_name = f"{label.split('_')[0]}.png"
        img_path = join(images_dir_path, img_name)
        old_label_path = join(labels_dir_path, label)
        label_path = join(remaining_label_dir_path, label)
        if os.path.exists(old_label_path) and os.path.exists(label_path) :
            COUNT+=1
    print(COUNT)

    names = dict()
    ALL_COUNT = 0
    for label in tqdm(all_labels):
        img_name = f"{label.split('_')[0]}.png"
        img_path = join(images_dir_path, img_name)
        if os.path.exists(img_path):
            ALL_COUNT+=1
            try:
                print(names[label])
            except:
                names[label] = True

    print(ALL_COUNT)