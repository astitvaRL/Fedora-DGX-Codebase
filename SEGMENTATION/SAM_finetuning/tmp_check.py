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
    dir1 = join(data_root, 'LIP_drawings/segmentations/train_segmentations/')
    dir2 = join(data_root, 'LIP_drawings/segmentations/val_segmentations/')

    labels1 = sorted(os.listdir(dir1))
    labels2 = sorted(os.listdir(dir2))

    for lab1 in tqdm(labels1):
        for lab2 in labels2:
            if lab1==lab2:
                print(lab1, lab2)