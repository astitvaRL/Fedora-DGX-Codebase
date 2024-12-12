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
from shutil import copyfile

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
    labels_dir_path = join(data_root, 'AD_SegMaps/labels_16k/')
    img_dir_path = join(data_root, 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image/')
    save_root = join(data_root, 'LIP_drawings_16k/')
    split = 'train'
    save_img_dir_path = join(save_root,f'images/{split}_images/')
    save_label_dir_path = join(save_root,f'segmentations/{split}_segmentations/')
    os.makedirs(save_img_dir_path, exist_ok=True)
    os.makedirs(save_label_dir_path, exist_ok=True)

    labels_definition_file_path = 'label_definition.json'

    # semantic definitions
    num_classes_coarse = 6 # coarse network also includes 'Neck' class which will be merged to torso before feeding to fine network if required
    num_classes_fine = 18
    num_classes_face = 11
    num_classes_all = 27
    exclude_neck_from_coarse = True
    semantics_coarse = SemanticSegmentationCoarse(labels_definition_path=labels_definition_file_path, num_classes=(num_classes_coarse-1) if exclude_neck_from_coarse else num_classes_coarse, exclude_neck=True)
    semantics_fine = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=num_classes_fine)
    semantics_face = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=num_classes_face)
    semantics_all = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=num_classes_all)

    labels = sorted(os.listdir(labels_dir_path))[:-2000]

    for labelname in tqdm(labels):
        label_path = join(labels_dir_path, labelname)
        label_im = cv2.imread(label_path)
        label_im = cv2.cvtColor(label_im, cv2.COLOR_BGR2RGB)
        label_id = semantics_all.colors_to_labels(label_im)
        cv2.imwrite(join(save_label_dir_path, labelname), label_id)
        image_name = f"{labelname.split('_')[0]}.png"
        src_img_path = join(img_dir_path, image_name)
        dst_img_path = join(save_img_dir_path, labelname)
        copyfile(src_img_path,dst_img_path)




