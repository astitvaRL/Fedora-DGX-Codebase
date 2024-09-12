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

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils.dataset import Dataset_body, Dataset_real_body
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentation
join = os.path.join


if __name__ == '__main__':
    torch.manual_seed(999)
    np.random.seed(999)

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    labels_definition_file_path = 'label_definition.json'
    ckpt_dir = './checkpoints'
    sam_original_ckpt_path = join(ckpt_dir,'sam_original/sam_vit_b_01ec64.pth')
    image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    second_image_dir_name = 'AD_SegMaps/drawings_synth_20k'
    label_id_dir_name = 'AD_SegMaps/labels_2k'

    COUNT = 0
    COUNT_MANIFOLD = 0
    names = sorted(os.listdir(join(data_root, label_id_dir_name)))
    for name in names:
        image_name = name.split('_')[0] + '.png'
        image_path = join(data_root, image_dir_name, image_name)
        if not os.path.exists(image_path):
            print(image_name)
            COUNT += 1
            # cmd = f"manifold -vip -cert ~/astitva_manifold.pem ls animated_drawings_images_prior_april22/tree/cropped_image/{image_name}"
            # try:
            #     os.system(cmd)
            # except:
            #     COUNT_MANIFOLD += 1
    # print(COUNT)
    # print(COUNT, COUNT_MANIFOLD)
