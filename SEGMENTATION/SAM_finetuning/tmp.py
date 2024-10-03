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
    save_image_dir = join(data_root, 'MANIFOLD/EVAL400')
    label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 
    save_label_dir = join(data_root, 'AD_SegMaps/labels_EVAL400')

    count = 0
    img_names = sorted(os.listdir(join(data_root, image_dir_name)))
    label_names = sorted(os.listdir(join(data_root, label_id_dir_name)))[-100:]
    for name in tqdm(label_names):
        imgname = name.split('_')[0] + '.png'
        imgpath = join(data_root, image_dir_name, imgname)
        save_imagepath = join(save_image_dir, imgname)
        if not os.path.exists(save_imagepath):
            print(save_imagepath)
            count += 1
            # shutil.copyfile(imgpath, join(save_image_dir, imgname))
            # shutil.copyfile(join(data_root, label_id_dir_name, name), join(save_label_dir, name))
    print(count)
