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
    image_dir = join(data_root,'MANIFOLD/EVAL400')
    label_dir = join(data_root, 'AD_SegMaps/labels_EVAL400')
    src_image_dir = join(data_root,'MANIFOLD/animated_drawings_images_prior_april22/cropped_image')


    count = 0
    label_names = sorted(os.listdir(join(data_root, label_dir)))
    for name in tqdm(label_names):
        imgname = name.split('_')[0] + '.png'
        if not os.path.exists(join(image_dir, imgname)):
            count += 1
            savepath = join(data_root, image_dir, imgname)
            print(os.path.exists(savepath))
            shutil.copyfile(join(src_image_dir, imgname), savepath)
    print(count)
