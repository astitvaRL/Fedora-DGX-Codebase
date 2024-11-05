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
import numpy as np
from sklearn.cluster import KMeans
import math

from utils.SemanticSegmentation import SemanticSegmentationFace

join = os.path.join


if __name__ == '__main__':

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    # image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    image_dir_name = '/mnt/users_scratch/astitva/DATA/MANIFOLD/EVAL400/'
    # label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 
    label_id_dir_name = '/mnt/users_scratch/astitva/DATA/AD_SegMaps/labels_EVAL400/'

    save_dir = join(data_root, 'MANIFOLD/EVAL_400_cropped_faces')
    save_label_id_dir = join(data_root, 'AD_SegMaps/labels_cropped_faces_EVAL_400')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_label_id_dir, exist_ok=True)


    label_names = sorted(os.listdir(join(data_root, label_id_dir_name)))

    labels_definition_file_path = 'label_definition.json'
    semantic = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=11)

    padding = 30

    for name in tqdm(label_names):
        imgname = name.split('_')[0] + '.png'
        imgpath = join(data_root, image_dir_name, imgname)
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR)
        labelpath = join(data_root, label_id_dir_name, name)
        gt2D = cv2.imread(labelpath)
        gt2D = cv2.cvtColor(gt2D, cv2.COLOR_BGR2RGB)
        gt2D = cv2.resize(gt2D, (1024,1024), interpolation=cv2.INTER_NEAREST)

        label_id = semantic.colors_to_labels(gt2D)

        face_region = label_id > 0
        Xs, Ys = np.where(face_region)
        if len(Xs)==0 or len(Ys)==0:
            continue
            print("Skipping...")
        x_min, x_max = np.min(Xs) - padding, np.max(Xs) + padding
        y_min, y_max = np.min(Ys) - padding, np.max(Ys) + padding
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > 1024:
            x_max = 1024
        if y_max > 1024:
            y_max = 1024

        img_cropped = img[x_min:x_max, y_min:y_max]
        img_cropped = cv2.resize(img_cropped, (1024, 1024))
        label_id = label_id[x_min:x_max, y_min:y_max]
        gt2D_cropped = gt2D[x_min:x_max, y_min:y_max]
        label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        gt2D_cropped = cv2.resize(gt2D_cropped, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        gt2D_cropped = cv2.cvtColor(gt2D_cropped, cv2.COLOR_BGR2RGB)
        gt2D_cropped[gt2D_cropped.sum(2)==0] = [255,255,255]

        save_imgpath = join(save_dir, name)
        save_labelpath = join(save_label_id_dir, name)
        # cv2.imwrite(save_imgpath, img_cropped)
        # cv2.imwrite(save_labelpath, label_id)
        cv2.imwrite(save_labelpath, gt2D_cropped)
        