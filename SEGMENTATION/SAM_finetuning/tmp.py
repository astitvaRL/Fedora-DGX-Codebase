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

from collections.abc import Generator

import cv2 as cv
import numpy as np


def bezier(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Generator[np.ndarray, None, None]:
    def calc(t):
        return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3
    # get the approximate pixel count of the curve
    approx = cv.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
    for t in np.linspace(0, 1, round(approx * 1.2)):
        yield np.round(calc(t)).astype(np.int32)
def generate_scratch(img: np.ndarray, max_length: float, end_brush_range: tuple[float, float], mid_brush_range: tuple[float, float]) -> np.ndarray:
    H, W = img.shape
    # generate the 2 end points of the bezier curve
    x, y, rho1, theta1 = np.random.uniform([0] * 4, [W, H, max_length, np.pi * 2])
    p1 = np.array([x, y, 0])
    p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]
    # generate the second point, make sure that it cannot be too far away from the middle point of the 2 end points
    rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
    p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]
    # generate the brush sizes of the 3 points
    p1[2], p2[2], p3[2] = np.random.uniform(*np.transpose([end_brush_range, mid_brush_range, end_brush_range]))
    for x, y, brush in bezier(p1, p2, p3):
        cv.circle(img, (x, y), brush, 255, -1)
    return img

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
    semantics = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=18)

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
        gt2D_labels = semantics.colors_to_labels(gt2D)
        MAX_LENGTH = 512  # maximum distance between two end points
        END_BRUSH_RANGE = (0, 1)  # brush size range of the two end points
        MID_BRUSH_RANGE = (2, 5)  # brush size range of the mid point
        SCRATCH_CNT = 200

        breakpoint()
        strokes = np.zeros_like(gt2D_labels)
        for _ in range(SCRATCH_CNT):
            generate_scratch(strokes, MAX_LENGTH, END_BRUSH_RANGE, MID_BRUSH_RANGE)
        
        mask = (strokes>0)&(gt2D_labels>0)
        img[mask] = gt2D[mask]
        cv2.imwrite('zi.png',img)
    