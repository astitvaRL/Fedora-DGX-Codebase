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


from utils.SemanticSegmentation import SemanticSegmentationAll

join = os.path.join

def get_features(mask, img):
    img[mask==0] = [0,0,0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return -1
    contour = contours[0]
    # Calculate area
    area = cv2.contourArea(contour)
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)
    # Calculate aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    # Calculate circularity
    circularity = 4 * np.pi * (area / (perimeter**2))
    # Calculate convexity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0
    # Calculate eccentricity
    moments = cv2.moments(contour)
    major_axis_length = np.sqrt((moments['mu20'] + moments['mu02']) / area)
    minor_axis_length = np.sqrt((moments['mu20'] - moments['mu02']) / area)
    eccentricity = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0
    # Calculate solidity
    solidity = area / (w * h)
    # Calculate extent
    extent = area / (w * h)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gabor filter
    gabor_filter = cv2.getGaborKernel((15, 15), 3.0, 0, 3.0, 0.5)
    gabor_response = cv2.filter2D(gray, -1, gabor_filter)
    gabor_feature = np.mean(gabor_response)
    # Apply HOG
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    hog_feature = np.mean(hog_features)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    canny_feature = np.mean(edges)

    # Append features to the list
    statistical_features = [area, perimeter, aspect_ratio, circularity, convexity, eccentricity, solidity, extent, gabor_feature, hog_feature]
    if math.isnan(sum(statistical_features)):
        return -1
    return statistical_features


if __name__ == '__main__':

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 

    save_dir = join(data_root, 'AD_SegMaps/mouth_clusters/')
    os.makedirs(save_dir, exist_ok=True)


    label_names = sorted(os.listdir(join(data_root, label_id_dir_name)))

    labels_definition_file_path = 'label_definition.json'
    semantic = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=27)

    padding = 0

    all_features = []
    all_paths = []
    all_BBs = []
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

        face_region = label_id == 6
        Xs, Ys = np.where(face_region)
        if len(Xs)==0 or len(Ys)==0:
            continue
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
        label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        mouth_mask = (label_id == 3) | (label_id == 23) | (label_id == 17)
        mouth_mask = mouth_mask.astype('uint8')*255
        features = get_features(mouth_mask, img_cropped)
        if features==-1:
            continue
        all_features.append(features)
        all_paths.append(imgpath)
        all_BBs.append([x_min,x_max,y_min,y_max])

all_features = np.array(all_features)
all_paths = np.array(all_paths)
all_BBs = np.array(all_BBs)

# normalize features
max_values = np.max(all_features, axis=0)
normalized_features = all_features / max_values

NUM_CLUSTERS = 5
kmeans = KMeans(n_clusters=NUM_CLUSTERS, max_iter=1000)
kmeans.fit(all_features)
labels = kmeans.labels_

for i in range(NUM_CLUSTERS):
    cluster_dir = join(save_dir, str(i))
    os.makedirs(cluster_dir, exist_ok=True)
    cluster_id = labels==i
    img_paths = all_paths[cluster_id]
    img_bbs = all_BBs[cluster_id]
    for img_idx in tqdm(range(len(img_paths)), f"Cluster-{i}"):
        src_imgpath = img_paths[img_idx]
        x_min, x_max, y_min, y_max = img_bbs[img_idx]
        savename = src_imgpath.split('/')[-1]
        dst_imgpath = join(cluster_dir, savename)
        img = cv2.imread(src_imgpath)
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR)
        img = img[x_min:x_max, y_min:y_max]
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dst_imgpath, img)
        