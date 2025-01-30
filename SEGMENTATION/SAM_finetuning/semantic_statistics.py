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

from utils.SemanticSegmentation import SemanticSegmentationAll

join = os.path.join

def normalize_classname(name):
    name = name.replace('Other_', '')
    name = name.replace('_', ' ')
    name = name.title()
    return name


if __name__ == '__main__':

    ignore_bg = True
    recompute = False

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    label_id_dir_name = 'AD_SegMaps/labels_16k'
    labels_definition_file_path = 'label_definition.json'
    num_classes = 27
    semantics = SemanticSegmentationAll(labels_definition_file_path, num_classes=num_classes)


    if recompute:
        ID_COUNT = [0]*num_classes
        ID_PIXEL_COUNT = [0]*num_classes
        names = sorted(os.listdir(join(data_root, label_id_dir_name)))
        
        for name in tqdm(names):
            label_image = io.imread(join(data_root, label_id_dir_name, name))
            label_id = semantics.colors_to_labels(label_image)
            ids = np.unique(label_id)
            if np.sum(label_id==255)>0:
                breakpoint()
            for x in ids:
                ID_COUNT[x] += 1
                ID_PIXEL_COUNT[x] += np.sum(label_id==x)

        np.save('ID_stats.npy', ID_COUNT)
        np.save('ID_PIXEL_stats.npy', ID_PIXEL_COUNT)


    idcount = np.load('ID_stats.npy')
    pixelcount = np.load('ID_PIXEL_stats.npy')

    name2id = semantics.data['label_name_to_id']
    label_dict = {}
    id2name = {value: key for key, value in name2id.items()}

    labels = [normalize_classname(id2name[i]) for i in range(1,num_classes-1)]
    labels.insert(0, 'Background')
    labels.insert(len(labels), 'Unknown')
    colors = np.array([semantics.data['label_name_to_color'][id2name[i]] for i in range(num_classes)])/255

    if ignore_bg:
        labels = labels[1:-1]
        idcount = idcount[1:-1]
        pixelcount = pixelcount[1:-1]
        colors = colors[1:-1]

    offset = 100
    plt.figure(figsize=(50,20))
    plt.bar(labels, idcount, color=colors)
    plt.xticks(fontsize=18)
    plt.yticks([])
    for i, count in enumerate(idcount):
        plt.text(i, count+offset, str(count), ha='center', fontsize=22)
    plt.tight_layout()
    plt.savefig(f'STATS/STATS_IMAGE_COUNT.svg')
    plt.savefig(f'STATS/STATS_IMAGE_COUNT.png')
    plt.close()

    offset = 1000000
    plt.figure(figsize=(50,20))
    plt.bar(labels, pixelcount, color=colors)
    plt.xticks(fontsize=18)
    plt.yticks([])
    for i, count in enumerate(pixelcount):
        plt.text(i, count+offset, str(count), ha='center', fontsize=22)
    plt.tight_layout()
    plt.savefig(f'STATS/STATS_PIXEL_COUNT.svg')
    plt.savefig(f'STATS/STATS_PIXEL_COUNT.png')
    plt.close()
