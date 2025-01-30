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
        
        names = sorted(os.listdir(join(data_root, label_id_dir_name)))
        ALL_IMAGES_PERC = []
        for name in tqdm(names):
            label_image = io.imread(join(data_root, label_id_dir_name, name))
            label_id = semantics.colors_to_labels(label_image)
            ids = np.unique(label_id)
            total_pixels = np.sum(label_id>0)
            ID_PIXEL_PERC = [0]*num_classes
            for x in ids:
                ID_PIXEL_PERC[x] = np.sum(label_id==x)/total_pixels
            ALL_IMAGES_PERC.append(ID_PIXEL_PERC)

        ALL_IMAGES_PERC = np.array(ALL_IMAGES_PERC)
        np.save('ID_PIXEL_percentage_stats.npy', ALL_IMAGES_PERC)


    pixelperc_all_classes = np.load('ID_PIXEL_percentage_stats.npy')
    
    pixel_perc = pixelperc_all_classes
    mean_perc = np.sum(pixel_perc, 0)/len(pixel_perc)
    var_perc = np.sum((pixel_perc-mean_perc)**2,0)/(len(pixel_perc)-1)
    std_perc = np.sqrt(var_perc)

    pixel_perc = np.round(pixel_perc*100, 2)
    mean_perc = np.round(mean_perc*100, 2)
    var_perc = np.round(var_perc*100, 2)
    std_perc = np.round(std_perc*100, 2)

    name2id = semantics.data['label_name_to_id']
    label_dict = {}
    id2name = {value: key for key, value in name2id.items()}

    labels = [normalize_classname(id2name[i]) for i in range(1,num_classes-1)]
    labels.insert(0, 'Background')
    labels.insert(len(labels), 'Unknown')
    colors = np.array([semantics.data['label_name_to_color'][id2name[i]] for i in range(num_classes)])/255

    if ignore_bg:
        labels = labels[1:-1]
        pixel_perc = pixel_perc[1:-1]
        mean_perc = mean_perc[1:-1]
        var_perc = var_perc[1:-1]
        std_perc = std_perc[1:-1]
        colors = colors[1:-1]
    
    offset = 0.2
    plt.figure(figsize=(50,20))
    plt.bar(labels, mean_perc, color=colors)
    plt.xticks(fontsize=18)
    plt.yticks([])
    for i, perc in enumerate(mean_perc):
        plt.text(i, perc+offset, f'{perc}%', ha='center', fontsize=22)
    plt.tight_layout()
    plt.savefig(f'STATS/STATS_MEAN_PERCENTAGE.svg')
    plt.savefig(f'STATS/STATS_MEAN_PERCENTAGE.png')
    plt.close()

    offset = 0.2
    plt.figure(figsize=(50,20))
    plt.bar(labels, std_perc, color=colors)
    plt.xticks(fontsize=18)
    plt.yticks([])
    for i, perc in enumerate(std_perc):
        plt.text(i, perc+offset, f'{perc}', ha='center', fontsize=22)
    plt.tight_layout()
    plt.savefig(f'STATS/STATS_STD_PERCENTAGE.svg')
    plt.savefig(f'STATS/STATS_STD_PERCENTAGE.png')
    plt.close()
