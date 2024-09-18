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


if __name__ == '__main__':

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    label_id_dir_name = 'AD_SegMaps/labels_2k_1024'
    syn_image_dir_name = 'AD_SegMaps/drawings_synth_17k_v3'
    save_dir = join(data_root, 'SYN_VIS_v3')
    os.makedirs(save_dir, exist_ok=True)

    TITLE_SIZE = 30
    NUM_SYN = 6
    names = sorted(os.listdir(join(data_root, syn_image_dir_name)))
    for name in tqdm(names):
        if not name.endswith('_0.png'):
            continue
        label_name = name[:-6] + '.png'
        label_path = join(data_root, label_id_dir_name, label_name)
        img_name = label_name.split('_')[0] + '.png'
        img_path = join(data_root, image_dir_name, img_name)
        # check_name = label_name[:-8] + f'{0}.png'
        # check_path = join(data_root, syn_image_dir_name, check_name)
        # if not os.path.exists(check_path):
        #     continue
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024))
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1,NUM_SYN+2, figsize=((NUM_SYN+2)*10,10))
        ax[0].imshow(image)
        # ax[0].set_title("Original Image", fontsize=TITLE_SIZE)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].imshow(label)
        # ax[1].set_title("Semantic Map", fontsize=TITLE_SIZE)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        for i in range(NUM_SYN):
            syn_image_name = label_name[:-4] + f'_{i}.png'
            syn_image_path = join(data_root, syn_image_dir_name, syn_image_name)
            syn_image = cv2.imread(syn_image_path)
            syn_image = cv2.cvtColor(syn_image, cv2.COLOR_BGR2RGB)
            syn_image = cv2.resize(syn_image, (1024, 1024))
            ax[i+2].imshow(syn_image)
            # ax[i+2].set_title(f"Synthetic Image {i+1}", fontsize=TITLE_SIZE)
            ax[i+2].set_xticks([])
            ax[i+2].set_yticks([])
        plt.tight_layout()
        plt.savefig(join(save_dir, f'{label_name}'))
        plt.close()
