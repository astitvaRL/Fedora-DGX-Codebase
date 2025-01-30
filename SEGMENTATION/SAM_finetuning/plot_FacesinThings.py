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
from torchmetrics import segmentation, classification

from segment_anything_parallel import sam_model_registry
from segment_anything_parallel_fine_infer import sam_model_registry as sam_model_registry_fine

from segment_anything_parallel_fine_infer.utils.transforms import ResizeLongestSide

from utils.dataset import DrawingsDatasetC2FAll
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationCoarse, SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationAll
from utils.augment import RandomAug

join = os.path.join

if __name__ == '__main__':
    torch.manual_seed(999)
    np.random.seed(999)
    torch.multiprocessing.set_start_method('spawn')

    NUM_CLASSES = 27
    plot = True

    # set paths
    root = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/FacesInThings/infer12k_ANIMSEG_E2E_FINE/best_eval_epoch/for_editing/'
    images_dir = join(root,"images")

    OUT_DIR = './COMPARISON/FacesInThings/'
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(os.listdir(images_dir))

    for filename in tqdm(files):
        if True:
            input_im = cv2.imread(join(images_dir, f'{filename}'))
            exp_seg = cv2.imread(join(root, f'vis/{filename}'))

            input_im =  cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
            exp_seg =  cv2.cvtColor(exp_seg, cv2.COLOR_BGR2RGB)
            exp_seg[exp_seg.sum(2)==0] = [255, 255, 255]

            # plot
            TITLE_SIZE = 30
            fig, ax = plt.subplots(1,2, figsize=(40,20))
            fig.tight_layout()
            ax[0].imshow(input_im)
            ax[0].axis('off')
            ax[1].imshow(exp_seg)
            ax[1].axis('off')
            plt.savefig(f"{OUT_DIR}/{filename}_plot.png")
            plt.close()
