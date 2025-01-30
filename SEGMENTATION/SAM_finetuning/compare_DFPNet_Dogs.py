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
import natsort

from segment_anything_parallel import sam_model_registry
from segment_anything_parallel_fine_infer import sam_model_registry as sam_model_registry_fine

from segment_anything_parallel_fine_infer.utils.transforms import ResizeLongestSide

from utils.dataset import DrawingsDatasetC2FAll
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationCoarse, SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationAll
from utils.augment import RandomAug

join = os.path.join

if __name__ == '__main__':

    NUM_CLASSES = 27
    labels_definition_file_path = './label_definition.json'

    # set paths
    image_dir_name = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/DFPNet-Drawings/Cartoon_sketches/Dog/val_images"
    GT_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/DFPNet-Drawings/Cartoon_sketches/Dog/val_segmentations"

    DFP_NET_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/DFPNet-Drawings/PREDICTIONS/dogs_val_named/"
    SAM_DOGS_SCRATCH_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/CartoonDogsPretrained_VAL/infer_DOGS_ANIMSEG_E2E_SAMEncoder_DOGS/201/"
    # SAM_DOGS_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/CartoonDogs_TEST_ep200/infer_DOGS_ANIMSEG_E2E_DrawingsEncoder_DOGS/200/"
    # SAM_DOGS_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/CartoonDogsPretrained_VAL/infer_DOGS_CartoonDogs_pretrainedSAMFine14k/best_train_epoch/"
    SAM_DOGS_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/CartoonDogsPretrained_VAL/infer_DOGS_CartoonDogs_pretrainedSAMFine14k_to_plot/best_train_epoch/"
    FINE_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/CartoonDogs_with_prior/infer_16k_ANIMSEG_E2E_FINE/500/"

    OUT_DIR = './COMPARISON/DOGS_ALL_to_plot/'
    os.makedirs(OUT_DIR, exist_ok=True)

    semantics = SemanticSegmentationAll(num_classes=NUM_CLASSES, labels_definition_path=labels_definition_file_path)

    files = sorted(os.listdir(image_dir_name))

    for filename in tqdm(files):
        # try:
        if True:
            # input image
            input_im = cv2.imread(join(image_dir_name, f'{filename}'))
            exp1_seg = cv2.imread(join(DFP_NET_ROOT, f'{filename[:-4]}.png'))
            exp3_id = cv2.imread(join(SAM_DOGS_ROOT, f'{filename[:-4]}_id.png'))
            exp2_id = cv2.imread(join(SAM_DOGS_SCRATCH_ROOT, f'{filename[:-4]}_id.png'))
            gt_id = cv2.imread(join(GT_ROOT, f'{filename[:-4]}.png'))
            fine_sem = cv2.imread(join(FINE_ROOT, f'{filename[:-4]}_fine.png'))
            
            exp2_id = exp2_id[:,:,0]
            exp3_id = exp3_id[:,:,0]
            gt_id = gt_id[:,:,0]

            input_im = cv2.resize(input_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
            exp1_seg = cv2.resize(exp1_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp2_id = cv2.resize(exp2_id, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp3_id = cv2.resize(exp3_id, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt_id = cv2.resize(gt_id, (1024,1024), interpolation=cv2.INTER_NEAREST)
            fine_sem = cv2.resize(fine_sem, (1024,1024), interpolation=cv2.INTER_NEAREST)

            # red channel values for dfpnet predictions
            head = 177 #1
            torso = 68 #2
            left_arm = 21 #3
            right_arm = 63 #4
            left_leg = 171 #5
            right_leg = 12 #6
            tail = 245 #7

            exp1_id = np.zeros_like(exp3_id)
            b,g,r = cv2.split(exp1_seg)
            exp1_id[r==177] = 1
            exp1_id[r==68] = 2
            exp1_id[r==21] = 3
            exp1_id[r==63] = 4
            exp1_id[r==171] = 5
            exp1_id[r==12] = 6
            exp1_id[r==245] = 7

            gt_sem = np.zeros((1024,1024,3)).astype('uint8')
            exp1_sem = np.zeros((1024,1024,3)).astype('uint8')
            exp2_sem = np.zeros((1024,1024,3)).astype('uint8')
            exp3_sem = np.zeros((1024,1024,3)).astype('uint8')

            colors = [
            [255,255,255], #0
            [255,50,50], #1
            [50,255,50], #2
            [255,255,50], #3
            [255,50,255], #4
            [50,128,255], #5
            [50,50,255], #6
            [128,128,128], #7
            ]
            for cidx in range(8):
                exp1_sem[exp1_id==cidx] = colors[cidx]
                exp2_sem[exp2_id==cidx] = colors[cidx]
                exp3_sem[exp3_id==cidx] = colors[cidx]
                gt_sem[gt_id==cidx] = colors[cidx]

            input_im =  cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
            exp1_sem =  cv2.cvtColor(exp1_sem, cv2.COLOR_BGR2RGB)
            exp2_sem =  cv2.cvtColor(exp2_sem, cv2.COLOR_BGR2RGB)
            exp3_sem =  cv2.cvtColor(exp3_sem, cv2.COLOR_BGR2RGB)
            gt_sem =  cv2.cvtColor(gt_sem, cv2.COLOR_BGR2RGB)

            exp1_sem[exp1_sem.sum(2)==0] = [255,255,255]
            exp2_sem[exp2_sem.sum(2)==0] = [255,255,255]
            exp3_sem[exp3_sem.sum(2)==0] = [255,255,255]
            fine_sem[fine_sem.sum(2)==0] = [255,255,255]
            gt_sem[gt_sem.sum(2)==0] = [255,255,255]

            # plot
            TITLE_SIZE = 30
            fig, ax = plt.subplots(1,6, figsize=(30,5))
            fig.tight_layout()
            ax[0].imshow(input_im)
            ax[0].axis('off')
            ax[1].imshow(exp1_sem)
            ax[1].axis('off')
            ax[2].imshow(exp2_sem)
            ax[2].axis('off')
            ax[3].imshow(exp3_sem)
            ax[3].axis('off')
            ax[4].imshow(gt_sem)
            ax[4].axis('off')
            ax[5].imshow(fine_sem)
            ax[5].axis('off')
            plt.savefig(f"{OUT_DIR}/{filename}_plot.png")
            plt.close()

        # except:
        #     continue

