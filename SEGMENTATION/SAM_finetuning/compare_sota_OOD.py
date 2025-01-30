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
    torch.manual_seed(999)
    np.random.seed(999)
    torch.multiprocessing.set_start_method('spawn')

    NUM_CLASSES = 27
    plot = True
    compute_metrics = False

    # define metrics
    MeanIoU = segmentation.MeanIoU(num_classes=NUM_CLASSES, include_background=True, per_class=True, input_format='index')
    GeneralizedDiceScore = segmentation.GeneralizedDiceScore(num_classes=NUM_CLASSES, include_background=True, per_class=True, input_format='index')
    MulticlassAccuracy = classification.MulticlassAccuracy(num_classes=NUM_CLASSES, average=None, ignore_index=0)
    
    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    labels_definition_file_path = 'label_definition.json'
    ckpt_dir = './checkpoints'
    sam_original_ckpt_path = join(ckpt_dir,'sam_original/sam_vit_b_01ec64.pth')
    image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    label_id_dir_name = 'AD_SegMaps/labels_16k'  


    SAPIENS_OOD_ROOT = "/mnt/users_scratch/astitva/DATA/SAPIENS_PREDICTIONS/OOD_paper/sapiens_0.3b/"
    SAM_SCRATCH_OOD_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/OOD_paper/SCRATCH_SAM/infer_16k_ANIMSEG_E2E_SCRATCH_ALL_CLASSES/best_eval_epoch"
    SAM_OOD_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/INFERENCE/OOD_paper/PRETRAINED/BEST/infer16k_ANIMSEG_E2E_FINE/best_eval_epoch/semantics"

    OUT_DIR = './COMPARISON/OOD_PAPER_whiteBG/'
    os.makedirs(OUT_DIR, exist_ok=True)

    semantics = SemanticSegmentationAll(num_classes=NUM_CLASSES, labels_definition_path=labels_definition_file_path)

    files = sorted(os.listdir(SAM_OOD_ROOT))

    for filename in tqdm(files):
        # input image
        input_im = cv2.imread(join(SAM_OOD_ROOT, f'{filename}/{filename}_original.png'))
        # sapiens
        exp1_seg = np.load(join(SAPIENS_OOD_ROOT, f'{filename}_seg.npy')).astype('uint8')
        exp1_seg = semantics.labels_to_colors(exp1_seg)
        exp1_seg =  cv2.cvtColor(exp1_seg, cv2.COLOR_BGR2RGB)
        # ours scratch
        exp2_seg = cv2.imread(join(SAM_SCRATCH_OOD_ROOT, f'{filename}_seg.png'))
        #ours 
        exp3_seg = cv2.imread(join(SAM_OOD_ROOT, f'{filename}/{filename}_seg_fine_masked.png'))

        input_im =  cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
        exp1_seg =  cv2.cvtColor(exp1_seg, cv2.COLOR_BGR2RGB)
        exp2_seg =  cv2.cvtColor(exp2_seg, cv2.COLOR_BGR2RGB)
        exp3_seg =  cv2.cvtColor(exp3_seg, cv2.COLOR_BGR2RGB)

        input_im = cv2.resize(input_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
        exp1_seg = cv2.resize(exp1_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
        exp2_seg = cv2.resize(exp2_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
        exp3_seg = cv2.resize(exp3_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)

        exp1_seg[exp1_seg.sum(2)==0] = [255,255,255]
        exp2_seg[exp2_seg.sum(2)==0] = [255,255,255]
        exp3_seg[exp3_seg.sum(2)==0] = [255,255,255]

        # plot
        TITLE_SIZE = 30
        fig, ax = plt.subplots(1,4, figsize=(80,20))
        fig.tight_layout()
        ax[0].imshow(input_im)
        ax[0].axis('off')
        ax[1].imshow(exp1_seg)
        ax[1].axis('off')
        ax[2].imshow(exp2_seg)
        ax[2].axis('off')
        ax[3].imshow(exp3_seg)
        ax[3].axis('off')
        plt.savefig(f"{OUT_DIR}/{filename}_comparison_OOD.png")
        plt.close()

