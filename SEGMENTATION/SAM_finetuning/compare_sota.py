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


    SAM_EVAL_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/EVALUATION/"
    SAPIENS_EVAL_ROOT = "/mnt/users_scratch/astitva/DATA/LIP_drawings_16k/sapiens_val_prediction_400/sapiens_0.3b/"
    DFPNET_EVAL_ROOT = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/DFPNet-Drawings/Cartoon_sketches/Drawings/save_preds_495_1024/'
    MASK2FORMER_EVAL_ROOT = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/MaskFormer/Mask2Former/demo/results/EVAL_DRAWINGS_FINAL_wBG/'

    exp1_out_dir = join(DFPNET_EVAL_ROOT)
    exp2_out_dir = join(SAPIENS_EVAL_ROOT)
    exp3_out_dir = join(MASK2FORMER_EVAL_ROOT)
    exp4_out_dir = join(SAM_EVAL_ROOT,"eval_16k_DecoderOnly_SINGLE_STAGE/best_eval/")
    exp5_out_dir = join(SAM_EVAL_ROOT,"16k_ANIMSEG_E2E_ALL_CLASSES_SlowLR/best/best_eval/")
    exp6_out_dir = join(SAM_EVAL_ROOT,"eval_16k_E2E_FULL/best_eval/")

    OUT_DIR = './COMPARISON/FOR_PAPER/'
    os.makedirs(OUT_DIR, exist_ok=True)

    semantics = SemanticSegmentationAll(num_classes=NUM_CLASSES, labels_definition_path=labels_definition_file_path)

    files = sorted(os.listdir(MASK2FORMER_EVAL_ROOT))

    # dfpnet_files = sorted(os.listdir(DFPNET_EVAL_ROOT))

    fidx = 0
    for filename in tqdm(files):
        if filename.endswith('_seg_BG.png'):
            filename_original = filename[:-11]
            filename_base = filename_original.split('_')[0]
            # input image
            input_im = cv2.imread(join(data_root, image_dir_name, f'{filename_base}.png'))
            # dfpnet
            exp1_seg = cv2.imread(join(exp1_out_dir, f'{fidx}.png'))
            fidx += 1
            # sapiens
            exp2_seg = np.load(join(exp2_out_dir, f'{filename_original}_seg.npy')).astype('uint8')
            exp2_seg = semantics.labels_to_colors(exp2_seg)
            exp2_seg =  cv2.cvtColor(exp2_seg, cv2.COLOR_BGR2RGB)
            #mask2former
            exp3_seg = cv2.imread(join(exp3_out_dir, f'{filename}'))
            #ours decoder only
            exp4_seg = cv2.imread(join(exp4_out_dir, f'{filename_base}/pred.png'))
            # ours e2e single stage
            exp5_seg = cv2.imread(join(exp5_out_dir, f'{filename_base}/pred.png'))
            #ours e2e hierarchical
            exp6_seg = cv2.imread(join(exp6_out_dir, f'{filename_base}/pred_all.png'))
            #gt
            gt_seg = cv2.imread(join(exp6_out_dir, f'{filename_base}/gt_all.png'))

            input_im =  cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
            exp1_seg =  cv2.cvtColor(exp1_seg, cv2.COLOR_BGR2RGB)
            exp2_seg =  cv2.cvtColor(exp2_seg, cv2.COLOR_BGR2RGB)
            exp3_seg =  cv2.cvtColor(exp3_seg, cv2.COLOR_BGR2RGB)
            exp4_seg =  cv2.cvtColor(exp4_seg, cv2.COLOR_BGR2RGB)
            exp5_seg =  cv2.cvtColor(exp5_seg, cv2.COLOR_BGR2RGB)
            exp6_seg =  cv2.cvtColor(exp6_seg, cv2.COLOR_BGR2RGB)
            gt_seg =  cv2.cvtColor(gt_seg, cv2.COLOR_BGR2RGB)

            input_im = cv2.resize(input_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
            exp1_seg = cv2.resize(exp1_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp2_seg = cv2.resize(exp2_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp3_seg = cv2.resize(exp3_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp4_seg = cv2.resize(exp4_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp5_seg = cv2.resize(exp5_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp6_seg = cv2.resize(exp6_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt_seg = cv2.resize(gt_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)

            exp1_seg[exp1_seg.sum(2)==0] = [255,255,255]
            exp2_seg[exp2_seg.sum(2)==0] = [255,255,255]
            exp3_seg[exp3_seg.sum(2)==0] = [255,255,255]
            exp4_seg[exp4_seg.sum(2)==0] = [255,255,255]
            exp5_seg[exp5_seg.sum(2)==0] = [255,255,255]
            exp6_seg[exp6_seg.sum(2)==0] = [255,255,255]
            gt_seg[gt_seg.sum(2)==0] = [255,255,255]

            # plot
            TITLE_SIZE = 30
            fig, ax = plt.subplots(1,8, figsize=(140,20))
            fig.tight_layout()
            ax[0].imshow(input_im)
            ax[0].axis('off')
            ax[1].imshow(exp1_seg)
            ax[1].axis('off')
            ax[2].imshow(exp2_seg)
            ax[2].axis('off')
            ax[3].imshow(exp3_seg)
            ax[3].axis('off')
            ax[4].imshow(exp4_seg)
            ax[4].axis('off')
            ax[5].imshow(exp5_seg)
            ax[5].axis('off')
            ax[6].imshow(exp6_seg)
            ax[6].axis('off')
            ax[7].imshow(gt_seg)
            ax[7].axis('off')
            plt.savefig(f"{OUT_DIR}/{filename_base}_comparison_all.png")
            plt.close()

