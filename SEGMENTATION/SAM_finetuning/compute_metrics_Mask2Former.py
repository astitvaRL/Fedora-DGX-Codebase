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

    NUM_CLASSES = 6
    plot = False
    compute_metrics = True

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

    # semantics = SemanticSegmentationAll(num_classes=NUM_CLASSES, labels_definition_path=labels_definition_file_path)
    semantics = SemanticSegmentationCoarse(num_classes=NUM_CLASSES, labels_definition_path=labels_definition_file_path)

    exp_eval_results_dir = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/EVALUATION/4k_ANIMSEG_E2E_COARSE/500/500"

    ref_dir = join(data_root, label_id_dir_name)
    files = sorted(os.listdir(ref_dir))[-2000:]

    classwise_ious = torch.zeros((NUM_CLASSES,))
    classwise_gds = torch.zeros((NUM_CLASSES,))
    classwise_acc = torch.zeros((NUM_CLASSES,))
    COUNT = 0
    for filename in tqdm(files):
        if filename.endswith('.png'):
            filename_base = filename.split('_')[0]
            input_im = cv2.imread(join(data_root, image_dir_name, f'{filename_base}.png'))
            exp_seg = cv2.imread(join(exp_eval_results_dir, f'{filename_base}.png/pred.png'))
            gt_seg = cv2.imread(join(exp_eval_results_dir, f'{filename_base}.png/gt.png'))
            input_im =  cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
            exp_seg =  cv2.cvtColor(exp_seg, cv2.COLOR_BGR2RGB)
            gt_seg =  cv2.cvtColor(gt_seg, cv2.COLOR_BGR2RGB)
            input_im = cv2.resize(input_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
            exp_seg = cv2.resize(exp_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt_seg = cv2.resize(gt_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)

            try:
                gt_t = torch.Tensor(semantics.colors_to_labels(gt_seg)).long().unsqueeze(0)
                exp_t = torch.Tensor(semantics.colors_to_labels(exp_seg)).long().unsqueeze(0)
                # Per-class IoU
                not_present = MeanIoU(gt_t, gt_t) == 0
                iou = MeanIoU(exp_t, gt_t)
                iou[not_present] = 1.0
                classwise_ious += iou

                # Per-class Generalized Dice Score
                not_present = GeneralizedDiceScore(gt_t, gt_t) == 0
                gds = GeneralizedDiceScore(exp_t, gt_t)
                gds[not_present] = 1.0
                classwise_gds += gds

                # Per-class Hausdorff Distance
                not_present = MulticlassAccuracy(gt_t, gt_t) == 0
                acc = MulticlassAccuracy(exp_t, gt_t)
                acc[not_present] = 1.0
                classwise_acc += acc

                COUNT += 1
            except:
                pass
                

    avg_classwise_ious = classwise_ious / COUNT
    avg_classwise_gds = classwise_gds / COUNT
    avg_classwise_acc = classwise_acc / COUNT

    final_mean_iou = avg_classwise_ious.sum()/NUM_CLASSES
    final_mean_gds = avg_classwise_gds.sum()/NUM_CLASSES
    final_mean_acc = avg_classwise_acc.sum()/NUM_CLASSES

    log = f'''
    MeanIoU={final_mean_iou}
    MeanGenDiceScore={final_mean_gds}
    MeanAcc={final_mean_acc}
    '''

    print(log)
    