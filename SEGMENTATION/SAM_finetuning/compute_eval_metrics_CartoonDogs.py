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

from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationCoarse, SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationAll
from utils.augment import RandomAug

join = os.path.join

def dfpnet_im2label(im):
    r = im[:,:,0]
    head = r==177
    torso = r==68
    larm = r==21
    rarm = r==63
    lleg = r==171
    rleg = r==12
    tail = r==245
    labels = np.zeros_like(r)
    labels[head] = 1
    labels[torso] = 2
    labels[larm] = 3
    labels[rarm] = 4
    labels[lleg] = 5
    labels[rleg] = 6
    labels[tail] = 7
    return labels

if __name__ == '__main__':
    torch.manual_seed(999)
    np.random.seed(999)
    torch.multiprocessing.set_start_method('spawn')

    NUM_CLASSES = 8
    plot = False
    compute_metrics = True
    DFPNet = False

    if DFPNet:
        dfpnet_pred_dir = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/DFPNet-Drawings/Cartoon_sketches/Dog/save_preds_495/'

    # define metrics
    MeanIoU = segmentation.MeanIoU(num_classes=NUM_CLASSES, include_background=True, per_class=True, input_format='index')
    GeneralizedDiceScore = segmentation.GeneralizedDiceScore(num_classes=NUM_CLASSES, include_background=True, per_class=True, input_format='index')
    MulticlassAccuracy = classification.MulticlassAccuracy(num_classes=NUM_CLASSES, average=None)
    ConfusionMatrix = classification.MulticlassConfusionMatrix(num_classes=NUM_CLASSES)

    # set paths
    data_root = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/DFPNet-Drawings/Cartoon_sketches/Dog/'
    image_dir_name = 'val_images'
    label_id_dir_name = 'val_segmentations'  

    labels_definition_file_path = 'label_definition.json'
    ckpt_dir = './checkpoints'
    sam_original_ckpt_path = join(ckpt_dir,'sam_original/sam_vit_b_01ec64.pth')
    SAM_EVAL_ROOT = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/EVALUATION/"

    semantics = SemanticSegmentationNoFace(num_classes=18, labels_definition_path=labels_definition_file_path)

    exp_out_dir = join(SAM_EVAL_ROOT,"CartoonDogs_pretrainedSAM14k_FastLR_wAug/best_eval/")

    ref_dir = join(data_root, image_dir_name)
    files = sorted(os.listdir(ref_dir))

    classwise_ious = torch.zeros((NUM_CLASSES,))
    classwise_gds = torch.zeros((NUM_CLASSES,))
    classwise_acc = torch.zeros((NUM_CLASSES,))
    confusion_matrix = torch.zeros((NUM_CLASSES,NUM_CLASSES))
    pixel_acc = 0
    COUNT = 0
    for filename in tqdm(files):
        if filename.endswith('.jpg'):
            filename_base = filename[:-4]
            input_im = cv2.imread(join(data_root, image_dir_name, f'{filename_base}.jpg'))
            exp_seg = cv2.imread(join(exp_out_dir, f'{filename_base}/pred.png'))
            if DFPNet:
                exp_seg = cv2.imread(join(dfpnet_pred_dir,  f'{filename_base}.png'))

            gt_seg = cv2.imread(join(exp_out_dir, f'{filename_base}/gt.png'))
            input_im =  cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
            exp_seg =  cv2.cvtColor(exp_seg, cv2.COLOR_BGR2RGB)
            gt_seg =  cv2.cvtColor(gt_seg, cv2.COLOR_BGR2RGB)
            input_im = cv2.resize(input_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
            exp_seg = cv2.resize(exp_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt_seg = cv2.resize(gt_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)

            exp_labels = semantics.colors_to_labels(exp_seg)
            gt_labels = semantics.colors_to_labels(gt_seg)

            exp_t = torch.Tensor(exp_labels).long().unsqueeze(0)
            gt_t = torch.Tensor(gt_labels).long().unsqueeze(0)
            
            gt_t = torch.Tensor(semantics.colors_to_labels(gt_seg)).long().unsqueeze(0)
            exp_t = torch.Tensor(semantics.colors_to_labels(exp_seg)).long().unsqueeze(0)
            if DFPNet:
                exp_t =  torch.Tensor(dfpnet_im2label(exp_seg)).long().unsqueeze(0)
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

            # Per-class Accuracy
            not_present = MulticlassAccuracy(gt_t, gt_t) == 0
            acc = MulticlassAccuracy(exp_t, gt_t)
            acc[not_present] = 1.0
            classwise_acc += acc

            confusion_matrix += ConfusionMatrix(exp_t.flatten(),gt_t.flatten())

            COUNT += 1

                
    avg_classwise_ious = classwise_ious / COUNT
    avg_classwise_gds = classwise_gds / COUNT
    avg_classwise_acc = classwise_acc / COUNT

    final_mean_iou = avg_classwise_ious.sum()/(NUM_CLASSES)
    final_mean_gds = avg_classwise_gds.sum()/(NUM_CLASSES)
    final_mean_acc = avg_classwise_acc.sum()/(NUM_CLASSES)

    confusion_matrix = confusion_matrix.numpy()
    pos = confusion_matrix.sum(1)
    tp = np.diag(confusion_matrix)
    pixel_accuracy = (tp.sum() / pos.sum()) * 100

    log = f'''
    NumValSamples={COUNT}
    MeanIoU={final_mean_iou}
    MeanGenDiceScore={final_mean_gds}
    MeanAcc={final_mean_acc}
    PixelAcc={pixel_accuracy}
    '''

    print(log)

    print("------------------------")

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)