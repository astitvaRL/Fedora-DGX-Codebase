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
    SAPIENS_EVAL_ROOT = "/mnt/users_scratch/astitva/DATA/LIP_drawings_16k/sapiens_val_prediction/sapiens_0.3b/"
    
    exp1_out_dir = join(SAPIENS_EVAL_ROOT)
    exp2_out_dir = join(SAM_EVAL_ROOT,"eval_16k_DecoderOnly_SINGLE_STAGE/best_eval/")
    exp3_out_dir = join(SAM_EVAL_ROOT,"eval_16k_E2E_SINGLE_STAGE/best_eval/")
    exp4_out_dir = join(SAM_EVAL_ROOT,"eval_16k_E2E_FULL/best_eval/")

    OUT_DIR = './COMPARISON/epoch_300/'
    os.makedirs(OUT_DIR, exist_ok=True)

    semantics = SemanticSegmentationAll(num_classes=NUM_CLASSES, labels_definition_path=labels_definition_file_path)

    files = sorted(os.listdir(exp1_out_dir))

    classwise_ious_1 = torch.zeros((NUM_CLASSES,))
    classwise_ious_2 = torch.zeros((NUM_CLASSES,))
    classwise_ious_3 = torch.zeros((NUM_CLASSES,))
    classwise_ious_4 = torch.zeros((NUM_CLASSES,))
    classwise_gds_1 = torch.zeros((NUM_CLASSES,))
    classwise_gds_2 = torch.zeros((NUM_CLASSES,))
    classwise_gds_3 = torch.zeros((NUM_CLASSES,))
    classwise_gds_4 = torch.zeros((NUM_CLASSES,))
    classwise_acc_1 = torch.zeros((NUM_CLASSES,))
    classwise_acc_2 = torch.zeros((NUM_CLASSES,))
    classwise_acc_3 = torch.zeros((NUM_CLASSES,))
    classwise_acc_4 = torch.zeros((NUM_CLASSES,))
    COUNT = 0
    for filename in tqdm(files):
        if filename.endswith('.png'):
            filename_base = filename.split('_')[0]
            input_im = cv2.imread(join(data_root, image_dir_name, f'{filename_base}.png'))
            exp1_seg = cv2.imread(join(exp1_out_dir, f'{filename}'))
            exp2_seg = cv2.imread(join(exp2_out_dir, f'{filename_base}/pred.png'))
            exp3_seg = cv2.imread(join(exp3_out_dir, f'{filename_base}/pred.png'))
            exp4_seg = cv2.imread(join(exp4_out_dir, f'{filename_base}/pred_all.png'))
            gt_seg = cv2.imread(join(exp2_out_dir, f'{filename_base}/gt.png'))
            input_im =  cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
            exp1_seg =  cv2.cvtColor(exp1_seg, cv2.COLOR_BGR2RGB)
            exp2_seg =  cv2.cvtColor(exp2_seg, cv2.COLOR_BGR2RGB)
            exp3_seg =  cv2.cvtColor(exp3_seg, cv2.COLOR_BGR2RGB)
            exp4_seg =  cv2.cvtColor(exp4_seg, cv2.COLOR_BGR2RGB)
            gt_seg =  cv2.cvtColor(gt_seg, cv2.COLOR_BGR2RGB)
            input_im = cv2.resize(input_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
            exp1_seg = cv2.resize(exp1_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp1_seg = exp1_seg[:,512:,:]
            exp1_seg = cv2.resize(exp1_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp2_seg = cv2.resize(exp2_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp3_seg = cv2.resize(exp3_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            exp4_seg = cv2.resize(exp4_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)
            gt_seg = cv2.resize(gt_seg, (1024,1024), interpolation=cv2.INTER_NEAREST)

            COUNT += 1

            if compute_metrics:
                gt_t = torch.Tensor(semantics.colors_to_labels(gt_seg)).long().unsqueeze(0)
                exp1_t = torch.Tensor(semantics.colors_to_labels(exp1_seg)).long().unsqueeze(0)
                exp2_t = torch.Tensor(semantics.colors_to_labels(exp2_seg)).long().unsqueeze(0)
                exp3_t = torch.Tensor(semantics.colors_to_labels(exp3_seg)).long().unsqueeze(0)
                exp4_t = torch.Tensor(semantics.colors_to_labels(exp4_seg)).long().unsqueeze(0)
                # Per-class IoU
                not_present = MeanIoU(gt_t, gt_t) == 0
                iou_1 = MeanIoU(exp1_t, gt_t)
                iou_2 = MeanIoU(exp2_t, gt_t)
                iou_3 = MeanIoU(exp3_t, gt_t)
                iou_4 = MeanIoU(exp4_t, gt_t)
                iou_1[not_present] = 1.0
                iou_2[not_present] = 1.0
                iou_3[not_present] = 1.0
                iou_4[not_present] = 1.0
                classwise_ious_1 += iou_1
                classwise_ious_2 += iou_2
                classwise_ious_3 += iou_3
                classwise_ious_4 += iou_4

                # Per-class Generalized Dice Score
                not_present = GeneralizedDiceScore(gt_t, gt_t) == 0
                gds_1 = GeneralizedDiceScore(exp1_t, gt_t)
                gds_2 = GeneralizedDiceScore(exp2_t, gt_t)
                gds_3 = GeneralizedDiceScore(exp3_t, gt_t)
                gds_4 = GeneralizedDiceScore(exp4_t, gt_t)
                gds_1[not_present] = 1.0
                gds_2[not_present] = 1.0
                gds_3[not_present] = 1.0
                gds_4[not_present] = 1.0
                classwise_gds_1 += gds_1
                classwise_gds_2 += gds_2
                classwise_gds_3 += gds_3
                classwise_gds_4 += gds_4

                # Per-class Hausdorff Distance
                not_present = MulticlassAccuracy(gt_t, gt_t) == 0
                acc_1 = MulticlassAccuracy(exp1_t, gt_t)
                acc_2 = MulticlassAccuracy(exp2_t, gt_t)
                acc_3 = MulticlassAccuracy(exp3_t, gt_t)
                acc_4 = MulticlassAccuracy(exp4_t, gt_t)
                acc_1[not_present] = 1.0
                acc_2[not_present] = 1.0
                acc_3[not_present] = 1.0
                acc_4[not_present] = 1.0
                classwise_acc_1 += acc_1
                classwise_acc_2 += acc_2
                classwise_acc_3 += acc_3
                classwise_acc_4 += acc_4

            if plot:
                # plot
                TITLE_SIZE = 30
                fig, ax = plt.subplots(1,6, figsize=(60,10))
                fig.tight_layout()
                ax[0].imshow(input_im)
                # ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                ax[0].axis('off')
                ax[1].imshow(exp1_seg)
                # ax[1].set_title("Sapiens (ALL)", fontsize=TITLE_SIZE)
                ax[1].axis('off')
                ax[2].imshow(exp2_seg)
                # ax[2].set_title("SAM DecoderOnly (ALL)", fontsize=TITLE_SIZE)
                ax[2].axis('off')
                ax[3].imshow(exp3_seg)
                # ax[3].set_title("SAM EncDec (ALL)", fontsize=TITLE_SIZE)
                ax[3].axis('off')
                ax[4].imshow(exp4_seg)
                # ax[4].set_title("SAM EncDec (Coarse-to-Fine)", fontsize=TITLE_SIZE)
                ax[4].axis('off')
                ax[5].imshow(gt_seg)
                # ax[5].set_title("Ground Truth", fontsize=TITLE_SIZE)
                ax[5].axis('off')
                plt.savefig(f"{OUT_DIR}/{filename_base}_plot.png")
                plt.close()

    avg_classwise_ious_1 = classwise_ious_1 / COUNT
    avg_classwise_ious_2 = classwise_ious_2 / COUNT
    avg_classwise_ious_3 = classwise_ious_3 / COUNT
    avg_classwise_ious_4 = classwise_ious_4 / COUNT

    avg_classwise_gds_1 = classwise_gds_1 / COUNT
    avg_classwise_gds_2 = classwise_gds_2 / COUNT
    avg_classwise_gds_3 = classwise_gds_3 / COUNT
    avg_classwise_gds_4 = classwise_gds_4 / COUNT

    avg_classwise_acc_1 = classwise_acc_1 / COUNT
    avg_classwise_acc_2 = classwise_acc_2 / COUNT
    avg_classwise_acc_3 = classwise_acc_3 / COUNT
    avg_classwise_acc_4 = classwise_acc_4 / COUNT

    final_mean_iou_1 = avg_classwise_ious_1.sum()/NUM_CLASSES
    final_mean_iou_2 = avg_classwise_ious_2.sum()/NUM_CLASSES
    final_mean_iou_3 = avg_classwise_ious_3.sum()/NUM_CLASSES
    final_mean_iou_4 = avg_classwise_ious_4.sum()/NUM_CLASSES

    final_mean_gds_1 = avg_classwise_gds_1.sum()/NUM_CLASSES
    final_mean_gds_2 = avg_classwise_gds_2.sum()/NUM_CLASSES
    final_mean_gds_3 = avg_classwise_gds_3.sum()/NUM_CLASSES
    final_mean_gds_4 = avg_classwise_gds_4.sum()/NUM_CLASSES

    final_mean_acc_1 = avg_classwise_acc_1.sum()/NUM_CLASSES
    final_mean_acc_2 = avg_classwise_acc_2.sum()/NUM_CLASSES
    final_mean_acc_3 = avg_classwise_acc_3.sum()/NUM_CLASSES
    final_mean_acc_4 = avg_classwise_acc_4.sum()/NUM_CLASSES

    log = f'''
    SAPIENS (ALL classes) : MeanIoU={final_mean_iou_1}; MeanGenDiceScore={final_mean_gds_1}; MeanAcc={final_mean_acc_1}
    SAM DecoderOnly (ALL Classes) : MeanIoU={final_mean_iou_2}; MeanGenDiceScore={final_mean_gds_2}; MeanAcc={final_mean_acc_2}
    SAM End-to-End (ALL Classes) : MeanIoU={final_mean_iou_3}; MeanDiceScore={final_mean_gds_3}; MeanAcc={final_mean_acc_3}
    SAM End-to-End (C2F) : MeanIoU={final_mean_iou_4}; MeanDiceScore={final_mean_gds_4}; MeanAcc={final_mean_acc_4}
    '''

    print(log)
    