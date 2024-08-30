import streamlit as st
import huggingface_hub
import os
import time
import io
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import cv2
import requests
import json
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
from tqdm import tqdm
import torch
import natsort
from matplotlib import pyplot as plt

from transformers import pipeline

from backend.image_generation import DMD2_T2IAdapt
from backend.SAM import SAM, SAM_custom

# -------------- LOADING FUNCTIONS ---------------------

def load_dmd2_pipeline():
    pipe = DMD2_T2IAdapt()
    return pipe

def load_custom_SAM_pipeline():
    ckpt_root = "backend\\ckpts\\segmentation\\"
    labels_definiton_file_path = "D:\\UI\\Streamlit\\AnimatedDrawingsPlusPlusTool\\backend\\repos\\segment_anything\\label_definitions\\sam_drawings_full.json"
    original_ckpt_path = os.path.join(ckpt_root, "sam_vit_b_01ec64.pth")
    finetuned_ckpt_path = os.path.join(ckpt_root, "model_best.pth")
    pipe = SAM_custom(original_ckpt_path=original_ckpt_path, custom_ckpt_path=finetuned_ckpt_path, num_classes=27, label_definitions_path=labels_definiton_file_path)
    return pipe


def load_SAM_pipeline():
    pipe = SAM()
    return pipe

# -------------- LOADING FUNCTIONS END -----------------


# ------------  HELPER FUNCTIONS ------------
def name_to_id(name):
    mapping = {"All":0, "Face":6, "Hair":20}
    return mapping[name]

def filter_class(labels_np, class_id):
    if class_id>0:
        binmask = labels_np==class_id
        labels_np[binmask] = class_id
        labels_np[~binmask] = 0
    return labels_np


def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(torch.from_numpy(mask1), torch.from_numpy(mask2)).to(torch.int).sum()
    if union == 0:
        return 0.0
    return intersection / union

# ---------- HELPER FUNCTIONS END ----------

# function for segmentation using SAM
def segment_drawings(path_to_image, pipe, pipe_original_sam, pipe_auto_sam, RANSAC_STEPS=3):
    
    # class_name = st.sidebar.selectbox("Class", ["All", "Face", "Hair"])
    class_id = name_to_id("Face")
    
    bg_image = path_to_image
    sampling = [500, 200, 50, 10, 5]
    darken_background = True
    use_gt_bbox = False
    use_gt_mask = False
    use_gt_info = False
    REFINED_PTS = []
    
    if os.path.exists(bg_image):
        # Open image and create a canvas component
        input_image = Image.open(bg_image).convert("RGB").resize((1024,1024))            
        input_image = input_image.resize((1024,1024))
        w_inp,h_inp = input_image.size


        # Do something interesting with the image data and paths
        bbox_info = []
        if len(bbox_info)==0:
            bbox_info = [0,0,w_inp-1,h_inp-1]
        else:
            # latest bbox is the one we want
            bbox_info = bbox_info[-1]
            bbox_from_canvas = True

        x,y,w_box,h_box = bbox_info
        bbox_coords = np.array([x, y, x + w_box, y + h_box])
        # if bbox_from_canvas:
        #     bbox_coords = bbox_coords*(w_inp/w_canv) # scale the bbox taken from canvas to match the input image size
        # print("BBOX:", bbox_coords)
        
        input_image_np = np.array(input_image)
        seg_out = None
        if use_gt_bbox:
            gt_bbox = [[np.array(bbox_coords).astype('float').tolist()]]
            bbox_mask = pipe_original_sam.segment_with_bbox(input_image=input_image_np, input_bbox=gt_bbox)[0]
            bbox_mask_np = bbox_mask.squeeze(0).numpy()
            bbox_mask_np = np.transpose(bbox_mask_np, (1, 2, 0))[:,:,0]
            seg_out = bbox_mask_np.astype(np.uint8)
            seg_out[bbox_mask_np] = 26
        elif use_gt_mask:
            if gt_mask is not None:
                gt_labels = Image.open(gt_mask).convert("RGB").resize((w_inp,h_inp))
                gt_labels_np = np.array(gt_labels)[:,:,0]
                seg_out = filter_class(gt_labels_np, class_id)
                
        else:
            labels, sem_mask = pipe.segment_with_bbox(input_image=input_image_np, input_bbox=bbox_coords)
            labels_np = np.array(labels)
            seg_out = np.copy(labels_np)
            seg_vanilla_all = np.copy(labels_np)
            seg_automask = np.copy(labels_np)
            seg_out = filter_class(seg_out, class_id)
            seg_vanilla_class = np.copy(seg_out)

            # automatic masks
            random_colors = np.random.randint(0, 255, size=(27, 3), dtype="uint8")
            all_automasks = np.zeros_like(input_image)
            outputs = pipe_auto_sam(input_image, points_per_batch=64)
            max_overlap = 0
            correct_mask_idx = 0
            for idx,mask in enumerate(outputs['masks']):
                all_automasks[mask>0] = random_colors[idx]
                mask[mask==1] = class_id
                mask = mask.astype(np.uint8)
                overlap = iou(mask, seg_out)
                if overlap>max_overlap:
                    max_overlap = overlap
                    correct_mask_idx = idx
            
            correct_mask = outputs['masks'][correct_mask_idx]
            seg_automask[correct_mask] = class_id
            seg_automask[~correct_mask] = 0

            # refinement
            refine_pts = np.argwhere(labels_np==class_id)
            refine_pts = np.flip(refine_pts, axis=1)
            num_of_refine_pts = sampling[0]
            if int(num_of_refine_pts)>0 and len(refine_pts)>0 and class_id>0:
                for step in tqdm(range(RANSAC_STEPS)):
                    np.random.shuffle(refine_pts)
                    num_of_refine_pts = sampling[step]
                    refine_pts = refine_pts[:int(num_of_refine_pts)]
                    # num_of_refine_pts /= 2
                    mask_vis = np.copy(labels_np)
                    REFINED_PTS.append((mask_vis, refine_pts))

            refine_pts = refine_pts.tolist()
            refined_mask_tensor = pipe_original_sam.segment_with_points(input_image, [refine_pts])[0]
            refined_mask = refined_mask_tensor.squeeze(0).numpy()
            refined_mask = np.transpose(refined_mask, (1, 2, 0))[:,:,0]

            seg_out = labels_np
            seg_out[refined_mask] = class_id
            seg_out[~refined_mask] = 0

        if seg_out is not None:
            composite_mask = seg_out>0
            composite_factor = 0.8
            input_image_np = np.array(input_image)
            seg_out = pipe.labels_to_colors(seg_out).astype(np.uint8)
            seg_vanilla_class = pipe.labels_to_colors(seg_vanilla_class).astype(np.uint8)
            seg_vanilla_all = pipe.labels_to_colors(seg_vanilla_all).astype(np.uint8)
            seg_automask = pipe.labels_to_colors(seg_automask).astype(np.uint8)
            if darken_background:
                input_image_comp_class = input_image_np*(1-composite_factor) + seg_out*composite_factor
                input_image_comp_vanilla_class = input_image_np*(1-composite_factor) + seg_vanilla_class*composite_factor
                input_image_comp_vanilla_all = input_image_np*(1-composite_factor) + seg_vanilla_all*composite_factor
                input_image_comp_automask = input_image_np*(1-composite_factor) + seg_automask*composite_factor
                input_image_comp_all_automasks = input_image_np*(1-composite_factor) + all_automasks*composite_factor

                for midx, data in enumerate(REFINED_PTS):
                    mask, refine_pts = data
                    mask = pipe.labels_to_colors(mask).astype(np.uint8)
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
                    input_image_comp_refined = input_image_np*(1-composite_factor) + mask*composite_factor
                    
                    refine_pts_vis = np.copy(input_image_comp_refined)
                    for _,pt in enumerate(refine_pts):
                        centerOfCircle = (int(pt[0]), int(pt[1]))
                        refine_pts_vis = cv2.circle(refine_pts_vis, centerOfCircle, 0, (255,255,255), 7)
                    cv2.imwrite(f'seg_sampled_{midx}.png',refine_pts_vis)



            input_image_comp_class = input_image_comp_class.astype(np.uint8)
            input_image_comp_vanilla_class = input_image_comp_vanilla_class.astype(np.uint8)
            input_image_comp_vanilla_all = input_image_comp_vanilla_all.astype(np.uint8)
            input_image_comp_automask = input_image_comp_automask.astype(np.uint8)
            input_image_comp_all_automasks = input_image_comp_all_automasks.astype(np.uint8)

            input_image_comp_vanilla_class = cv2.cvtColor(input_image_comp_vanilla_class, cv2.COLOR_BGR2RGB)
            input_image_comp_vanilla_all = cv2.cvtColor(input_image_comp_vanilla_all, cv2.COLOR_BGR2RGB)
            input_image_comp_automask = cv2.cvtColor(input_image_comp_automask, cv2.COLOR_BGR2RGB)
            input_image_comp_all_automasks = cv2.cvtColor(input_image_comp_all_automasks, cv2.COLOR_BGR2RGB)

            cv2.imwrite('seg_vanilla.png',input_image_comp_vanilla_class)
            cv2.imwrite('seg_all.png',input_image_comp_vanilla_all)
            cv2.imwrite('seg_automask.png',input_image_comp_automask)
            cv2.imwrite('seg_all_automasks.png',input_image_comp_all_automasks)
            result = Image.fromarray(input_image_comp_class).convert("RGB")
            result.save('seg_final.png')

# ------------------------------------------------------

if __name__ == "__main__":
    
    pipe = load_custom_SAM_pipeline()
    pipe_original_sam = load_SAM_pipeline()
    pipe_auto_sam = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)


    DRAWINGS_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\drawings\\"
    LABELS_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\labels_resized\\"
    SAVE_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\visualizations_AUTOMASK\\"
    os.makedirs(SAVE_DIR, exist_ok=True)

    gt_labels = natsort.natsorted(os.listdir(LABELS_DIR))

    RANSAC_STEPS = 5

    for label in tqdm(gt_labels):

        try:
            path_to_label = os.path.join(LABELS_DIR, label)
            path_to_image = os.path.join(DRAWINGS_DIR, label.split('_')[0]+'.png')

            segment_drawings(path_to_image, pipe, pipe_original_sam, pipe_auto_sam, RANSAC_STEPS)

            input_image = Image.open(path_to_image).convert("RGB").resize((1024,1024))
            gt = Image.open(path_to_label).convert("RGB").resize((1024,1024))
            seg_all = Image.open('seg_all.png').convert("RGB").resize((1024,1024))
            seg_final = Image.open('seg_final.png').convert("RGB").resize((1024,1024))
            seg_vanilla = Image.open('seg_vanilla.png').convert("RGB").resize((1024,1024))
            seg_automask = Image.open('seg_automask.png').convert("RGB").resize((1024,1024))
            seg_all_automask = Image.open('seg_all_automasks.png').convert("RGB").resize((1024,1024))
            SAMPLED = []
            for idx in range(RANSAC_STEPS):
                seg_sampled = Image.open(f'seg_sampled_{idx}.png').convert("RGB").resize((1024,1024))
                SAMPLED.append(seg_sampled)

            # TITLE_SIZE = 40
            # fig, ax = plt.subplots(2,RANSAC_STEPS, figsize=(50,20))
            # plt.subplots_adjust(wspace=0.1, hspace=0.1)
            # ax[0,0].imshow(input_image)
            # ax[0,0].set_title("Input Image", fontsize=TITLE_SIZE)
            # ax[0,0].axis('off')
            # ax[0,1].imshow(seg_all)
            # ax[0,1].set_title("Predicted (All)", fontsize=TITLE_SIZE)
            # ax[0,1].axis('off')
            # ax[0,2].imshow(seg_vanilla)
            # ax[0,2].set_title("Predicted (Face)", fontsize=TITLE_SIZE)
            # ax[0,2].axis('off')
            # ax[0,3].imshow(seg_automask)
            # ax[0,3].set_title("Refined via SAM", fontsize=TITLE_SIZE)
            # ax[0,3].axis('off')
            # ax[0,4].imshow(gt)
            # ax[0,4].set_title("Ground Truth", fontsize=TITLE_SIZE)
            # ax[0,4].axis('off')
            # for idx in range(RANSAC_STEPS):
            #     ax[1,idx].imshow(SAMPLED[idx])
            #     ax[1,idx].set_title(f"RANSAC Step #{idx+1}", fontsize=TITLE_SIZE)
            #     ax[1,idx].axis('off')

            TITLE_SIZE = 20
            fig, ax = plt.subplots(1,4, figsize=(40,10))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            ax[0].imshow(input_image)
            ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
            ax[0].axis('off')
            ax[1].imshow(seg_vanilla, cmap='gray')
            ax[1].set_title("Prediction via Finetuned SAM", fontsize=TITLE_SIZE)
            ax[1].axis('off')
            ax[2].imshow(seg_all_automask)
            ax[2].set_title("All Masks", fontsize=TITLE_SIZE)
            ax[2].axis('off')
            ax[3].imshow(seg_automask)
            ax[3].set_title("Mask with highest IoU", fontsize=TITLE_SIZE)
            ax[3].axis('off')

            plt.savefig(f"{SAVE_DIR}\\{label.split('_')[0]}.png", bbox_inches='tight')

        except:
            print(f"Failed to process {label}")
            continue
