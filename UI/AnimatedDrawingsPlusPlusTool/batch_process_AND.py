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
import natsort
from matplotlib import pyplot as plt

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
# ---------- HELPER FUNCTIONS END ----------

# function for segmentation using SAM
def segment_drawings(path_to_image, pipe, pipe_original_sam):
    RANSAC_STEPS = 3 #15
    
    
    # class_name = st.sidebar.selectbox("Class", ["All", "Face", "Hair"])
    class_id = name_to_id("Face")
    
    bg_image = path_to_image
    num_of_refine_pts = 5
    darken_background = True
    use_gt_bbox = False
    use_gt_mask = False
    use_gt_info = False
    REFINED_MASKS = []
    
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
        print("BBOX:", bbox_coords)
        
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
            seg_out = filter_class(seg_out, class_id)
            seg_vanilla_class = np.copy(seg_out)
            # refinement
            class_pts = np.argwhere(labels_np==class_id)
            if int(num_of_refine_pts)>0 and len(class_pts)>0 and class_id>0:
                refined_mask_ransac = np.ones_like(labels_np==class_id).astype(bool)
                for step in tqdm(range(RANSAC_STEPS)):
                    refine_pts = np.argwhere(labels_np==class_id)
                    refine_pts = np.flip(refine_pts, axis=1)
                    np.random.shuffle(refine_pts)
                    refine_pts = refine_pts[:int(num_of_refine_pts)]
                    refine_pts = refine_pts.tolist()
                    refined_mask_tensor = pipe_original_sam.segment_with_points(input_image, [refine_pts])[0]
                    refined_mask = refined_mask_tensor.squeeze(0).numpy()
                    refined_mask = np.transpose(refined_mask, (1, 2, 0))[:,:,0]
                    refined_mask_ransac  = refined_mask_ransac & refined_mask

                    refined_mask_vis = np.copy(labels_np)
                    refined_mask_vis[refined_mask] = class_id
                    refined_mask_vis[~refined_mask] = 0
                    # cv2.imwrite(f'vis_{step}.png',refine_pts_vis)
                    REFINED_MASKS.append((refined_mask_vis, refine_pts))


                seg_out = labels_np
                seg_out[refined_mask_ransac] = class_id
                seg_out[~refined_mask_ransac] = 0

        if seg_out is not None:
            composite_mask = seg_out>0
            composite_factor = 0.8
            input_image_np = np.array(input_image)
            seg_out = pipe.labels_to_colors(seg_out).astype(np.uint8)
            seg_vanilla_class = pipe.labels_to_colors(seg_vanilla_class).astype(np.uint8)
            seg_vanilla_all = pipe.labels_to_colors(seg_vanilla_all).astype(np.uint8)
            if darken_background:
                input_image_comp_class = input_image_np*(1-composite_factor) + seg_out*composite_factor
                input_image_comp_vanilla_class = input_image_np*(1-composite_factor) + seg_vanilla_class*composite_factor
                input_image_comp_vanilla_all = input_image_np*(1-composite_factor) + seg_vanilla_all*composite_factor

                for midx, data in enumerate(REFINED_MASKS):
                    refined_mask, refine_pts = data
                    refined_mask = pipe.labels_to_colors(refined_mask).astype(np.uint8)
                    refined_mask = cv2.cvtColor(refined_mask, cv2.COLOR_RGB2BGR)
                    input_image_comp_refined = input_image_np*(1-composite_factor) + refined_mask*composite_factor
                    
                    refine_pts_vis = np.copy(input_image_comp_refined)
                    for _,pt in enumerate(refine_pts):
                        centerOfCircle = (int(pt[0]), int(pt[1]))
                        refine_pts_vis = cv2.circle(refine_pts_vis, centerOfCircle, 0, (255,255,255), 7)
                    cv2.imwrite(f'seg_refined_{midx}.png',refine_pts_vis)



            input_image_comp_class = input_image_comp_class.astype(np.uint8)
            input_image_comp_vanilla_class = input_image_comp_vanilla_class.astype(np.uint8)
            input_image_comp_vanilla_all = input_image_comp_vanilla_all.astype(np.uint8)
            # result = Image.fromarray(input_image_np).convert("RGB").resize((w_canv,w_canv))
            input_image_comp_vanilla_class = cv2.cvtColor(input_image_comp_vanilla_class, cv2.COLOR_BGR2RGB)
            input_image_comp_vanilla_all = cv2.cvtColor(input_image_comp_vanilla_all, cv2.COLOR_BGR2RGB)
            cv2.imwrite('seg_vanilla.png',input_image_comp_vanilla_class)
            cv2.imwrite('seg_all.png',input_image_comp_vanilla_all)
            result = Image.fromarray(input_image_comp_class).convert("RGB")
            result.save('seg_final.png')

# ------------------------------------------------------

if __name__ == "__main__":
    
    pipe = load_custom_SAM_pipeline()
    pipe_original_sam = load_SAM_pipeline()

    DRAWINGS_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\drawings\\"
    LABELS_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\labels_resized\\"
    SAVE_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\visualizations\\"
    os.makedirs(SAVE_DIR, exist_ok=True)

    gt_labels = natsort.natsorted(os.listdir(LABELS_DIR))[32+124+40:]

    for label in tqdm(gt_labels):
        try:
            path_to_label = os.path.join(LABELS_DIR, label)
            path_to_image = os.path.join(DRAWINGS_DIR, label.split('_')[0]+'.png')

            segment_drawings(path_to_image, pipe, pipe_original_sam)

            input_image = Image.open(path_to_image).convert("RGB").resize((1024,1024))
            gt = Image.open(path_to_label).convert("RGB").resize((1024,1024))
            seg_all = Image.open('seg_all.png').convert("RGB").resize((1024,1024))
            seg_final = Image.open('seg_final.png').convert("RGB").resize((1024,1024))
            seg_vanilla = Image.open('seg_vanilla.png').convert("RGB").resize((1024,1024))
            seg_refined_0 = Image.open('seg_refined_0.png').convert("RGB").resize((1024,1024))
            seg_refined_1 = Image.open('seg_refined_1.png').convert("RGB").resize((1024,1024))
            seg_refined_2 = Image.open('seg_refined_2.png').convert("RGB").resize((1024,1024))

            
            fig, ax = plt.subplots(3,3, figsize=(60,60))
            TITLE_SIZE = 60
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            ax[0,0].imshow(input_image)
            ax[0,0].set_title("Input Image", fontsize=TITLE_SIZE)
            ax[0,0].axis('off')
            ax[0,1].imshow(seg_all)
            ax[0,1].set_title("Finetuned Prediction (All)", fontsize=TITLE_SIZE)
            ax[0,1].axis('off')
            ax[0,2].imshow(seg_vanilla)
            ax[0,2].set_title("Finetuned Prediction (Face)", fontsize=TITLE_SIZE)
            ax[0,2].axis('off')
            ax[1,0].imshow(seg_refined_0)
            ax[1,0].set_title("Sampling-based Refinement #1", fontsize=TITLE_SIZE)
            ax[1,0].axis('off')
            ax[1,1].imshow(seg_refined_1)
            ax[1,1].set_title("Sampling-based Refinement #2", fontsize=TITLE_SIZE)
            ax[1,1].axis('off')
            ax[1,2].imshow(seg_refined_2)
            ax[1,2].set_title("Sampling-based Refinement #3", fontsize=TITLE_SIZE)
            ax[1,2].axis('off')
            ax[2,0].imshow(seg_final)
            ax[2,0].set_title("Refined (AND Operation)", fontsize=TITLE_SIZE)
            ax[2,0].axis('off')
            ax[2,1].imshow(ImageChops.difference(seg_final, seg_vanilla))
            ax[2,1].set_title("Finetuned ~ Refined", fontsize=TITLE_SIZE)
            ax[2,1].axis('off')
            ax[2,2].imshow(gt)
            ax[2,2].set_title("Ground Truth", fontsize=TITLE_SIZE)
            ax[2,2].axis('off')

            plt.savefig(f"{SAVE_DIR}\\{label.split('_')[0]}.png", bbox_inches='tight')

        except:
            print(f"Failed to process {label}")
            continue
