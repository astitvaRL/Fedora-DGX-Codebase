import os

# setup cache path for huggingface
os.environ["CACHE_DIR"] = "/mnt/users_scratch/astitva/CACHE/"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HOME"] = os.environ["CACHE_DIR"]
os.environ["HF_DATASETS_CACHE"] = os.environ["CACHE_DIR"]
os.environ["TRANSFORMERS_CACHE"] = os.environ["CACHE_DIR"]

import cv2
import numpy as np
import segmentation_refinement as segref
from matplotlib import pyplot as plt
from PIL import Image
import time

from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from utils.deform import tps_warp_box_mouth, tps_warp_preset_eyes, tps_warp_preset_mouth
from utils.preset_config import PresetConfig

import sys
sys.path.append('./utils/external/SAM/')
from utils.SemanticSegmentation import SemanticSegmentationAll, SemanticSegmentationFace
from utils.SegmentDrawings import SAM_face

join = os.path.join

def bgr_conversion(img):
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

labels_definition_file_path = "./label_definition.json"
semantics_face = SemanticSegmentationFace(labels_definition_file_path)
semantics = SemanticSegmentationAll(labels_definition_file_path)

# set paths
data_root = "/mnt/users_scratch/astitva/DATA/"
labels_definition_file_path = "./label_definition.json"
# image_dir_name = "LIP_drawings_16k/images/val_images/"
image_dir_name = "MANIFOLD/animated_drawings_images_prior_april22/cropped_image"
# image_dir_name = "IN_THE_WILD_faces"
preset_dir = "./presets"
preset_class = "eyes"  # DON'T FORGET TO CHANGE CANONICAL COORDINATES & CANNY THRESHOLDS ACCORDINGLY IN THE SHAPE & STYLIZATION SCRIPTS
out_parent_dir = "/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/PresetGeneration/OUTPUT/DRAWINGS_DATASET_GT_SEG"
os.makedirs(out_parent_dir, exist_ok=True)

# prest configuration
preset_config = PresetConfig(preset_class)
# preset_prompts = preset_config.config["prompts"]
shape_ids = preset_config.config["shape_ids"]
output_root = f"{out_parent_dir}/{preset_class}"
os.makedirs(output_root, exist_ok=True)


# load presets
preset_cache_dict = {}
for shape_idx in range(len(shape_ids)):
    shape_id = shape_ids[shape_idx]
    preset_image = cv2.imread(join(preset_dir, preset_class, f"{shape_id}.png"), -1)
    preset_image = cv2.resize(
        preset_image, (1024, 1024), interpolation=cv2.INTER_NEAREST
    )
    alpha_mask = preset_image[:,:,3]==255
    canvas = (preset_image[:,:,0]==0) & (preset_image[:,:,1]>250) & (preset_image[:,:,2]==0)
    # mouth = (~canvas) & alpha_mask
    eye = (~canvas) & alpha_mask
    pupil = preset_image[:,:,0]>200
    # tongue = preset_image[:,:,0]>250
    # teeth = (preset_image[:,:,0]>0) & (preset_image[:,:,0]<250)
    # prepare conditioning image
    label_face_id_tmp = np.zeros((1024,1024)).astype('uint8')
    # label_face_id_tmp[mouth] = 2
    # label_face_id_tmp[teeth] = 7
    # label_face_id_tmp[tongue] = 10
    label_face_id_tmp[eye] = 9
    label_face_id_tmp[pupil] = 3
    cond_image = semantics_face.labels_to_colors(label_face_id_tmp)
    final_preset = np.zeros((1024,1024,4)).astype('uint8')
    alpha = label_face_id_tmp>0
    final_preset[:,:,3] = alpha.astype('uint8')*255
    final_preset[:,:,:3] = cond_image
    cv2.imwrite(f'./preset_labels_eyes/{shape_id}.png',cv2.cvtColor(final_preset, cv2.COLOR_RGBA2BGRA))
    # if tongue_mask.max() and not shape_idx==14:
    #     preset_image[tongue_mask] = [0,0,0,255]
    #     tongue_mask = cv2.erode(tongue_mask.astype('uint8'), np.ones((9,9)), iterations=3)
    #     tongue_mask = tongue_mask>0
    #     preset_image[tongue_mask] = [255,255,255,255]
    #     preset_image[teeth_mask] = [0,0,0,255]
    # preset_cache_dict[shape_id] = preset_image

# for shape_idx in range(len(shape_ids)):
#     shape_id = shape_ids[shape_idx]
#     preset_image = preset_cache_dict[shape_id]
#     # mouth = preset_image.sum(2)>0
#     teeth = (preset_image[:,:,0]>0) & (preset_image[:,:,0]<250)
#     tongue = preset_image[:,:,0]>250
#     # prepare conditioning image
#     label_face_id_tmp = np.zeros((1024,1024)).astype('uint8')
#     # label_face_id_tmp[mouth] = 2
#     label_face_id_tmp[teeth] = 7
#     label_face_id_tmp[tongue] = 10
#     cond_image = semantics_face.labels_to_colors(label_face_id_tmp)
#     final_preset = np.zeros((1024,1024,4)).astype('uint8')
#     alpha = label_face_id_tmp>0
#     final_preset[:,:,3] = alpha.astype('uint8')*255
#     final_preset[:,:,:3] = cond_image
#     breakpoint()

