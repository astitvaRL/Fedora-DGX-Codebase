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

from tqdm import tqdm
from utils.deform import tps_warp_box_mouth, tps_warp_preset_eyes, tps_warp_preset_mouth
from utils.preset_config import PresetConfig

import sys
sys.path.append('./utils/external/SAM/')
from utils.SemanticSegmentation import SemanticSegmentationAll
from utils.SegmentDrawings import SAM_face

join = os.path.join


# set paths
data_root = "/mnt/users_scratch/astitva/DATA/"
labels_definition_file_path = "./label_definition.json"
image_dir_name = "MANIFOLD/animated_drawings_images_prior_april22/cropped_image"
label_id_dir_name = "AD_SegMaps/labels_16k"

output_root = join(data_root, "DRAWINGS_ALL_CLASSES_16k/")
output_img_dir = join(output_root, 'image')
output_label_id_dir = join(output_root, 'labels')
output_segmap_dir = join(output_root, 'vis')
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_id_dir, exist_ok=True)
os.makedirs(output_segmap_dir, exist_ok=True)

# load semantic definitions
semantics = SemanticSegmentationAll(labels_definition_file_path)

# load labels
start = 0
end = -1
labels = sorted(os.listdir(join(data_root, label_id_dir_name)))[start:end]

# iterate over images
for label_name in tqdm(labels):

    # load images
    img_name = f"{label_name.split('_')[0]}.png"
    img_full = cv2.imread(join(data_root, image_dir_name, img_name))
    img_full = cv2.resize(img_full, (1024, 1024))
    label_full = cv2.imread(join(data_root, label_id_dir_name, label_name))
    label_full = cv2.cvtColor(label_full, cv2.COLOR_BGR2RGB)
    label_full = cv2.resize(label_full, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label_id = semantics.colors_to_labels(label_full)

    # # extract face region
    # face_region = label_id == 5 #| (label_id == 3) | (label_id == 4) | (label_id == 5) | (label_id == 6) | (label_id == 12) | (label_id == 13) | (label_id == 17) | (label_id == 18) | (label_id == 22) | (label_id == 23)
    # if face_region.max()==False:
    #     continue
    # # face cropping window
    # face_padding = 5
    # face_Xs, face_Ys = np.where(face_region)
    # face_x_min, face_x_max = np.min(face_Xs) - face_padding, np.max(face_Xs) + face_padding
    # face_y_min, face_y_max = np.min(face_Ys) - face_padding, np.max(face_Ys) + face_padding
    # if face_x_min < 0:
    #     face_x_min = 0
    # if face_y_min < 0:
    #     face_y_min = 0
    # if face_x_max > 1024:
    #     face_x_max = 1024
    # if face_y_max > 1024:
    #     face_y_max = 1024

    img_face = img_full
    label_face = label_full
    label_id_face = label_id

    img_face = cv2.resize(img_face, (1024,1024))
    label_face = cv2.resize(label_face, (1024,1024), interpolation=cv2.INTER_NEAREST)
    label_id_face = cv2.resize(label_id_face, (1024,1024), interpolation=cv2.INTER_NEAREST)

    label_id_im = np.zeros_like(img_face)
    label_id_im[:,:,0] = label_id_face
    label_id_im[:,:,1] = label_id_face
    label_id_im[:,:,2] = label_id_face

    cv2.imwrite(join(output_img_dir, label_name), img_face)
    cv2.imwrite(join(output_label_id_dir, label_name),label_id_im)
    label_face = cv2.cvtColor(label_face, cv2.COLOR_BGR2RGB)
    cv2.imwrite(join(output_segmap_dir, label_name),label_face)

