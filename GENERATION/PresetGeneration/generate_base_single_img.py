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
from color_transfer import color_transfer

from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from utils.deform import tps_warp_box_mouth, tps_warp_preset_eyes, tps_warp_preset_mouth
from utils.preset_config import PresetConfig

import sys
sys.path.append('./utils/external/SAM/')
from utils.SemanticSegmentation import SemanticSegmentationAll
from utils.SegmentDrawings import SAM_face
from utils.stylization import ControllableStylization

join = os.path.join


# set paths
data_root = "/mnt/users_scratch/astitva/DATA/"
labels_definition_file_path = "./label_definition.json"
image_dir_name = "MANIFOLD/animated_drawings_images_prior_april22/cropped_image"
label_id_dir_name = "AD_SegMaps/labels_7k_1024"
preset_dir = "./presets"
preset_class = "eyes"  # DON'T FORGET TO CHANGE CANONICAL COORDINATES & CANNY THRESHOLDS ACCORDINGLY IN THE SHAPE & STYLIZATION SCRIPTS

# prest configuration
preset_config = PresetConfig(preset_class)
preset_prompts = preset_config.config["prompts"]
shape_ids = preset_config.config["shape_ids"]
output_root = f"OUTPUT/output_BASE_SINGLE"


# load inpainting model
inpainting_model = SimpleLama()

# load semantic definitions
semantics = SemanticSegmentationAll(labels_definition_file_path)

#load SAM model
sam_face_model = SAM_face()

# load refiner
refiner = segref.Refiner(device="cuda:0")  # device can also be 'cpu'

# load labels
start = 0
end = -1

label_name = "TMP/image_seg.png"
img_name = "TMP/image.png"

# create output directory
output_dir = join(output_root, label_name.split("/")[0])
os.makedirs(output_dir, exist_ok=True)
print(output_dir)

img_full = cv2.imread(img_name)
wo, ho = img_full.shape[:2]
img_full = cv2.resize(img_full, (1024, 1024))
img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
label_full = cv2.imread(label_name)
label_full = cv2.resize(label_full, (1024, 1024), interpolation=cv2.INTER_NEAREST)
label_full = cv2.cvtColor(label_full, cv2.COLOR_BGR2RGB)
label_id = semantics.colors_to_labels(label_full)


# INPAINTING

kernel = np.ones((5, 5), np.uint8)

# inpaint all
inpainting_mask = (label_id == 3) | (label_id == 23) | (label_id == 17) | (label_id == 4) | (label_id == 22)
inpainting_mask = cv2.dilate(
    255 * inpainting_mask.astype("uint8"), kernel, iterations=3
)
img_base = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask)

# inpaint mouth
inpainting_mask = (label_id == 3) | (label_id == 23) | (label_id == 17)
inpainting_mask = cv2.dilate(
    255 * inpainting_mask.astype("uint8"), kernel, iterations=3
)
img_no_mouth = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask)

# inpaint eyes
inpainting_mask = (label_id == 4) | (label_id == 22)
inpainting_mask = cv2.dilate(
    255 * inpainting_mask.astype("uint8"), kernel, iterations=3
)
img_no_eyes = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask)

# extract face region
face_region = (label_id == 2) | (label_id == 3) | (label_id == 4) | (label_id == 5) | (label_id == 6) | (label_id == 12) | (label_id == 13) | (label_id == 17) | (label_id == 18) | (label_id == 22) | (label_id == 23)

# face cropping window
face_padding = 20
face_Xs, face_Ys = np.where(face_region)
face_x_min, face_x_max = np.min(face_Xs) - face_padding, np.max(face_Xs) + face_padding
face_y_min, face_y_max = np.min(face_Ys) - face_padding, np.max(face_Ys) + face_padding
if face_x_min < 0:
    face_x_min = 0
if face_y_min < 0:
    face_y_min = 0
if face_x_max > 1024:
    face_x_max = 1024
if face_y_max > 1024:
    face_y_max = 1024

img_base = np.array(img_base)
img_no_mouth = np.array(img_no_mouth)
img_no_eyes = np.array(img_no_eyes)

img_face = img_base[face_x_min:face_x_max, face_y_min:face_y_max]
img_face_no_mouth = img_no_mouth[face_x_min:face_x_max, face_y_min:face_y_max]
img_face_no_eyes = img_no_eyes[face_x_min:face_x_max, face_y_min:face_y_max]
img_face = cv2.resize(img_face, (1024,1024))
img_face_no_mouth = cv2.resize(img_face_no_mouth, (1024,1024))
img_face_no_eyes = cv2.resize(img_face_no_eyes, (1024,1024))

# revert to original size
img_face = cv2.resize(img_face, (ho,wo))
img_face_no_mouth = cv2.resize(img_face_no_mouth, (ho,wo))
img_face_no_eyes = cv2.resize(img_face_no_eyes, (ho,wo))
img_base = cv2.resize(img_base, (ho,wo))
img_no_mouth = cv2.resize(img_no_mouth, (ho,wo))
img_no_eyes = cv2.resize(img_no_eyes, (ho,wo))

cv2.imwrite(join(output_dir,f'{label_name.split("/")[0]}_base_face.png'), cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR))
cv2.imwrite(join(output_dir,f'{label_name.split("/")[0]}_no_mouth_face.png'), cv2.cvtColor(img_face_no_mouth, cv2.COLOR_RGB2BGR))
cv2.imwrite(join(output_dir,f'{label_name.split("/")[0]}_no_eyes_face.png'), cv2.cvtColor(img_face_no_eyes, cv2.COLOR_RGB2BGR))
cv2.imwrite(join(output_dir,f'{label_name.split("/")[0]}_base.png'), cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR))
cv2.imwrite(join(output_dir,f'{label_name.split("/")[0]}_no_mouth.png'), cv2.cvtColor(img_no_mouth, cv2.COLOR_RGB2BGR))
cv2.imwrite(join(output_dir,f'{label_name.split("/")[0]}_no_eyes.png'), cv2.cvtColor(img_no_eyes, cv2.COLOR_RGB2BGR))

