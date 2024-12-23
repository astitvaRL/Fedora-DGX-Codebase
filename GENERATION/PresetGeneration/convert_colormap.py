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

from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from utils.deform import tps_warp_box_mouth, tps_warp_preset_eyes, tps_warp_preset_mouth
from utils.preset_config import PresetConfig

import sys
sys.path.append('./utils/external/SAM/')
from utils.SemanticSegmentation import SemanticSegmentationFace, SemanticSegmentationAll
from utils.SegmentDrawings import SAM_face
from utils.stylization import ControllableStylization

join = os.path.join


# set paths
data_root = "/mnt/users_scratch/astitva/DATA/"
labels_definition_file_path = "./label_definition.json"
colormaps_path = join(data_root, "FACES_DRAWINGS16k/segmaps")

output_vis_root = join(data_root, "FACES_DRAWINGS16k/vis")
os.makedirs(output_vis_root, exist_ok=True)

# load semantic definitions
semantics_src = SemanticSegmentationFace(labels_definition_file_path)
semantics_tgt = SemanticSegmentationAll(labels_definition_file_path)

breakpoint()

# load labels
start = 0
end = -1
labels = sorted(os.listdir(colormaps_path))[start:end]

# iterate over images
for label_name in tqdm(labels):

    # load images
    label_full = cv2.imread(join(colormaps_path, label_name))
    label_full = cv2.cvtColor(label_full, cv2.COLOR_BGR2RGB)

    cv2.imwrite(join(output_vis_root, label_name), label_full)

