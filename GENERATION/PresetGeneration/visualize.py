import os
# setup cache path for huggingface
os.environ['CACHE_DIR'] = '/mnt/users_scratch/astitva/CACHE/'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HOME'] = os.environ['CACHE_DIR']
os.environ['HF_DATASETS_CACHE'] = os.environ['CACHE_DIR']
os.environ['TRANSFORMERS_CACHE']= os.environ['CACHE_DIR']

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from simple_lama_inpainting import SimpleLama

from utils.SemanticSegmentation import SemanticSegmentationAll
from utils.shape import tps_warp_preset_mouth, tps_warp_preset_eyes
from utils.stylization import ControllableStylization
from utils.preset_config import PresetConfig

join = os.path.join

# set paths
# set paths
data_root = '/mnt/users_scratch/astitva/DATA/'
labels_definition_file_path = './label_definition.json'
image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 
preset_dir = './presets'
preset_class = 'eyes' # DONT FORGET TO CHANGE CANNY THRESHOLDS ACCORDINGLY IN THE STYLIZATION SCRIPT
output_root = f'output_{preset_class}'
visualize_dir = 'visualize_eyes'
os.makedirs(visualize_dir, exist_ok=True)

folders = os.listdir(output_root)

# prest configuration
preset_config = PresetConfig(preset_class)
preset_prompts = preset_config.config['prompts']
shape_ids = preset_config.config['shape_ids']

for folder in tqdm(folders):
    subdir = join(output_root, folder)
    files = sorted(os.listdir(subdir))
    fig, ax = plt.subplots(1,6, figsize=(60,10))
    fig.tight_layout()
    ax[0].imshow(Image.open(join(subdir, files[-1])))
    # ax[0].set_title('Original Image', fontsize=35)
    ax[0].axis('off')
    for idx in range(len(files)-1):
        ax[idx+1].imshow(Image.open(join(subdir, files[idx])))
        # ax[idx+1].set_title(preset_prompts[idx], fontsize=35)
        ax[idx+1].axis('off')
    save_name = (subdir.split('/')[-1]).split('_')[0] + '.png'
    plt.savefig(join(visualize_dir, save_name))
    plt.close()
