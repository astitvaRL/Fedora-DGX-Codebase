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

from utils.SemanticSegmentation import SemanticSegmentationAll
from utils.shape import tps_warp_preset
from utils.stylization import ControllableStylization

join = os.path.join

# set paths
data_root = '/mnt/users_scratch/astitva/DATA/'
labels_definition_file_path = './label_definition.json'
image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 
preset_dir = './presets'
preset_type = 'mouth'
shape_id = '3'
output_dir = join('output')
os.makedirs(output_dir, exist_ok=True)

# load stylization model
control_stylization = ControllableStylization()

# load semantic definitions
semantics = SemanticSegmentationAll(labels_definition_file_path)

# load labels
labels = sorted(os.listdir(join(data_root, label_id_dir_name)))[:5]

# load preset
preset_image = cv2.imread(join(preset_dir, preset_type, f'{shape_id}.png'),-1)
preset_image = cv2.resize(preset_image, (1024,1024), interpolation=cv2.INTER_NEAREST)

for label_name in tqdm(labels):
    img_name = f"{label_name.split('_')[0]}.png"
    img = cv2.imread(join(data_root, image_dir_name, img_name))
    img = cv2.resize(img, (1024,1024))
    label = cv2.imread(join(data_root, label_id_dir_name, label_name))
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    label_id = semantics.colors_to_labels(label)
    tps_output = tps_warp_preset(label_id=label_id, type='mouth', shape_id='5', preset_image=preset_image)
    if tps_output==-1:
        print('Skipping...')
        continue
    shape_mask = tps_output[0]
    cond_image = img.copy()
    cond_image[shape_mask] = np.array([0,0,0])
    # cv2.imwrite(join(output_dir, f'{label_name.split("_")[0]}_{preset_type}_{shape_id}.png'), img)

    # stylization
    ref_prompt = "an image of a cartoon character"
    style_prompt = "hand drawn"
    target_prompt = "an image of a cartoon character with tongue out"
    ref_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    generated, conditioning = control_stylization.generate(ref_img, cond_image, ref_prompt, style_prompt, target_prompt)

    # save images
    cv2.imwrite(join(output_dir, f'{label_name.split("_")[0]}.png'), img) # original image
    generated.save(join(output_dir, f'{label_name.split("_")[0]}_{preset_type}_{shape_id}.png'))
    conditioning.save(join(output_dir, f'{label_name.split("_")[0]}_{preset_type}_{shape_id}_canny.png'))
    
