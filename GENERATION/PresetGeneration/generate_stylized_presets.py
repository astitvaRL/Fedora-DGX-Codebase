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
data_root = '/mnt/users_scratch/astitva/DATA/'
labels_definition_file_path = './label_definition.json'
image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 
preset_dir = './presets'
preset_class = 'mouth' # DONT FORGET TO CHANGE CANNY THRESHOLDS ACCORDINGLY IN THE STYLIZATION SCRIPT
preset_config = PresetConfig(preset_class)
preset_prompts = preset_config.config['prompts']
shape_ids = preset_config.config['shape_ids']
output_root = f'output_{preset_class}_v2'

# load stylization model
control_stylization = ControllableStylization()

# load inpainting model
inpainting_model = SimpleLama()

# load semantic definitions
semantics = SemanticSegmentationAll(labels_definition_file_path)

# load labels
start = 6000
end = -1
labels = sorted(os.listdir(join(data_root, label_id_dir_name)))

# iterate over images
for label_name in tqdm(labels):

    # create output directory
    output_dir = join(output_root, label_name.split('_')[0])
    os.makedirs(output_dir, exist_ok=True)

    # load images
    img_name = f"{label_name.split('_')[0]}.png"
    img_full = cv2.imread(join(data_root, image_dir_name, img_name))
    img_full = cv2.resize(img_full, (1024,1024))
    img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
    label_full = cv2.imread(join(data_root, label_id_dir_name, label_name))
    label_full = cv2.cvtColor(label_full, cv2.COLOR_BGR2RGB)
    label_id = semantics.colors_to_labels(label_full)
    #inpainting
    kernel = np.ones((5,5),np.uint8)
    inpainting_mask =  label_id==0 # default background
    if preset_class=='mouth':
        inpainting_mask =  (label_id == 3) | (label_id == 23) | (label_id == 17)
    elif preset_class=='eyes':
        inpainting_mask =  (label_id == 4) | (label_id == 22)
    inpainting_mask = cv2.dilate(255*inpainting_mask.astype('uint8'), kernel, iterations=3)
    img_base = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask)
    # extract face region
    face_region = label_id==6
    Xs, Ys = np.where(face_region)
    padding = 10
    x_min, x_max = np.min(Xs)-padding, np.max(Xs)+padding
    y_min, y_max = np.min(Ys)-padding, np.max(Ys)+padding
    if x_min<0: x_min = 0
    if y_min<0: y_min = 0
    if x_max>1024: x_max = 1024
    if y_max>1024: y_max = 1024
    img_cropped = img_full[x_min:x_max, y_min:y_max]
    label_id = label_id[x_min:x_max, y_min:y_max]
    img_base = np.array(img_base)[x_min:x_max, y_min:y_max]
    img_cropped = cv2.resize(img_cropped, (1024,1024))
    img_base = cv2.resize(img_base, (1024,1024))
    label_id = cv2.resize(label_id, (1024,1024), interpolation=cv2.INTER_NEAREST)

    # save original image
    Image.fromarray(img_cropped).save(join(output_dir, f'{label_name.split("_")[0]}_original.png'))
    
    # define reference prompt and style
    ref_prompt = "face of a cartoon character"
    style_prompt = "hand drawn"

    # iterate over presets
    for preset_idx in tqdm(range(len(preset_prompts))):
        # load preset
        shape_id = shape_ids[preset_idx]
        preset_image = cv2.imread(join(preset_dir, preset_class, f'{shape_id}.png'),-1)
        preset_image = cv2.resize(preset_image, (1024,1024), interpolation=cv2.INTER_NEAREST)
        
        # get preset shape
        tps_output = None
        if preset_class=='mouth':
            tps_output = tps_warp_preset_mouth(label_id=label_id, type=preset_class, preset_image=preset_image)
        elif preset_class=='eyes':
            tps_output = tps_warp_preset_eyes(label_id=label_id, type=preset_class, preset_image=preset_image)
        if tps_output==-1 or tps_output is None:
            print('Skipping...')
            continue
        shape_mask = tps_output[0]
        original_mouth_mask = tps_output[1]

        # place preset shape over inpainted image
        cond_image = img_base.copy()
        cond_image[shape_mask] = np.array([0,0,0])

        # stylization
        target_prompt = f"face of a cartoon character {preset_prompts[preset_idx]}"
        ref_img = img_cropped.copy()
        generated, conditioning = control_stylization.generate(ref_img, cond_image, ref_prompt, style_prompt, target_prompt)
        img_base_np = np.array(img_base)
        generated = np.array(generated)
        blurred_mask = cv2.blur(shape_mask.astype('float32'), (7,7))
        blurred_mask = np.repeat(blurred_mask[..., np.newaxis], 3, axis=2)
        img_base_np = img_base_np * (1-blurred_mask) + generated * blurred_mask
        img_base_np = img_base_np.astype('uint8')
        
        # save images
        Image.fromarray(img_base_np).save(join(output_dir, f'{label_name.split("_")[0]}_{preset_class}_{shape_id}.png')) # generated image

    # invalidate the reference latent
    control_stylization.reference_latent = None
    control_stylization.inversion_callback = None
