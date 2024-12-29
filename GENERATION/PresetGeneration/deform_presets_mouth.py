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
# from color_transfer import color_transfer

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


# set paths
data_root = "/mnt/users_scratch/astitva/DATA/"
labels_definition_file_path = "./label_definition.json"
image_dir_name = "MANIFOLD/animated_drawings_images_prior_april22/cropped_image"
label_id_dir_name = "AD_SegMaps/labels_7k_1024"
preset_dir = "./presets"
preset_class = "arpabets"  # DON'T FORGET TO CHANGE CANONICAL COORDINATES & CANNY THRESHOLDS ACCORDINGLY IN THE SHAPE & STYLIZATION SCRIPTS

# prest configuration
preset_config = PresetConfig(preset_class)
# preset_prompts = preset_config.config["prompts"]
shape_ids = preset_config.config["shape_ids"]
output_root = f"/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/PresetGeneration/OUTPUT/DEFORMED_PRESETS/{preset_class}"

os.makedirs(output_root, exist_ok=True)

# load inpainting model
inpainting_model = SimpleLama()

# load semantic definitions
semantics = SemanticSegmentationAll(labels_definition_file_path)
semantics_face = SemanticSegmentationFace(labels_definition_file_path)

#load SAM model
sam_face_model = SAM_face()

# load refiner
refiner = segref.Refiner(device="cuda:0")  # device can also be 'cpu'

# load labels
start = 0
end = -1
labels = sorted(os.listdir(join(data_root, label_id_dir_name)))[start:end]

# iterate over images
for label_name in tqdm(labels):

    condition = label_name.startswith('07ae7e78bf78421cb2219c236d156cbd') or label_name.startswith('0a5b805185614f839b5f015b650970dc') or label_name.startswith('07cdc5cab5af41e2a102e5453678ca72') or label_name.startswith('07a23dcc43ea436ebf6b75b04c7611d4') or label_name.startswith('07a73dddd01848a19eb97839500a2843') or label_name.startswith('07aba8228cb54faf9f6e4e6cab561331') or label_name.startswith('07ad6ccc1ac34288ab7c4f0a013ba3c3') or label_name.startswith('07aedcb335a04981a016c0c7efed77ba') or label_name.startswith('07af86863ff04f3abc0e0442cdb882b7') or label_name.startswith('07b1a55d68b9425caccb1aadcc58379a') or label_name.startswith('07b9b86ec22e48e1807785b2cd64cb76') or label_name.startswith('07b6c37d7a6944ee98f544d633defeeb') or label_name.startswith('07b8bf4a421744c9b7f985cf6e8fe544') or label_name.startswith('07b97debed234daaa04313b000637b81') or label_name.startswith('07babac076024ce7a89b72e93d16cc99') or label_name.startswith('07c4c1c55b5b4098bf8f5b96defd6d2c') or label_name.startswith('07c7e9cc8d364378912accf8b9c5eb57')
    if not condition:
        continue

    # create output directory
    output_dir_metadata = join(output_root, label_name.split(".")[0], 'metadata')
    output_dir_images = join(output_root, label_name.split(".")[0], 'images')
    output_dir_labels = join(output_root, label_name.split(".")[0], 'labels')
    os.makedirs(output_dir_metadata, exist_ok=True)
    os.makedirs(f'{output_dir_metadata}/original_image/', exist_ok=True)
    os.makedirs(f'{output_dir_metadata}/original_label/', exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    # load images
    img_name = f"{label_name.split('_')[0]}.png"
    img_full = cv2.imread(join(data_root, image_dir_name, img_name))
    w_orig, h_orig = img_full.shape[:2]
    img_full = cv2.resize(img_full, (1024, 1024))
    img_full = bgr_conversion(img_full)
    label_full = cv2.imread(join(data_root, label_id_dir_name, label_name))
    label_full = bgr_conversion(label_full)
    label_full = cv2.resize(label_full, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label_id = semantics.colors_to_labels(label_full)
    label_face_id = semantics_face.colors_to_labels(label_full)
   
    # inpainting
    kernel = np.ones((5, 5), np.uint8)
    inpainting_mask = label_id == 0  # default background
    if preset_class == "mouth" or preset_class == "mouth_talk" or preset_class == "arpabets":
        inpainting_mask = (label_id == 3) | (label_id == 23) | (label_id == 17)
    elif preset_class == "eyes":
        inpainting_mask = (label_id == 4) | (label_id == 22)
    inpainting_mask = cv2.dilate(
        255 * inpainting_mask.astype("uint8"), kernel, iterations=3
    )
    base_image = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask)
   
    # extract face region
    roi = label_face_id>0
    Xs, Ys = np.where(roi)
    if len(Xs)==0 or len(Ys)==0: continue
    padding = 10
    x_adj = padding
    y_adj = padding
    x_min, x_max = np.min(Xs) - x_adj, np.max(Xs) + x_adj
    y_min, y_max = np.min(Ys) - y_adj, np.max(Ys) + y_adj
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > 1024:
        x_max = 1024
    if y_max > 1024:
        y_max = 1024
    img_cropped = img_full[x_min:x_max, y_min:y_max]
    label_id = label_id[x_min:x_max, y_min:y_max]
    label_face_id_cropped = label_face_id[x_min:x_max, y_min:y_max]
    roi_cropped = roi[x_min:x_max, y_min:y_max]
    # mouth_region_cropped = mouth_region[x_min:x_max, y_min:y_max]
    base_image_cropped = np.array(base_image)[x_min:x_max, y_min:y_max]
    h_crop, w_crop, _ = img_cropped.shape

    roi_cropped = cv2.resize(roi_cropped.astype('uint8'), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    img_cropped = cv2.resize(img_cropped, (1024, 1024))
    base_image_cropped = cv2.resize(base_image_cropped, (1024, 1024))
    label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label_face_id = cv2.resize(label_face_id_cropped, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    dims = {'w_crop':w_crop,'h_crop':h_crop, 'w_orig':w_orig, 'h_orig':h_orig}
    np.savez(f'{output_dir_metadata}/dims.npz',**dims)
    cv2.imwrite(f'{output_dir_metadata}/original.png', bgr_conversion(img_full))
    cv2.imwrite(f'{output_dir_metadata}/original_image/original.png', bgr_conversion(img_cropped))
    cv2.imwrite(f'{output_dir_metadata}/inpainted.png', bgr_conversion(np.array(base_image)))
    cv2.imwrite(f'{output_dir_metadata}/inpainted_face.png', bgr_conversion(base_image_cropped))
    cv2.imwrite(f'{output_dir_metadata}/segmap.png', bgr_conversion(label_full))
    cv2.imwrite(f'{output_dir_metadata}/label.png', label_id)
    cv2.imwrite(f'{output_dir_metadata}/original_label/original.png', label_face_id)

    deformed_masks_all = []
    # iterate over presets for deformation
    for preset_idx in tqdm(range(len(shape_ids))):
        # load preset
        shape_id = shape_ids[preset_idx]
        preset_image = cv2.imread(join(preset_dir, preset_class, f"{shape_id}.png"), -1)
        preset_image = cv2.resize(
            preset_image, (1024, 1024), interpolation=cv2.INTER_NEAREST
        )

        # get preset shape
        tps_output = None
        if preset_class == "mouth" or preset_class == "mouth_talk" or preset_class == "arpabets":
            tps_output = tps_warp_preset_mouth(img_cropped, label_id, preset_image)


        elif preset_class == "eyes":
            tps_output = tps_warp_preset_eyes(
                label_id=label_face_id, label_type=preset_class, preset_shape=preset_image
            )
        if tps_output == -1 or tps_output is None:
            print("Skipping...")
            continue

        # mask from tps  deformation
        deformed_mask = tps_output[1]
        deformed_preset = tps_output[3]
        # if shape_id=='18' or shape_id=='19': #remove the green area around the preset (which was used to enable salient point detection on preset)
        deformed_mask[deformed_preset[:,:,1]==255]=0
        teeth = (deformed_preset[:,:,0]>0) & (deformed_preset[:,:,0]<200)
        tongue = deformed_preset[:,:,0]>200

        # place preset shape over inpainted image
        base_image_np = np.array(base_image_cropped)
        # cond_image = base_image_np.copy().astype("float32")

        cond_mask = inpainting_mask[x_min:x_max, y_min:y_max]
        cond_mask = cv2.resize(cond_mask, (1024,1024), interpolation = cv2.INTER_NEAREST)
        label_id_cropped = label_id.copy()
        label_id_cropped[cond_mask==255] = 6

        #smoth deformed mask boundaries
        blur_kernel = (53,53)
        # deformed_mask = cv2.GaussianBlur(deformed_mask.astype('uint8')*255, blur_kernel, 0)
        deformed_mask = deformed_mask>0

        # prepare conditioning image
        label_id_cropped[deformed_mask] = 3
        label_id_cropped[teeth] = 17
        label_id_cropped[tongue] = 23
        cond_image = semantics.labels_to_colors(label_id_cropped)

        cond_face_labels = semantics_face.colors_to_labels(cond_image)

        cv2.imwrite(f'{output_dir_images}/{shape_id}.png', bgr_conversion(img_cropped))
        cv2.imwrite(f'{output_dir_labels}/{shape_id}.png', cond_face_labels)

        # include every part in final binary mask
        deformed_mask = (deformed_mask>0) | (teeth>0) | (tongue>0)
        deformed_masks_all.append(deformed_mask)

    # generating via SEAN
    sean_outdir = join(output_root, label_name.split(".")[0], 'sean_output')
    os.makedirs(sean_outdir, exist_ok=True)
    os.environ['SEAN_OUTDIR'] = f'{sean_outdir}'
    os.environ['IMAGE_DIR'] = f'{output_dir_metadata}/original_image'
    os.environ['LABEL_DIR'] = f'{output_dir_metadata}/original_label'
    os.environ['PRESETS_IMAGE_DIR'] = f'{output_dir_images}'
    os.environ['PRESETS_LABEL_DIR'] = f'{output_dir_labels}'
    # run the script
    os.system('bash seg2image.sh')

    # compositing
    asset_outdir = join(output_root, label_name.split(".")[0], 'assets')
    composite_outdir = join(output_root, label_name.split(".")[0], 'assets_composited')
    os.makedirs(asset_outdir, exist_ok=True)
    os.makedirs(composite_outdir, exist_ok=True)
    for preset_idx in tqdm(range(len(shape_ids))):
        shape_id = shape_ids[preset_idx]
        deformed_mask = deformed_masks_all[preset_idx]
        
        generated = cv2.imread(f'{sean_outdir}/{preset_idx}.png')
        generated = bgr_conversion(generated)

        deformed_mask_im = cv2.dilate(deformed_mask.astype('uint8')*255, (23,23))
        deformed_mask_im = refiner.refine(generated, deformed_mask_im, fast=False, L=900)
        deformed_mask_im = np.repeat(deformed_mask_im[..., np.newaxis], 3, axis=2)
        deformed_mask_im = cv2.blur(deformed_mask_im, (23, 23))
        deformed_mask_im = deformed_mask_im.astype("float32")/255
        final_image = base_image_np * (1 - deformed_mask_im) + generated * deformed_mask_im
        final_image = final_image.astype("uint8")
        asset_image = np.concatenate([generated,255*deformed_mask_im[:,:,0:1]],-1).astype('uint8')

        # composite on full image
        final_image_resized = cv2.resize(final_image, (w_crop, h_crop))
        asset_image_resized = cv2.resize(asset_image, (w_crop, h_crop))
        img_composited = img_full.copy()
        img_composited[x_min:x_max, y_min:y_max] = final_image_resized
        asset_image_full = np.zeros((img_full.shape[0],img_full.shape[1],4)).astype('uint8')
        asset_image_full[x_min:x_max, y_min:y_max] = asset_image_resized

        asset_image_cropped = asset_image_full[x_min:x_max, y_min:y_max]
        img_composited_cropped = img_composited[x_min:x_max, y_min:y_max]
        asset_image_cropped = cv2.resize(asset_image_cropped, (1024, 1024))
        img_composited_cropped = cv2.resize(img_composited_cropped, (1024, 1024))

        asset_image_final = cv2.resize(asset_image_full, (h_orig, w_orig))
        img_composited_final = cv2.resize(img_composited, (h_orig, w_orig))

        # save assets
        cv2.imwrite(f'{asset_outdir}/{preset_class}_{shape_id}.png', bgr_conversion(asset_image_final))
        cv2.imwrite(f'{asset_outdir}/{preset_class}_{shape_id}_face.png', bgr_conversion(asset_image_cropped))
        # save composited images
        cv2.imwrite(f'{composite_outdir}/{preset_class}_{shape_id}.png', bgr_conversion(img_composited_final))
        cv2.imwrite(f'{composite_outdir}/{preset_class}_{shape_id}_face.png', bgr_conversion(img_composited_cropped))
