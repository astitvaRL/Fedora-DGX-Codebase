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


# set paths
data_root = "/mnt/users_scratch/hjessmith/DATA/"
labels_definition_file_path = "./label_definition.json"
image_dir_name = "MANIFOLD/animated_drawings_images_prior_april22/cropped_image"
# image_dir_name = "IN_THE_WILD_faces"
preset_dir = "./presets"
preset_class = "eyes"  # DON'T FORGET TO CHANGE CANONICAL COORDINATES & CANNY THRESHOLDS ACCORDINGLY IN THE SHAPE & STYLIZATION SCRIPTS
out_parent_dir = "/mnt/users_scratch/hjessmith/WORKSPACE/Fedora-DGX-Codebase/GENERATION/PresetGeneration/OUTPUT/DRAWINGS"

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
    preset_cache_dict[shape_id] = preset_image


# load inpainting model
inpainting_model = SimpleLama()

# load semantic definitions
semantics = SemanticSegmentationAll(labels_definition_file_path)
semantics_face = SemanticSegmentationFace(labels_definition_file_path)

#load SAM model
sam_ckpt_dir = '/mnt/users_scratch/hjessmith/CHECKPOINTS/checkpoints/'
semsegpipe = SAM_face(ckpt_dir=sam_ckpt_dir)
semsegpipe.load_best_eval_ckpt = False
semsegpipe.epoch_coarse = 500
semsegpipe.epoch_face = 500
semsegpipe.epoch_fine = 500


# load refiner
refiner = segref.Refiner(device="cuda:0")  # device can also be 'cpu'

# load labels
start = 0
end = -1
images = sorted(os.listdir(join(data_root, image_dir_name)))[start:end]

# additional controls
outline_only = False
save_deformed_segmaps = True


# PRESET DEFORMATION

for input_image_name in tqdm(images):

    start = time.time()

    condition = input_image_name.startswith('07aedcb335a04981a016c0c7efed77ba')
    # condition = input_image_name.startswith('07a0fc851f2c48058d53325894e49496') or input_image_name.startswith('0a0a4add3fb9438babce15098f9efad8') or input_image_name.startswith('07cdc5cab5af41e2a102e5453678ca72') or input_image_name.startswith('07a23dcc43ea436ebf6b75b04c7611d4') or input_image_name.startswith('0a5b805185614f839b5f015b650970dc') or input_image_name.startswith('07aba8228cb54faf9f6e4e6cab561331') or input_image_name.startswith('0a6bf1b9d15842b6822a92a6b536faf1') or input_image_name.startswith('07aedcb335a04981a016c0c7efed77ba') or input_image_name.startswith('07af86863ff04f3abc0e0442cdb882b7') or input_image_name.startswith('07b1a55d68b9425caccb1aadcc58379a') or input_image_name.startswith('07b9b86ec22e48e1807785b2cd64cb76') or input_image_name.startswith('07b6c37d7a6944ee98f544d633defeeb') or input_image_name.startswith('07b8bf4a421744c9b7f985cf6e8fe544') or input_image_name.startswith('0a3b9f4c787743458c7ca1cc77b902ea') or input_image_name.startswith('07babac076024ce7a89b72e93d16cc99') or input_image_name.startswith('0934abc208ff441bb98a9b849997aac4') or input_image_name.startswith('07cdc5cab5af41e2a102e5453678ca72')
    # condition = input_image_name.startswith('16')
    if not condition:
        continue

    # create output directory
    output_dir_metadata = join(output_root, input_image_name.split(".")[0], 'metadata')
    output_dir_images = join(output_root, input_image_name.split(".")[0], 'tmp_images')
    output_dir_labels = join(output_root, input_image_name.split(".")[0], 'tmp_labels')
    output_dir_segmaps = join(output_root, input_image_name.split(".")[0], 'tmp_segmaps')
    os.makedirs(output_dir_metadata, exist_ok=True)
    os.makedirs(f'{output_dir_metadata}/original_image/', exist_ok=True)
    os.makedirs(f'{output_dir_metadata}/original_label/', exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)
    if save_deformed_segmaps:
        os.makedirs(output_dir_segmaps, exist_ok=True)
 
    # load images
    img_full = cv2.imread(join(data_root, image_dir_name, input_image_name))
    w_orig, h_orig = img_full.shape[:2]
    img_full = cv2.resize(img_full, (1024, 1024))
    img_full = bgr_conversion(img_full)

    # semantic segmentation 
    sam_output = semsegpipe.predict(img_full)
    if not sam_output[0]:
        print("Segmentation Failed! Skipping...")
        continue
    label_full = sam_output[1]

    # segmap to labels
    label_id = semantics.colors_to_labels(label_full)
    label_face_id = semantics_face.colors_to_labels(label_full)

    # inpainting
    kernel = np.ones((5, 5), np.uint8)
    inpainting_mask = label_face_id == 0  # default background
    if preset_class == "mouth":
        inpainting_mask = (label_face_id == 2) | (label_face_id == 7) | (label_face_id == 10)
    elif preset_class == "eyes":
        inpainting_mask = (label_face_id == 3) | (label_face_id == 9)
    inpainting_mask = cv2.dilate(
        255 * inpainting_mask.astype("uint8"), kernel, iterations=3
    )
    base_image = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask)

    # inpainting both eyes and mouth
    kernel = np.ones((5, 5), np.uint8)
    inpainting_mask_em = (label_face_id == 2) | (label_face_id == 7) | (label_face_id == 10) | (label_face_id == 3) | (label_face_id == 9)
    inpainting_mask_em = cv2.dilate(
        255 * inpainting_mask_em.astype("uint8"), kernel, iterations=3
    )
    base_image_em = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask_em)
   
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
    base_image_em_cropped = np.array(base_image_em)[x_min:x_max, y_min:y_max]
    h_crop, w_crop, _ = img_cropped.shape

    roi_cropped = cv2.resize(roi_cropped.astype('uint8'), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    img_cropped = cv2.resize(img_cropped, (1024, 1024))
    base_image_cropped = cv2.resize(base_image_cropped, (1024, 1024))
    base_image_em_cropped = cv2.resize(base_image_em_cropped, (1024, 1024))
    label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label_face_id = cv2.resize(label_face_id_cropped, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    dims = {'w_crop':w_crop,'h_crop':h_crop, 'w_orig':w_orig, 'h_orig':h_orig}
    np.savez(f'{output_dir_metadata}/dims.npz',**dims)
    cv2.imwrite(f'{output_dir_metadata}/image_input.png', bgr_conversion(img_full))
    cv2.imwrite(f'{output_dir_metadata}/image_face.png', bgr_conversion(img_cropped))
    cv2.imwrite(f'{output_dir_metadata}/original_image/original.png', bgr_conversion(img_cropped))
    cv2.imwrite(f'{output_dir_metadata}/inpainted_eyes_only.png', bgr_conversion(np.array(base_image)))
    cv2.imwrite(f'{output_dir_metadata}/inpainted_eyes_mouth.png', bgr_conversion(np.array(base_image_em)))
    cv2.imwrite(f'{output_dir_metadata}/inpainted_face_eyes_only.png', bgr_conversion(base_image_cropped))
    cv2.imwrite(f'{output_dir_metadata}/inpainted_face_eyes_mouth.png', bgr_conversion(base_image_em_cropped))
    cv2.imwrite(f'{output_dir_metadata}/segmap.png', bgr_conversion(label_full))
    cv2.imwrite(f'{output_dir_metadata}/label.png', label_id)
    cv2.imwrite(f'{output_dir_metadata}/original_label/original.png', label_face_id)

    deformed_masks_all = {}

    # iterate over presets for deformation
    for shape_idx in tqdm(range(len(shape_ids))):

        shape_id = shape_ids[shape_idx]
        preset_image = preset_cache_dict[shape_id]

        # get deformed preset
        tps_output = None
        if preset_class == "mouth":
            tps_output = tps_warp_preset_mouth(img_cropped, label_face_id, preset_image)

        elif preset_class == "eyes":
            tps_output = tps_warp_preset_eyes(
                label_id=label_face_id, label_type=preset_class, preset_shape=preset_image
            )
        if tps_output == -1 or tps_output is None:
            print("Skipping...")
            continue

        # mask from tps  deformation
        deformed_mask = tps_output[1]
        deformed_preset = tps_output[0]
        # if shape_id=='18' or shape_id=='19': #remove the green area around the preset (which was used to enable salient point detection on preset)
        deformed_mask[deformed_preset[:,:,1]==255]=0
        pupil = deformed_preset[:,:,0]>200
        # eye_region = deformed_preset[:,:,0]>250

        # place preset shape over inpainted image
        base_image_np = np.array(base_image_cropped)
        # cond_image = base_image_np.copy().astype("float32")

        cond_mask = inpainting_mask[x_min:x_max, y_min:y_max]
        cond_mask = cv2.resize(cond_mask, (1024,1024), interpolation = cv2.INTER_NEAREST)
        label_face_id_tmp = label_face_id.copy()
        label_face_id_tmp[cond_mask==255] = 5

        #smoth deformed mask boundaries
        blur_kernel = (53,53)
        # deformed_mask = cv2.GaussianBlur(deformed_mask.astype('uint8')*255, blur_kernel, 0)
        deformed_mask = deformed_mask>0

        # prepare conditioning image
        label_face_id_tmp[deformed_mask] = 9
        label_face_id_tmp[pupil] = 3
        # label_face_id_tmp[empty] = 5
        cond_image = semantics_face.labels_to_colors(label_face_id_tmp)

        # cond_face_labels = semantics_face.colors_to_labels(cond_image)
        cond_face_labels = label_face_id_tmp.copy()

        if outline_only:
            cond_face_outline = cond_face_labels.copy()
            cond_face_edges = cv2.Canny(cond_image, 0, 150)
            kernel_outline = np.ones((5, 5), np.uint8) 
            cond_face_edges = cv2.dilate(cond_face_edges, kernel, iterations=1)
            mask = (cond_face_labels==2) | pupil
            cond_face_outline[mask] = 5
            mask_edges = (cond_face_labels==2) & (cond_face_edges>0)
            cond_face_outline[mask_edges] = 9
            mask_edges = pupil & (cond_face_edges>0)
            cond_face_outline[mask_edges] = 3


        cv2.imwrite(f'{output_dir_images}/{shape_id}.png', bgr_conversion(img_cropped))
        if outline_only:
            cv2.imwrite(f'{output_dir_labels}/{shape_id}.png', cond_face_outline)
            if save_deformed_segmaps:
                cond_face_outline_vis = semantics_face.labels_to_colors(cond_face_outline)
                cv2.imwrite(f'{output_dir_segmaps}/{shape_id}.png', bgr_conversion(cond_face_outline_vis))
        else:
            cv2.imwrite(f'{output_dir_labels}/{shape_id}.png', cond_face_labels)
            if save_deformed_segmaps:
                cond_face_labels_vis = semantics_face.labels_to_colors(cond_face_labels)
                cv2.imwrite(f'{output_dir_segmaps}/{shape_id}.png', bgr_conversion(cond_face_labels_vis))


        # include every part in final binary mask
        deformed_mask = (deformed_mask>0) | (pupil>0) 
        deformed_masks_all[shape_id] = deformed_mask

    # GENERATION VIA SEAN
    sean_outdir = join(output_root, input_image_name.split(".")[0], 'sean_output')
    os.makedirs(sean_outdir, exist_ok=True)
    os.environ['SEAN_OUTDIR'] = f'{sean_outdir}'
    os.environ['META_DIR'] = f'{output_dir_metadata}'
    os.environ['IMAGE_DIR'] = f'{output_dir_metadata}/original_image'
    os.environ['LABEL_DIR'] = f'{output_dir_metadata}/original_label'
    os.environ['PRESETS_IMAGE_DIR'] = f'{output_dir_images}'
    os.environ['PRESETS_LABEL_DIR'] = f'{output_dir_labels}'
    # run the script
    os.system('bash seg2image.sh')



    # COMPOSITING

    asset_outdir = join(output_root, input_image_name.split(".")[0], 'assets')
    composite_outdir = join(output_root, input_image_name.split(".")[0], 'assets_composited')
    os.makedirs(asset_outdir, exist_ok=True)
    os.makedirs(composite_outdir, exist_ok=True)

    for key in deformed_masks_all.keys():
       
        shape_id = shape_ids[int(key)]
        deformed_mask = deformed_masks_all[shape_id]
        
        generated = cv2.imread(f'{sean_outdir}/{shape_id}.png')
        generated = bgr_conversion(generated)

        deformed_mask_im = cv2.dilate(deformed_mask.astype('uint8')*255, (23,23))
        deformed_mask_im = refiner.refine(generated, deformed_mask_im, fast=False, L=900)
        deformed_mask_im = np.repeat(deformed_mask_im[..., np.newaxis], 3, axis=2)
        deformed_mask_im = cv2.blur(deformed_mask_im, (17, 17))
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

    end = time.time()
    print()
    print(f'DONE! Took {end-start} seconds!')