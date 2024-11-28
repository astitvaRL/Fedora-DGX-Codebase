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
label_id_dir_name = "AD_SegMaps/labels_7k_mouth_type1_1024"
preset_dir = "./presets"
preset_class = "eyes"  # DON'T FORGET TO CHANGE CANONICAL COORDINATES & CANNY THRESHOLDS ACCORDINGLY IN THE SHAPE & STYLIZATION SCRIPTS

# prest configuration
preset_config = PresetConfig(preset_class)
preset_prompts = preset_config.config["prompts"]
shape_ids = preset_config.config["shape_ids"]
output_root = f"OUTPUT/output_{preset_class}_PRESET_type1"

# load stylization model
control_stylization = ControllableStylization()

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
labels = sorted(os.listdir(join(data_root, label_id_dir_name)))[start:end]

# iterate over images
for label_name in tqdm(labels):
    # label_name = "sample_seg.png"
    # img_name = "sample_bizzare.png"

    if not label_name.startswith('0a5b805185614f839b5f015b650970dc'):
        continue

    # create output directory
    output_dir = join(output_root, label_name.split("_")[0])
    os.makedirs(output_dir, exist_ok=True)

    # load images
    img_name = f"{label_name.split('_')[0]}.png"
    img_full = cv2.imread(join(data_root, image_dir_name, img_name))
    # img_full = cv2.imread(img_name)
    img_full = cv2.resize(img_full, (1024, 1024))
    img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
    label_full = cv2.imread(join(data_root, label_id_dir_name, label_name))
    # label_full = cv2.imread(label_name)
    label_full = cv2.cvtColor(label_full, cv2.COLOR_BGR2RGB)
    label_id = semantics.colors_to_labels(label_full)
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
    img_base = inpainting_model(Image.fromarray(img_full.copy()), inpainting_mask)
    # extract face region
    face_region = (label_id == 2) | (label_id == 3) | (label_id == 4) | (label_id == 5) | (label_id == 6) | (label_id == 12) | (label_id == 13) | (label_id == 17) | (label_id == 18) | (label_id == 22) | (label_id == 23)
    eyes_region = (label_id == 22) | (label_id == 4)  
    # eyes_region = face_region.copy() 
    # preset cropping window
    Xs, Ys = np.where(eyes_region)
    if len(Xs)==0 or len(Ys)==0: continue
    padding = 20
    difference = (np.max(Xs)-np.min(Xs)) - (np.max(Ys)-np.min(Ys))
    x_adj = padding
    y_adj = padding
    if difference<0: x_adj += np.abs(difference//2)
    else: y_adj += np.abs(difference//2)
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
    face_region_cropped = face_region[x_min:x_max, y_min:y_max]
    eyes_region_cropped = eyes_region[x_min:x_max, y_min:y_max]
    img_base = np.array(img_base)[x_min:x_max, y_min:y_max]
    h_crop, w_crop, _ = img_cropped.shape

   
    img_cropped = cv2.resize(img_cropped, (1024, 1024))
    img_base = cv2.resize(img_base, (1024, 1024))
    label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # # save original image
    # Image.fromarray(img_cropped).save(
    #     join(output_dir, f'{label_name.split("_")[0]}_original.png')
    # )

    # define reference prompt and style
    ref_prompt = "zoomed in eyes of a cartoon character"
    style_prompt = "hand drawn"

    # iterate over presets
    for preset_idx in tqdm(range(len(preset_prompts))):
        # load preset
        shape_id = shape_ids[preset_idx]
        preset_image = cv2.imread(join(preset_dir, preset_class, f"{shape_id}.png"), -1)
        preset_image = cv2.resize(
            preset_image, (1024, 1024), interpolation=cv2.INTER_NEAREST
        )

        # get preset shape
        tps_output = None
        if preset_class == "mouth" or preset_class == "mouth_talk" or preset_class == "arpabets":
            # tps_output = tps_warp_box_mouth(
            #     image=img_cropped,
            #     label_id=label_id,
            #     label_type=preset_class,
            #     preset_shape=preset_image,
            #     shape_id=shape_id,
            # )
            tps_output = tps_warp_preset_mouth(img_cropped, label_id, preset_image)

            # 
            # 
            # 
        elif preset_class == "eyes":
            tps_output = tps_warp_preset_eyes(
                label_id=label_id, label_type=preset_class, preset_shape=preset_image
            )
        if tps_output == -1 or tps_output is None:
            print("Skipping...")
            continue

        # deformed, _, mouth_pose = tps_output
        mouth_pose = 'preset'

        # resize crops to 1024x1024
        img_cropped = cv2.resize(img_cropped, (1024, 1024))
        img_base = cv2.resize(img_base, (1024, 1024))
        label_id = cv2.resize(label_id, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        # deformed = cv2.resize(deformed, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        face_region_cropped = cv2.resize(face_region_cropped.astype('uint8'), (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # mask from tps  deformation
        # deformed_mask = deformed[:, :, 3] == 255
        deformed_mask = tps_output[1]
        deformed_preset = tps_output[0]
        # if shape_id=='18' or shape_id=='19': #remove the green area around the preset (which was used to enable salient point detection on preset)
        deformed_mask[deformed_preset[:,:,1]==255]=0
        pupil = deformed_preset[:,:,0]>200
        empty = (deformed_preset[:,:,1]>200) & (deformed_preset[:,:,0]<200)

        # refined_binmask = refiner.refine(deformed[:,:,:3].astype('uint8'), deformed_mask.astype('uint8')*255, fast=False, L=900)

        # place preset shape over inpainted image
        img_base_np = np.array(img_base)
        # cond_image = img_base_np.copy().astype("float32")

        cond_mask = inpainting_mask[x_min:x_max, y_min:y_max]
        cond_mask = cv2.resize(cond_mask, (1024,1024), cv2.INTER_NEAREST)
        label_id_cropped = label_id.copy()
        label_id_cropped[cond_mask==255] = 6

        #smoth deformed mask boundaries
        blur_kernel = (53,53)
        # deformed_mask = cv2.GaussianBlur(deformed_mask.astype('uint8')*255, blur_kernel, 0)
        deformed_mask = deformed_mask>0


        # prepare conditioning image
        label_id_cropped[deformed_mask] = 3
        label_id_cropped[pupil] = 4
        label_id_cropped[empty] = 6

        cond_image = semantics.labels_to_colors(label_id_cropped)
        cond_image[cond_image.sum(2)==0] = [255,255,255]
        # cond_image[deformed_mask] = deformed[:, :, :3][deformed_mask]
        cond_image = cond_image.astype("uint8")
        
        # stylization
        # target_prompt = f"zoomed in face of a cartoon character {preset_prompts[preset_idx]}"
        target_prompt = f"zoomed in face of an animated handrawn character"

        ref_img = img_cropped.copy()

        # generated = np.zeros_like(ref_img)
        generated, conditioning = control_stylization.generate(
            ref_img, cond_image, ref_prompt, style_prompt, target_prompt
        )
        generated = np.array(generated)
        # blur to match resolution of original image
        generated = cv2.blur(generated, (5, 5))
        
        # include every part in final binary mask
        deformed_mask = (deformed_mask>0) | (pupil>0)
        deformed_mask_im = None
        predict_generated_mask = False
        pred_segmap = None
        if predict_generated_mask:
            pred_segmap, pred_labels = sam_face_model.predict(generated, face_region_cropped)
            deformed_mask = (pred_labels==2) | (pred_labels==7) | (pred_labels==10)
            deformed_mask_im = refiner.refine(generated, deformed_mask.astype('uint8')*255, fast=False, L=900)
        else:
            deformed_mask_im = cv2.dilate(deformed_mask.astype('uint8')*255, (23,23))
            deformed_mask_im = refiner.refine(generated, deformed_mask_im, fast=False, L=900)
        deformed_mask_im = np.repeat(deformed_mask_im[..., np.newaxis], 3, axis=2)
        deformed_mask_im = cv2.blur(deformed_mask_im, (51, 51))
        deformed_mask_im = deformed_mask_im.astype("float32")/255
        final_image = img_base_np * (1 - deformed_mask_im) + generated * deformed_mask_im
        final_image = final_image.astype("uint8")
        asset_image = np.concatenate([generated,255*deformed_mask_im[:,:,0:1]],-1).astype('uint8')

        # composite on full image
        final_image_resized = cv2.resize(final_image, (w_crop, h_crop))
        asset_image_resized = cv2.resize(asset_image, (w_crop, h_crop))
        img_composited = img_full.copy()
        img_composited[x_min:x_max, y_min:y_max] = final_image_resized
        asset_image_final = np.zeros((img_full.shape[0],img_full.shape[1],4)).astype('uint8')
        asset_image_final[x_min:x_max, y_min:y_max] = asset_image_resized


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

        img_face = img_full[face_x_min:face_x_max, face_y_min:face_y_max]
        img_composited_face = img_composited[face_x_min:face_x_max, face_y_min:face_y_max]
        asset_image_face = asset_image_final[face_x_min:face_x_max, face_y_min:face_y_max]
        img_face = cv2.resize(img_face, (1024, 1024))
        img_composited_face = cv2.resize(img_composited_face, (1024, 1024))
        asset_image_face = cv2.resize(asset_image_face, (1024, 1024))
        

        # plot images
        TITLE_SIZE = 35
        fig, ax = plt.subplots(1,5, figsize=(50,10))
        ax[0].imshow(img_face)
        ax[0].set_title("Original Image", fontsize=TITLE_SIZE)
        ax[0].axis('off')
        ax[1].imshow(ref_img)
        ax[1].set_title("Reference Style Image (Cropped)", fontsize=TITLE_SIZE)
        ax[1].axis('off')
        ax[2].imshow(cond_image)
        ax[2].set_title("Modified Segmap (Mouth)", fontsize=TITLE_SIZE)
        ax[2].axis('off')
        ax[3].imshow(generated)
        ax[3].set_title("Generated", fontsize=TITLE_SIZE)
        ax[3].axis('off')
        # ax[4].imshow(pred_segmap)
        # ax[4].set_title("New Segmap", fontsize=TITLE_SIZE)
        # ax[4].axis('off')
        ax[4].imshow(img_composited_face)
        ax[4].set_title("Composited", fontsize=TITLE_SIZE)
        ax[4].axis('off')
        plt.savefig( join(output_dir,f'{label_name.split("_")[0]}_{preset_class}_{shape_id}_plot.png') )
        plt.close()
        cv2.imwrite(join(output_dir,f'{label_name.split("_")[0]}_{preset_class}_{shape_id}_face.png'), cv2.cvtColor(img_composited_face, cv2.COLOR_RGB2BGR))
        cv2.imwrite(join(output_dir,f'{label_name.split("_")[0]}_{preset_class}_{shape_id}_asset.png'), cv2.cvtColor(asset_image_face, cv2.COLOR_RGBA2BGRA))

        # save images
        # Image.fromarray(final_image).save(
        #     join(
        #         output_dir,
        #         f'{label_name.split("_")[0]}_{preset_class}_pose{mouth_pose}_{shape_id}.png',
        #     )
        # )
        # Image.fromarray(condgenerated_adjusted_image).save(
        #     join(
        #         output_dir,
        #         f'{label_name.split("_")[0]}_{preset_class}_pose{mouth_pose}_{shape_id}_gen.png',
        #     )
        # )


    # invalidate the reference latent
    control_stylization.reference_latent = None
    control_stylization.inversion_callback = None
