import yaml
import os
import shutil
from pathlib import Path
import sys
import cv2
import numpy as np
from server import create_annotation_level_data
import natsort
from tqdm import tqdm

def resize_max_dim(image, max_dim):
    height, width = image.shape[:2]

    # Calculate the scaling factor
    if height > width:
        scale = max_dim / height
    else:
        scale = max_dim / width

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


if __name__ == '__main__':
    SEG_DATA_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\labels_resized\\"
    IMG_DATA_DIR = "D:\\DATA\\Amateur Drawing Semantic Segmentations\\20240716-1034 (1)\\drawings_resized\\"
    TMP_FN = "tmp\\tmp.png"
    SAVE_DIR = "labels_animated\\"

    os.makedirs(SAVE_DIR, exist_ok=True)

    FAIL_COUNT = 0

    files = natsort.natsorted(os.listdir(SEG_DATA_DIR))
    for fn in tqdm(files):
        try:

            # get the input original image
            image_fn = os.path.join(IMG_DATA_DIR, f"{fn.split('_')[0]}.png")
            image = cv2.imread(image_fn)

            # get the semantic segmentatation mask
            sem_seg_mask_fn =  os.path.join(SEG_DATA_DIR, f"{fn}")
            sem_seg_mask = cv2.imread(sem_seg_mask_fn)

            # ensure both are of same dimensions:
            assert image.shape[:2] == sem_seg_mask.shape[:2], "image and sem mask don't have same dimensions"

            # resize so max dim is 512
            image = resize_max_dim(image, 512)
            sem_seg_mask = resize_max_dim(sem_seg_mask, 512)

            # compute the foreground background mask
            mask_r = sem_seg_mask[:, :, 0] != 0
            mask_g = sem_seg_mask[:, :, 1] != 0
            mask_b = sem_seg_mask[:, :, 2] != 0
            mask = mask_r | mask_g | mask_b

            # make sem_seg_mask transparent where no character
            sem_seg_mask_rgba = np.zeros([sem_seg_mask.shape[0], sem_seg_mask.shape[1], 4])
            sem_seg_mask_rgba[:, :, :3] = sem_seg_mask
            sem_seg_mask_rgba[:, :, 3][mask] = 255
            sem_seg_mask = sem_seg_mask_rgba

            # write out tmp fb mask, sem sem mask, and image
            save_current_dir = os.path.join(SAVE_DIR, f"{fn[:-4]}")
            os.makedirs(save_current_dir, exist_ok=True)
            fb_mask_tmp_fn = os.path.join(save_current_dir, 'fb_mask.png')
            cv2.imwrite(fb_mask_tmp_fn, 255 * mask.astype(np.uint8))

            sem_seg_mask_tmp_fn = os.path.join(save_current_dir, 'sem_seg_mask.png')
            cv2.imwrite(sem_seg_mask_tmp_fn, sem_seg_mask)

            image_tmp_fn = os.path.join(save_current_dir, 'image.png')
            cv2.imwrite(image_tmp_fn, image)

            # # create the annotations and rig, get mvc
            mvc_fn = create_annotation_level_data(image_tmp_fn, fb_mask_tmp_fn, str(Path(image_fn).stem))

            # swap out the original texture with the semantic segmentation mask
            src = sem_seg_mask_tmp_fn
            dst1 = Path(mvc_fn).parent / 'rig' / 'left' / 'texture_front.png'
            dst2 = Path(mvc_fn).parent / 'rig' / 'left' / 'texture_back.png'
            dst3 = Path(mvc_fn).parent / 'rig' / 'right' / 'texture_front.png'
            dst4 = Path(mvc_fn).parent / 'rig' / 'right' / 'texture_back.png'

            shutil.copy(src, dst1)
            shutil.copy(src, dst2)
            shutil.copy(src, dst3)
            shutil.copy(src, dst4)

            # modify yaml as needed
            with open(Path(mvc_fn) / 'mvc.yaml', 'r') as f:
                mvc = yaml.load(f, Loader=yaml.SafeLoader)
                mvc['controller'] = {}
                mvc['controller']['MODE'] = 'image_render'
                mvc['controller']['OUTPUT_IMAGE_DIR'] = str(Path(Path(sem_seg_mask_fn).parent/'images'))  # only used if mode is 'video_render'

                mvc['view']['DRAW_BVH'] = False
                mvc['view']['CAMERA_POS'] = [1.0654328, 0.6566088, 1.398207]

                mvc['scene']['ANIMATED_CHARACTERS'][0]['motion_cfg'] = 'dev_configs/demo/motion_sem_seg_aug.yaml'

            with open(Path(mvc_fn) / 'mvc.yaml', 'w') as f:
                yaml.dump(mvc, f)

        except:
            FAIL_COUNT += 1
            print("Failed!")
            continue

    print(f"Failed {FAIL_COUNT} times")
