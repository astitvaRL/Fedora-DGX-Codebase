import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
# from skimage import io
import imageio as io
from tqdm import tqdm
import json


join = os.path.join

def composite(base, mouth, eyes, use_default_mouth=False, use_default_eyes=False):
    mask_mouth = mouth[:,:,3]/255
    mask_eyes = eyes[:,:,3]/255
    mask = mask_mouth + mask_eyes
    if use_default_eyes:
      mask = mask_mouth
    if use_default_mouth:
        mask = mask_eyes
    mask_im = np.repeat(mask[..., np.newaxis], 3, axis=2)
    mask_mouth_im = np.repeat(mask_mouth[..., np.newaxis], 3, axis=2)
    mask_eyes_im = np.repeat(mask_eyes[..., np.newaxis], 3, axis=2)
    composited = base*(1-mask_im) 
    if not use_default_eyes:
        composited += mask_eyes_im*eyes[:,:,:3]
    if not use_default_mouth:
        composited += mask_mouth_im*mouth[:,:,:3]
    return composited.astype('uint8')

if __name__ == '__main__':

    # set paths
    files_root = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/PresetGeneration/OUTPUT/lipsync_drawings_v2_ep60/'

    OUT_DIR = './PLOTS/COMPOSITED_v3/'
    os.makedirs(OUT_DIR, exist_ok=True)

    mouth_preset_ids = [3, 13, 8, 11, 19]
    eyes_preset_ids = [1, 3, 8, 7, 5]

    mouth_root = join(files_root,'mouth')
    eyes_root = join(files_root,'eyes')

    files = sorted(os.listdir(mouth_root))
    
    for filename in tqdm(files):
        try:
        # if True:
            input_im = cv2.imread(join(mouth_root, f'{filename}/metadata/image_face.png'))
            input_im = cv2.resize(input_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
            input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)

            base_im = cv2.imread(join(eyes_root, f'{filename}/metadata/inpainted_face_eyes_mouth.png'))
            base_im = cv2.resize(base_im, (1024,1024), interpolation=cv2.INTER_LINEAR)
            base_im = cv2.cvtColor(base_im, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(1,len(mouth_preset_ids)+1, figsize=((len(mouth_preset_ids)+1)*10,10))
            fig.tight_layout()
            ax[0].imshow(input_im)
            ax[0].axis('off')

            for idx in range(len(mouth_preset_ids)):
                m_id = mouth_preset_ids[idx]
                mouth_asset = cv2.imread(join(mouth_root, f'{filename}/assets/mouth_{m_id}_face.png'),-1)
                mouth_asset = cv2.resize(mouth_asset, (1024,1024), interpolation=cv2.INTER_LINEAR)
                mouth_asset = cv2.cvtColor(mouth_asset, cv2.COLOR_BGRA2RGBA)
                e_id = eyes_preset_ids[idx]
                eyes_asset = cv2.imread(join(eyes_root, f'{filename}/assets/eyes_{e_id}_face.png'),-1)
                eyes_asset = cv2.resize(eyes_asset, (1024,1024), interpolation=cv2.INTER_LINEAR)
                eyes_asset = cv2.cvtColor(eyes_asset, cv2.COLOR_BGRA2RGBA)
                comp_im = composite(base_im, mouth_asset, eyes_asset)
                ax[idx+1].imshow(comp_im)
                ax[idx+1].axis('off')

            plt.savefig(f'{OUT_DIR}/{filename}_plot.png')
            plt.close()

        except:
            print("Missing Data!")
            plt.close()
