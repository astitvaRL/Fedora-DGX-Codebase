import os
import cv2
import numpy as np
from tqdm import tqdm

join = os.path.join

idx=30

presets_path = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/PresetGeneration/OUTPUT/output_mouth_talk_PRESET'
subdirs = sorted(os.listdir(presets_path))[idx:]
for subdir in subdirs:
    subdir_path = join(presets_path,subdir)
    presets = sorted(os.listdir(subdir_path))
    video=cv2.VideoWriter(f'{idx}.mp4',cv2.VideoWriter_fourcc(*'DIVX'),30,(1024,1024))
    for i in tqdm(range(60)):
        for presetname  in presets:
            if presetname.endswith('_face.png'):
                preset_path = join(subdir_path, presetname)
                preset_im = cv2.imread(preset_path)
                for i in range(5):
                    video.write(preset_im)
    video.release()
    idx+=1
