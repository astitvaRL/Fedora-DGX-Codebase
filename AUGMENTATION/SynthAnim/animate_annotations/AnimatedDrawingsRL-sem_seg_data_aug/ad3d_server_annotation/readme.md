Prereqs:
- Create new conda environment `conda create --name sem_sem_data_aug python=3.8.18` 
- Install Animated Drawings `cd ..; pip install -e .`
- Install additional requirements from requirements.txt in this dir. You may get package version errors, but you can ignore them `cd ad3d_server_annotation; pip install -r requirements.txt`
- Follow instructions in pose_estimation/readme_weights.md to get weights for pose estimation.


Data Augmentation:
- cwd should be ad3d_server_annation
- generate character rig using semantic segmentation mask: `python generate_semantic_segmentation_data_augmentations.py {path to image} {path to semantic segmentation}`
- run Animated Drawings to generate the images: `python ../animated_drawings/render.py {path to image parent}/{image stem}/rig/mvc.yaml`