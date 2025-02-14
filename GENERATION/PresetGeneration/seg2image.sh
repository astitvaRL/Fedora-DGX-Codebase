PRESET_SCRIPTS_ROOT='/mnt/users_scratch/hjessmith/WORKSPACE/Fedora-DGX-Codebase/GENERATION/PresetGeneration/'
SEAN_ROOT='/mnt/users_scratch/hjessmith/WORKSPACE/Fedora-DGX-Codebase/GENERATION/Seg2Image/SEAN/'
SEAN_EPOCH='50'

cd $SEAN_ROOT

# estimate style code
python test.py --name seg2drawings --load_size 1024 --crop_size 1024 --dataset_mode custom --label_dir $LABEL_DIR --image_dir $IMAGE_DIR  --label_nc 11 --no_instance --batchSize 1  --gpu_ids 0 --which_epoch $SEAN_EPOCH

# generate with deformed segmaps
python test_generate_assets.py --name seg2drawings --load_size 1024 --crop_size 1024 --dataset_mode custom --label_dir $PRESETS_LABEL_DIR --image_dir $PRESETS_IMAGE_DIR --label_nc 11 --no_instance --batchSize 1  --gpu_ids 0 --which_epoc $SEAN_EPOCH

# remove current style code
rm -rf /mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/GENERATION/Seg2Image/SEAN/styles_test/style_codes/original.png

cd $PRESET_SCRIPTS_ROOT
