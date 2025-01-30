#!/bin/bash

CS_PATH='./Cartoon_sketches/Dog'
# CS_PATH='/mnt/users_scratch/astitva/DATA/dogs_OOD'
# CS_PATH='./dataset/LIP/TrainVal'
BS=12
GPU_IDS='0'
INPUT_SIZE='384,384'
# SNAPSHOT_FROM='./snapshots_drawings/DFPnet_epoch_495.pth'
SNAPSHOT_FROM='./snapshots/DFPnet_epoch_495.pth'
# SNAPSHOT_FROM='./snapshots/'
# SNAPSHOT_FROM='/home/jeromewan/SJTU_Thesis/Non_local_CE2P/Trained_Models/1000_images/UResnet_v3_ASPP/HD/final_concat_conv3x3_afetr_upsampling/LIP_epoch_495.pth'
#DATASET='./dataset/LIP/TrainVal_images/TrainVal_images'
DATASET='val'
NUM_CLASSES=8
# NUM_CLASSES=27
SAVE_PATH_DIR='./PREDICTIONS/dogs_val_named/'

python evaluate_save_preds.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-path ${SAVE_PATH_DIR}