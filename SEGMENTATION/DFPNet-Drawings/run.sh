#!/bin/bash
uname -a
#date
#env
date
CS_PATH='./Cartoon_sketches/Dog/'
LR=1e-3
WD=5e-4
#10 max for annnV2
BS=12
# GPU_IDS=1,2,3
GPU_IDS=0,1,2,3,4,5,6,7
RESTORE_FROM='./resnet101-imagenet.pth'
INPUT_SIZE='384,384'  
SNAPSHOT_DIR='./snapshots'
DATASET='train'
NUM_CLASSES=8 
START_EPOCH=1
EPOCHS=500

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python train_Uresnet_v3_ASPP_HD.py --data-dir ${CS_PATH} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --start-epoch ${START_EPOCH} \
       --epochs ${EPOCHS}

#python evaluate.py
