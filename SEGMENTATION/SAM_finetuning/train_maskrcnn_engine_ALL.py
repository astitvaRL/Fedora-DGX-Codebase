import sys
sys.path.append('./')
sys.path.append('../')
from utils.detection.engine import train_one_epoch, evaluate

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import imageio as io
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision
from torchvision import transforms
import monai
import json
from monai.networks import one_hot
import kornia

from segment_anything_parallel import SamPredictor, sam_model_registry
from segment_anything_parallel.utils.transforms import ResizeLongestSide
from utils.mask_rcnn import custom_maskrcnn

from utils.dataset_maskrcnn import DrawingsDetectionDataset
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationAll
from utils.augment import RandomAug

join = os.path.join

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    torch.manual_seed(999)
    np.random.seed(999)
    torch.multiprocessing.set_start_method('spawn')

    # set paths
    data_root = '/mnt/users_scratch/astitva/DATA/'
    labels_definition_file_path = 'label_definition.json'
    ckpt_dir = './checkpoints'
    sam_original_ckpt_path = join(ckpt_dir,'sam_original/sam_vit_b_01ec64.pth')
    image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    label_id_dir_name = 'AD_SegMaps/labels_16k' 
    train_input_visualization_dir = './TMP/TRAIN_INPUT_VIS'
    cache_dir = join(data_root, 'dataset_caches/cache_dummy_ALL_REAL16k')
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = join(cache_dir, 'train_cache.pt')
    test_cache_path = join(cache_dir, 'test_cache.pt')
    task_name = '16k_MaskRCNN_ALL_CLASSES' # finetuned checkpoint will be saved here
    model_save_path = join(ckpt_dir, task_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(join(model_save_path, 'train_seg_vis'), exist_ok=True)
    os.makedirs(join(model_save_path, 'eval'), exist_ok=True)
    os.makedirs(join(model_save_path, 'all_ckpts'), exist_ok=True)
    os.makedirs(join(data_root, label_id_dir_name), exist_ok=True)
    os.makedirs(train_input_visualization_dir, exist_ok=True)
    
    # training choice
    cache_available = False # save dataset cache after first epoch
    resume_training = False
    visualize_train_input = False
    ignore_background = False
    bbox_given = False
    bg_mask_given = False
    
    if visualize_train_input:
        os.makedirs(train_input_visualization_dir, exist_ok=True)

     # load original SAM cackpoint for finetuning, or load an existing checkpoint for further training
    init_checkpoint = join(sam_original_ckpt_path)
    epoch_start = 0 # dont change this, change the one below
    if resume_training:
        epoch_start = 0 # change this
        resume_ckpt = join(ckpt_dir, 'ANIMSEG_E2E_NoBinmask_Coarse_REAL7k/model_eval_best.pth')
        init_checkpoint = resume_ckpt

    device = 'cuda:0'
    device_ids = [i for i in range(torch.cuda.device_count())]

    # semantic segmentation definition
    num_classes = 27
    semantics = SemanticSegmentationAll(labels_definition_file_path, num_classes=num_classes)

    # prepare sam model for preprocessing
    # sam_model_type = 'vit_b'
    # sam_model = sam_model_registry[sam_model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)
    # sam_model.image_encoder.to(device)
    # sam_model.prompt_encoder.to(device)
    # sam_model.mask_decoder.to(device)
    # sam_model.prompt_encoder.parallel_training = True
    # image_encoder = torch.nn.DataParallel(sam_model.image_encoder, device_ids=device_ids)
    # prompt_encoder = torch.nn.DataParallel(sam_model.prompt_encoder, device_ids=device_ids)
    # mask_decoder = torch.nn.DataParallel(sam_model.mask_decoder, device_ids=device_ids)

    # prepare mask_rcnn model
    mask_rcnn = custom_maskrcnn(num_classes=num_classes)
    mask_rcnn = mask_rcnn.to(device)
    # model = torch.nn.DataParallel(mask_rcnn, device_ids=device_ids)

    # create dataset
    train_dataset = DrawingsDetectionDataset(labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, label_id_dir_name = label_id_dir_name, mode='train', num_test_samples=2000)
    test_dataset = DrawingsDetectionDataset(labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, label_id_dir_name = label_id_dir_name, mode='test', num_test_samples=2000)

    # set semantic definition
    train_dataset.num_classes = num_classes
    test_dataset.num_classes = num_classes
    train_dataset.semantics = semantics
    test_dataset.semantics = semantics

    cache_start = time.time()
    if cache_available:
        print("Preloading Caches...")
        train_dataset.cache_available = True
        train_dataset.load_cache(train_cache_path)
        test_dataset.cache_available = True
        test_dataset.load_cache(test_cache_path)
    else: # initialize empty cache and populate during first epoch, save after first epoch
        print("NEW CACHES will be saved here -->", cache_dir)
        train_dataset.init_cache()
        test_dataset.init_cache()
    cache_end = time.time()
    
    print(f"CACHES ARE READY! Took {cache_end-cache_start} seconds ---", train_dataset.cache.shape, test_dataset.cache.shape)

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=True, collate_fn=collate_fn)

    # training config
    num_epochs = 1000
    save_frequency = 1
    eval_frequency = 2
    train_loss_log = []
    eval_loss_log = []
    best_loss = 1e10
    best_eval_loss = 1e10


# construct an optimizer
    params = [p for p in mask_rcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # Set up the losses
    dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    focal_loss = monai.losses.FocalLoss(reduction='mean', gamma=2.0)

    # define differentiable non-learnable upsampling layer
    # upsample = torch.nn.Upsample(scale_factor=4, mode='nearest')

    # augmentations
    input_size = (1024, 1024)
    crop_size = (800, 800)
    randomaug = RandomAug(target_size=input_size, crop_size=crop_size)

    # start training
    # mask_rcnn.train()
    for epoch in range(epoch_start, num_epochs):
        epoch_loss = 0
        # TRAINING
        for step, (images, targets) in enumerate(tqdm(train_dataloader, "Training")):
            
            # image_data = image_data_cpu.to(device)
            # gt = gt.to(device)

            # augmentations
            # image_data, gt, bg_mask, _ = randomaug.apply_augmentation(image_data, gt, bg_mask)
            # image_data = randomaug.apply_color_jitter(image_data, gt)

            # resize gt and bg_mask
            # gt = F.resize(gt, 1024, torchvision.transforms.InterpolationMode.NEAREST) # prediction will be umsampled to 1024x1024
            # bg_mask = F.resize(bg_mask, 256, torchvision.transforms.InterpolationMode.NEAREST) # decoder takes mask size 256x256

            ################################################################################################
            ######### ------- plt visualizations after resizing  (last sample from batch) -------- #########
            ################################################################################################

            # if visualize_train_input and epoch==0:
            #     for batch_idx in range(image_data.shape[0]):
            #         original_image_data_vis = image_data_cpu.cpu().numpy()[batch_idx] # last sample from batch
            #         original_image_data_vis = (original_image_data_vis + 1.0)/2.0
            #         original_image_data_vis = np.transpose(original_image_data_vis,(1,2,0))
            #         image_data_vis = image_data.cpu().numpy()[batch_idx] # last sample from batch
            #         image_data_vis = (image_data_vis + 1.0)/2.0
            #         image_data_vis = np.transpose(image_data_vis,(1,2,0))
            #         gt_vis = gt.cpu().numpy()[batch_idx] # last sample from batch
            #         gt_vis = semantics.labels_to_colors(cv2.resize(gt_vis[0], (1024,1024), interpolation=cv2.INTER_NEAREST))
            #         bg_mask_vis = bg_mask.cpu().numpy()[batch_idx] # last sample from batch
            #         bg_mask_vis = cv2.resize(bg_mask_vis[0], (1024,1024), interpolation=cv2.INTER_NEAREST)
            #         fig, ax = plt.subplots(1,4,figsize=(40,10))
            #         TITLE_SIZE = 30
            #         ax[0].imshow(original_image_data_vis)
            #         ax[0].set_title('Original Image', fontsize=TITLE_SIZE)
            #         ax[1].imshow(image_data_vis)
            #         ax[1].set_title('Augmented Image', fontsize=TITLE_SIZE)
            #         ax[2].imshow(bg_mask_vis, cmap='gray')
            #         ax[2].set_title('Background Mask', fontsize=TITLE_SIZE)
            #         ax[3].imshow(gt_vis)
            #         ax[3].set_title('Semantic Map', fontsize=TITLE_SIZE)
            #         tmp_save_path = join(train_input_visualization_dir, f"{step}_{batch_idx}.png")
            #         plt.savefig(tmp_save_path)
            #         plt.close()

            ################################################################################################
            ######### ----------------------------------------------------------------------------- ########
            ################################################################################################


            # convert gt to one hot encoding
            # gt = gt.long().squeeze(1)
            # gt = torch.nn.functional.one_hot(gt,num_classes)
            # B,_, H, W = gt.shape
            # gt = torch.permute(gt,(0,3,1,2))

            # # overwrite bbox to a fixed one
            # bbox[:,0] = 0
            # bbox[:,1] = 0
            # bbox[:,2] = 256
            # bbox[:,3] = 256
            # bbox = bbox.to(device)
            
            # image_data = []
            # target_data = []
            # for data in zipped_data:
            #     image_data.append(data[0][0])
            #     target_data.append(data[0][1])


            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            train_one_epoch(mask_rcnn, optimizer, train_dataloader, device, epoch, print_freq=10)
            # with torch.cuda.amp.autocast(enabled=True):
            # loss_dict = mask_rcnn(images, targets)
