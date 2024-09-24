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

from segment_anything_parallel import SamPredictor, sam_model_registry
from segment_anything_parallel.utils.transforms import ResizeLongestSide

from utils.dataset import DrawingsDataset
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationBinary
from utils.augment import RandomAug

join = os.path.join


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
    label_id_dir_name = 'AD_SegMaps/labels_7k_1024' 
    cache_dir = join(data_root, 'dataset_caches/cache_Coarse_REAL7k')
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = join(cache_dir, 'train_cache.pt')
    test_cache_path = join(cache_dir, 'test_cache.pt')
    task_name = 'ANIMSEG_DecoderOnly_Binary_REAL7k' # finetuned checkpoint will be saved here
    model_save_path = join(ckpt_dir, task_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(join(model_save_path, 'train_seg_vis'), exist_ok=True)
    os.makedirs(join(model_save_path, 'eval'), exist_ok=True)
    os.makedirs(join(model_save_path, 'all_ckpts'), exist_ok=True)
    os.makedirs(join(data_root, label_id_dir_name), exist_ok=True)
    train_input_visualization_dir = join(data_root, 'TMP_INPUT_VIS')
    
    # training choice
    cache_available = True # save dataset cache after first epoch
    resize_labels = False # False if already resized
    resume_training = True
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
        resume_ckpt = join(ckpt_dir, 'ANIMSEG_EncDec_Coarse_REAL7k/model_eval_best.pth')
        init_checkpoint = resume_ckpt

    device = 'cuda:0'
    device_ids = [i for i in range(torch.cuda.device_count())]

    # semantic segmentation definition
    num_classes = 2
    semantics = SemanticSegmentationBinary(labels_definition_file_path, num_classes=num_classes)

    # prepare SAM model
    model_type = 'vit_b'
    sam_model = sam_model_registry[model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)
    sam_model.image_encoder.to(device)
    sam_model.prompt_encoder.to(device)
    sam_model.mask_decoder.to(device)
    sam_model.prompt_encoder.parallel_training = True
    image_encoder = torch.nn.DataParallel(sam_model.image_encoder, device_ids=device_ids)
    prompt_encoder = torch.nn.DataParallel(sam_model.prompt_encoder, device_ids=device_ids)
    mask_decoder = torch.nn.DataParallel(sam_model.mask_decoder, device_ids=device_ids)

    if resize_labels:
        load_dir_name = label_id_dir_name[:-5] # remove '_1024'
        labels = sorted(os.listdir(join(data_root, load_dir_name)))
        print('Resizing Labels...')
        for label_name in tqdm(labels):
            save_path = join(data_root, label_id_dir_name, label_name.split('.png')[0]+'_1024.png')
            if not os.path.exists(save_path):
                label = cv2.imread(join(data_root, load_dir_name, label_name))
                label = cv2.resize(label, (1024,1024), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(save_path, label)

    # create dataset
    train_dataset = DrawingsDataset(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, label_id_dir_name = label_id_dir_name, mode='train')
    test_dataset = DrawingsDataset(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, label_id_dir_name = label_id_dir_name, mode='test')

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
    train_dataloader = DataLoader(train_dataset, batch_size=160, num_workers=0, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=0, shuffle=False, drop_last=True)

    # training config
    num_epochs = 1000
    save_frequency = 1
    eval_frequency = 1
    train_loss_log = []
    eval_loss_log = []
    best_loss = 1e10
    best_eval_loss = 1e10

    # Freeze all layers of image encoder
    for param in image_encoder.parameters():
        param.requires_grad = False

    ## verify
    # for name, param in image_encoder.named_parameters():
    #     print(name, param.requires_grad)

    # Set up the optimizer
    optimizer = torch.optim.Adam(mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    # Set up the losses
    dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    focal_loss = monai.losses.FocalLoss(reduction='mean', gamma=2.0)

    # define differntiable non-learnable upsampling layer
    upsample = torch.nn.Upsample(scale_factor=4, mode='nearest')

    # augmentations
    input_size = (1024, 1024)
    crop_size = (800, 800)
    randomaug = RandomAug(target_size=input_size, crop_size=crop_size)

    # start training
    sam_model.train()
    for epoch in range(epoch_start, num_epochs):
        epoch_loss = 0
        # TRAINING
        for step, (image_data_cpu, gt, bg_mask, bbox) in enumerate(tqdm(train_dataloader, "Training")):
            
            image_data = image_data_cpu.to(device)
            gt = gt.to(device)
            bg_mask = bg_mask.to(device)

            # augmentations
            image_data, gt, bg_mask, _ = randomaug.apply_augmentation(image_data, gt, bg_mask)
            image_data = randomaug.apply_color_jitter(image_data, gt)

            # resize gt and bg_mask
            gt = F.resize(gt, 1024, torchvision.transforms.InterpolationMode.NEAREST) # prediction will be umsampled to 1024x1024
            bg_mask = F.resize(bg_mask, 256, torchvision.transforms.InterpolationMode.NEAREST) # decoder takes mask size 256x256

            ################################################################################################
            ######### ------- plt visualizations after resizing  (last sample from batch) -------- #########
            ################################################################################################

            if visualize_train_input and epoch==0:
                for batch_idx in range(image_data.shape[0]):
                    original_image_data_vis = image_data_cpu.cpu().numpy()[batch_idx] # last sample from batch
                    original_image_data_vis = (original_image_data_vis + 1.0)/2.0
                    original_image_data_vis = np.transpose(original_image_data_vis,(1,2,0))
                    image_data_vis = image_data.cpu().numpy()[batch_idx] # last sample from batch
                    image_data_vis = (image_data_vis + 1.0)/2.0
                    image_data_vis = np.transpose(image_data_vis,(1,2,0))
                    gt_vis = gt.cpu().numpy()[batch_idx] # last sample from batch
                    gt_vis = semantics.labels_to_colors(cv2.resize(gt_vis[0], (1024,1024), interpolation=cv2.INTER_NEAREST))
                    bg_mask_vis = bg_mask.cpu().numpy()[batch_idx] # last sample from batch
                    bg_mask_vis = cv2.resize(bg_mask_vis[0], (1024,1024), interpolation=cv2.INTER_NEAREST)
                    fig, ax = plt.subplots(1,4,figsize=(40,10))
                    TITLE_SIZE = 30
                    ax[0].imshow(original_image_data_vis)
                    ax[0].set_title('Original Image', fontsize=TITLE_SIZE)
                    ax[1].imshow(image_data_vis)
                    ax[1].set_title('Augmented Image', fontsize=TITLE_SIZE)
                    ax[2].imshow(bg_mask_vis, cmap='gray')
                    ax[2].set_title('Background Mask', fontsize=TITLE_SIZE)
                    ax[3].imshow(gt_vis)
                    ax[3].set_title('Semantic Map', fontsize=TITLE_SIZE)
                    tmp_save_path = join(train_input_visualization_dir, f"{step}_{batch_idx}.png")
                    plt.savefig(tmp_save_path)
                    plt.close()

            ################################################################################################
            ######### ----------------------------------------------------------------------------- ########
            ################################################################################################


            # convert gt to binary one hot encoding
            gt[gt>0] = 1
            gt = gt.long().squeeze(1)
            gt = torch.nn.functional.one_hot(gt,num_classes)
            B,_, H, W = gt.shape
            gt = torch.permute(gt,(0,3,1,2))

            # overwrite bbox to a fixed one
            bbox[:,0] = 0
            bbox[:,1] = 0
            bbox[:,2] = 256
            bbox[:,3] = 256
            bbox = bbox.to(device)
            
            # not computing gradients for image encoder and prompt encoder
            with torch.no_grad():           
                sparse_embeddings, dense_embeddings, image_pe = prompt_encoder(
                    points=None,
                    boxes=bbox[:, None, :] if bbox_given else None,
                    masks=bg_mask if bg_mask_given else None,
                )
                image_data = F.resize(image_data, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
                embedding = image_encoder(image_data)
                
            # computing gradients for mask decoder only
            mask_predictions, _ = mask_decoder(
                image_embeddings=embedding, # (B, 256, 64, 64)
                # image_pe=prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                image_pe=image_pe, # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
            )
            
            #upsample mask predictions
            mask_predictions = upsample(mask_predictions)

            # Optional, doesn't help much if mask is already being passed as prompt
            if ignore_background and bg_mask_given:  
                mask_predictions = mask_predictions*bg_mask
                gt = gt*bg_mask
                
            loss = 0.7*dice_loss(mask_predictions, gt) + 0.3*focal_loss(mask_predictions, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= (step+1)
        train_loss_log.append(epoch_loss)

        # save dataset cache after first epoch
        if not(cache_available) and epoch==0:
            train_dataset.save_cache(train_cache_path)
            train_dataset.cache_available = True
        
        # save the latest model checkpoint as required
        if epoch%save_frequency==0:
            print(f'TRAIN EPOCH: {epoch}, Loss: {epoch_loss}')
            # save the latest model checkpoint
            torch.save(sam_model.state_dict(), join(model_save_path, f'all_ckpts/model_{epoch}.pth'))
            labels_out = torch.argmax(torch.Tensor(mask_predictions[-1]), dim=0) # last sample from randomized current batch
            plt.imshow(semantics.labels_to_colors(labels_out.cpu().numpy().astype('uint8')))
            plt.savefig(join(model_save_path, f"train_seg_vis/{epoch}.png"))
            # save the best model checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(sam_model.state_dict(), join(model_save_path, 'model_best.pth'))
        
        # EVALUATION (on test set)
        if epoch%eval_frequency==0:    
            # reset metrics for latest epoch
            eval_loss = 0
            for step, (image_data_eval, gt, bg_mask, bbox) in enumerate(tqdm(test_dataloader,"EVAL")):
                eval_epoch_dir = join(model_save_path, f"eval/{epoch}")
                os.makedirs(eval_epoch_dir, exist_ok=True)
                # move input to device
                image_data_eval = image_data_eval.to(device)
                gt = gt.to(device)
                bg_mask = bg_mask.to(device)
                bbox = bbox.to(device)            
                # not computing any gradients during evaluation
                with torch.no_grad():
                    gt = F.resize(gt, 1024, torchvision.transforms.InterpolationMode.NEAREST)
                    gt[gt>0] = 1
                    gt = torch.nn.functional.one_hot(gt.squeeze(1),num_classes)
                    B,_, H, W = gt.shape
                    gt = torch.permute(gt,(0,3,1,2))
                    bg_mask = F.resize(bg_mask, 256, torchvision.transforms.InterpolationMode.NEAREST)
                    # resizing by a factor of 4 (1024-->256)
                    bbox = bbox//4 
                    # prompt encoding
                    sparse_embeddings, dense_embeddings, image_pe = prompt_encoder(
                        points=None,
                        boxes=bbox[:, None, :] if bbox_given else None,
                        masks=bg_mask if bg_mask_given else None,
                    )
                    # image embedding estimation
                    image_data_eval = F.resize(image_data_eval, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
                    embedding = image_encoder(image_data_eval)
                    # segmentation prediction
                    mask_predictions, _ = mask_decoder(
                        image_embeddings=embedding.to(device), # (B, 256, 64, 64)
                        image_pe=image_pe, # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                        multimask_output=True,
                    )
                    #upsample mask predictions
                    mask_predictions = upsample(mask_predictions)
                    if bg_mask_given: 
                        # excluding background from loss computation
                        bg_mask = F.resize(bg_mask, 1024, torchvision.transforms.InterpolationMode.NEAREST)
                        gt = gt*bg_mask
                        mask_predictions = mask_predictions*bg_mask
                    # compute eval loss
                    eval_loss += dice_loss(mask_predictions, gt.to(device)).item()
                    # visualizing last sample from every batch
                    labels_out = torch.argmax(torch.Tensor(mask_predictions[-1]), dim=0)  # last sample from batch
                    labels_out_vis = semantics.labels_to_colors(labels_out.cpu().numpy().astype('uint8'))
                    labels_out_vis = cv2.resize(labels_out_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                    gt_vis = torch.argmax(torch.Tensor(gt[-1]), dim=0)  # last sample from batch
                    gt_vis = semantics.labels_to_colors(gt_vis.cpu().numpy().astype('uint8'))
                    gt_vis = cv2.resize(gt_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                    bg_mask_vis = cv2.resize(bg_mask[-1][0].cpu().numpy().astype('uint8'), (1024,1024), interpolation=cv2.INTER_NEAREST)  # last sample from batch
                    image_data_vis = image_data_eval.cpu().numpy()[-1] # last sample from batch
                    image_data_vis = (image_data_vis - image_data_vis.min()) / (image_data_vis.max() - image_data_vis.min())
                    image_data_vis = np.transpose(image_data_vis,(1,2,0))
                    # plot eval results
                    TITLE_SIZE = 30
                    fig, ax = plt.subplots(1,3, figsize=(30,10))
                    ax[0].imshow(image_data_vis)
                    ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    # ax[1].imshow(bg_mask_vis, cmap='gray')
                    # ax[1].set_title("Mask", fontsize=TITLE_SIZE)
                    # ax[1].axis('off')
                    ax[1].imshow(labels_out_vis)
                    ax[1].set_title("Prediction", fontsize=TITLE_SIZE)
                    ax[1].axis('off')
                    ax[2].imshow(gt_vis)
                    ax[2].set_title("GT", fontsize=TITLE_SIZE)
                    ax[2].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{step}.png")
                    plt.close()
            # logging eval loss and metrics
            eval_loss /= (step+1)
            print(f'EVAL EPOCH: {epoch}, Loss: {eval_loss}')
            eval_loss_log.append(eval_loss)
            np.save(join(model_save_path,f"eval_loss_log_latest.npy"),np.array(eval_loss_log))

            # save test cache after first validation epoch
            if not(cache_available) and epoch==0:
                test_dataset.save_cache(test_cache_path) 
                test_dataset.cache_available = True

            # save best eval model checkpoint
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(sam_model.state_dict(), join(model_save_path, 'model_eval_best.pth'))

        # logging train loss
        np.save(join(model_save_path,f"train_loss_log_latest.npy"),np.array(train_loss_log))
