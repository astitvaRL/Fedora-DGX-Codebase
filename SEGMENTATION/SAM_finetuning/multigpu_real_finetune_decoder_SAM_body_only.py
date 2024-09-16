import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
from skimage import io
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

from utils.dataset import Dataset_body, Dataset_body_real
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentation
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
    label_id_dir_name = 'AD_SegMaps/labels_2k_1024' 
    embed_dir_name = f"{image_dir_name}_embeddings" # precomputed image embeddings will be saved here if not saved already
    embedding_dir_path = join(data_root, embed_dir_name)
    cache_dir = join(data_root, 'cache_real_2k')
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = join(cache_dir, 'train_cache.pt')
    test_cache_path = join(cache_dir, 'test_cache.pt')
    task_name = 'multigpu_vanilla_randomaug_real_2k' # finetuned checkpoint will be saved here
    model_save_path = join(ckpt_dir, task_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(join(model_save_path, 'train_seg_vis'), exist_ok=True)
    os.makedirs(join(model_save_path, 'eval'), exist_ok=True)
    
    # training choice
    cache_available = False # save dataset cache after first epoch
    precompute_embeddings = False # False if already precomputed and saved
    resize_labels = False # False if already resized
    resume_training = False
    visualization_debug = True
    ignore_background = False
    bbox_given = False
    bg_mask_given = True
    random_flip = True
    # prepare SAM model
    model_type = 'vit_b'

     # load original SAM cackpoint for finetuning, or load an existing checkpoint for further training
    init_checkpoint = join(sam_original_ckpt_path)
    epoch_start = 0 # dont change this, change the one below
    if resume_training:
        epoch_start = 0 # change this
        resume_ckpt = join(ckpt_dir, 'multigpu_vanilla_randomaug_syn_17k/model_eval_best.pth')
        init_checkpoint = resume_ckpt

    device = 'cuda:0'
    device_ids = [i for i in range(torch.cuda.device_count())]
    num_classes = 18 
    sam_model = sam_model_registry[model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)
    sam_model.image_encoder.to(device)
    sam_model.prompt_encoder.to(device)
    sam_model.mask_decoder.to(device)
    sam_model.prompt_encoder.parallel_training = True
    image_encoder = torch.nn.DataParallel(sam_model.image_encoder, device_ids=device_ids)
    prompt_encoder = torch.nn.DataParallel(sam_model.prompt_encoder, device_ids=device_ids)
    mask_decoder = torch.nn.DataParallel(sam_model.mask_decoder, device_ids=device_ids)

    if resize_labels:
        labels = sorted(os.listdir(join(data_root, label_id_dir_name)))
        print('Resizing Labels...')
        for label_name in tqdm(labels):
            save_path = join(data_root, label_id_dir_name, label_name.split('.png')[0]+'_1024.png')
            if not os.path.exists(save_path):
                label = cv2.imread(join(data_root, label_id_dir_name, label_name))
                label = cv2.resize(label, (1024,1024), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(label, label)

    # precompute image embeddings using original SAM model
    if precompute_embeddings:
        os.makedirs(embedding_dir_path, exist_ok=True)
        print('Precomputing image embeddings...')
        names = sorted(os.listdir(join(data_root, image_dir_name)))
        for name in tqdm(names):
            image_data = io.imread(join(data_root, image_dir_name, name))
            if image_data.shape[-1]>3 and len(image_data.shape)==3:
                image_data = image_data[:,:,:3]
            if len(image_data.shape)==2:
                image_data = np.repeat(image_data[:,:,None], 3, axis=-1)
            sam_transform = ResizeLongestSide(image_encoder.img_size)
            resize_img = sam_transform.apply_image(image_data)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            assert input_image.shape == (1, 3, image_encoder.img_size, image_encoder.img_size), 'input image should be resized to 1024*1024'
            # precompute and save the image embedding
            with torch.no_grad():
                embedding = image_encoder(input_image)
                np.save(join(embedding_dir_path, name.split('.png')[0]+'.npy'), embedding.cpu().numpy()[0])
        print('Image embeddings saved at -->', embedding_dir_path)

    # create dataset
    train_dataset = Dataset_body_real(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, img_embed_dir_name = embed_dir_name, label_id_dir_name = label_id_dir_name, mode='train')
    test_dataset = Dataset_body_real(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, img_embed_dir_name = embed_dir_name, label_id_dir_name = label_id_dir_name, mode='test')
    
    cache_start = time.time()
    if cache_available:
        print("Preloading Caches...")
        train_dataset.cache_available = True
        train_dataset.load_cache(train_cache_path)
        test_dataset.cache_available = True
        test_dataset.load_cache(test_cache_path)
    else: # initialize empty cache and populate during first epoch, save after first epoch
        print("NEW CACHES WILL BE CREATED!")
        train_dataset.init_cache()
        test_dataset.init_cache()
    cache_end = time.time()
    
    print(f"CACHES ARE READY! Took {cache_end-cache_start} seconds ---", train_dataset.cache.shape, test_dataset.cache.shape)

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=False, drop_last=True)

    # training config
    num_epochs = 1000
    save_frequency = 1
    eval_frequency = 1
    train_loss_log = []
    eval_loss_log = []
    best_loss = 1e10
    best_eval_loss = 1e10
    semantics = SemanticSegmentation(labels_definition_file_path, num_classes=num_classes)

    # Set up the optimizer, losses, hyperparameters
    optimizer = torch.optim.Adam(mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    focal_loss = monai.losses.FocalLoss(reduction='mean', gamma=2.0)

   # Freeze all layers of image encoder
    for param in image_encoder.parameters():
        param.requires_grad = False

    ## verify
    # for name, param in image_encoder.named_parameters():
    #     print(name, param.requires_grad)

    # augmentations
    crop_size = (800, 800)
    input_size = (1024, 1024)

    # start training
    sam_model.train()
    for epoch in range(epoch_start, num_epochs):
        epoch_loss = 0
        # TRAINING
        for step, (image_data, gt, bg_mask, bbox) in enumerate(tqdm(train_dataloader, "Training")):
            # not loading precomputed embeddings during training
            image_data = image_data.to(device)
            gt = gt.to(device)
            bg_mask = bg_mask.to(device)
            with torch.no_grad():
                crop_probability = np.random.uniform(0,1)
                if crop_probability<0.3:
                    # random crop
                    crop = torchvision.transforms.RandomCrop(crop_size)
                    input_all = torch.cat([image_data, gt, bg_mask], axis=1)
                    input_all = crop(input_all)
                    image_data = input_all[:,:3,:,:] # encoder takes image size 1024x1024
                    gt = input_all[:,3:4,:,:]
                    bg_mask = input_all[:,4:,:,:]
                    # for bounding box
                    bbox[:,0] = 0
                    bbox[:,1] = 0
                    bbox[:,2] = 256
                    bbox[:,3] = 256
                
                if random_flip:
                    fliph = torchvision.transforms.RandomHorizontalFlip(p=0.4)                
                    flipv = torchvision.transforms.RandomVerticalFlip(p=0.4)                
                    input_all = torch.cat([image_data, gt, bg_mask], axis=1)
                    input_all = fliph(input_all)
                    # input_all = flipv(input_all)
                    image_data = input_all[:,:3,:,:] # encoder takes image size 1024x1024
                    gt = input_all[:,3:4,:,:]
                    bg_mask = input_all[:,4:,:,:]
                    # for bounding box
                    bbox[:,0] = 0
                    bbox[:,1] = 0
                    bbox[:,2] = 256
                    bbox[:,3] = 256
                
                if np.random.uniform(0,1)<0.5: # add noise to bounding box
                    # resizing bbox by a factor of 4 (1024-->256)
                    bbox = bbox//4 
                    # add random noise to bounding box within 256x256 range
                    bbox[:,0] = torch.clamp(bbox[:,0] + torch.randint(-20,20,(bbox.shape[0],)), 0, 256)
                    bbox[:,1] = torch.clamp(bbox[:,1] + torch.randint(-20,20,(bbox.shape[0],)), 0, 256)
                    bbox[:,2] = torch.clamp(bbox[:,2] + torch.randint(-20,20,(bbox.shape[0],)), 0, 256)
                    bbox[:,3] = torch.clamp(bbox[:,3] + torch.randint(-20,20,(bbox.shape[0],)), 0, 256)
                gt = F.resize(gt, 256, torchvision.transforms.InterpolationMode.NEAREST) # decoder takes image size 256x256
                bg_mask = F.resize(bg_mask, 256, torchvision.transforms.InterpolationMode.NEAREST) # decoder takes mask size 256x256

                ######### ------- plt visualizations after resizing -------- #########
                if visualization_debug:
                    image_data_vis = image_data.cpu().numpy()[0]
                    image_data_vis = (image_data_vis*2.0 + 1.0)/2.0
                    gt_vis = gt.cpu().numpy()[0]
                    bg_mask_vis = bg_mask.cpu().numpy()[0]
                    fig, ax = plt.subplots(1,3,figsize=(30,10))
                    ax[0].imshow(np.transpose(image_data_vis,(1,2,0)))
                    ax[1].imshow(gt_vis[0])
                    ax[2].imshow(bg_mask_vis[0])
                    breakpoint()
                ######### -------------------------------------------------- #########

                # convert gt to one hot encoding
                gt = gt.long().squeeze(1)
                gt = torch.nn.functional.one_hot(gt,num_classes)
                B,_, H, W = gt.shape
                gt = torch.permute(gt,(0,3,1,2))

                bbox = bbox.to(device)            

                sparse_embeddings, dense_embeddings, image_pe = prompt_encoder(
                    points=None,
                    boxes=bbox[:, None, :] if bbox_given else None,
                    masks=bg_mask if bg_mask_given else None,
                )

                # predict image embedding
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

            # Optional, doesn't help much if mask is already being passed as prompt
            if ignore_background: 
                mask_predictions = mask_predictions*bg_mask
                gt = gt*bg_mask
            loss = 0.8*dice_loss(mask_predictions, gt) + 0.2*focal_loss(mask_predictions, gt)
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
            torch.save(sam_model.state_dict(), join(model_save_path, 'model_latest.pth'))
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
            for step, (image_data, gt, bg_mask, bbox) in enumerate(tqdm(test_dataloader,"EVAL")):
            # loading precomputed embeddings during training
                eval_epoch_dir = join(model_save_path, f"eval/{epoch}")
                os.makedirs(eval_epoch_dir, exist_ok=True)
                # not computing gradients for image encoder, prompt encoder and mask decoder during evaluation
                with torch.no_grad():
                    # resize gt to 256x256
                    gt = F.resize(gt, 256, torchvision.transforms.InterpolationMode.NEAREST)
                    gt = torch.nn.functional.one_hot(gt.squeeze(1),num_classes)
                    B,_, H, W = gt.shape
                    gt = torch.permute(gt,(0,3,1,2))
                    # resize bg_mask to 256x256
                    bg_mask = F.resize(bg_mask, 256, torchvision.transforms.InterpolationMode.NEAREST)
                    bg_mask = bg_mask.to(device)
                    # resizing by a factor of 4 (1024-->256)
                    bbox = bbox//4 
                    bbox = bbox.to(device)            
                    # prompt encoding
                    sparse_embeddings, dense_embeddings, image_pe = prompt_encoder(
                        points=None,
                        boxes=bbox[:, None, :] if bbox_given else None,
                        masks=bg_mask if bg_mask_given else None,
                    )
                    # image embedding estimation
                    image_data = image_data.to(device)
                    image_data = F.resize(image_data, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
                    embedding = image_encoder(image_data)
                    # segmentation prediction
                    mask_predictions, _ = mask_decoder(
                        image_embeddings=embedding.to(device), # (B, 256, 64, 64)
                        image_pe=image_pe, # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                        multimask_output=True,
                    )
                    # visualizing last sample from every batch
                    labels_out = torch.argmax(torch.Tensor(mask_predictions[-1]), dim=0) 
                    labels_out_vis = semantics.labels_to_colors(labels_out.cpu().numpy().astype('uint8'))
                    labels_out_vis = cv2.resize(labels_out_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                    gt_vis = torch.argmax(torch.Tensor(gt[-1]), dim=0)
                    gt_vis = semantics.labels_to_colors(gt_vis.cpu().numpy().astype('uint8'))
                    gt_vis = cv2.resize(gt_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                    bg_mask_vis = cv2.resize(bg_mask[-1][0].cpu().numpy().astype('uint8'), (1024,1024), interpolation=cv2.INTER_NEAREST)
                    image_data_vis = np.transpose(image_data[-1].cpu().numpy().astype('uint8'), (1,2,0))
                    # plot eval results
                    TITLE_SIZE = 30
                    fig, ax = plt.subplots(1,4, figsize=(40,10))
                    ax[0].imshow(image_data_vis)
                    ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    ax[1].imshow(bg_mask_vis, cmap='gray')
                    ax[1].set_title("Mask", fontsize=TITLE_SIZE)
                    ax[1].axis('off')
                    ax[2].imshow(labels_out_vis)
                    ax[2].set_title("Prediction", fontsize=TITLE_SIZE)
                    ax[2].axis('off')
                    ax[3].imshow(gt_vis)
                    ax[3].set_title("GT", fontsize=TITLE_SIZE)
                    ax[3].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{step}.png")
                    plt.close()
                    
                    # compute eval loss
                    eval_loss += dice_loss(mask_predictions, gt.to(device)).item()
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
