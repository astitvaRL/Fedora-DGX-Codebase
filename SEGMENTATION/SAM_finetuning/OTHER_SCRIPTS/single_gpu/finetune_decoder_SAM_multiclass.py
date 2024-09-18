import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
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

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils.dataset import Dataset_body
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentation
join = os.path.join


if __name__ == '__main__':
    # torch.manual_seed(999)
    # np.random.seed(999)

    # set paths
    data_root = '/home/astitva/DATA/AD_SegMaps/'
    labels_definition_file_path = 'label_definition.json'
    ckpt_dir = './checkpoints'
    sam_original_ckpt_path = join(ckpt_dir,'sam_original/sam_vit_b_01ec64.pth')
    image_dir_name = 'drawings_resized_synth'
    label_id_dir_name = 'labels_resized' 
    embed_dir_name = f"{image_dir_name}_embeddings" # precomputed image embeddings will be saved here if not saved already
    embedding_dir_path = join(data_root, embed_dir_name)
    task_name = 'animseg_synth_20k_body' # finetuned checkpoint will be saved here
    model_save_path = join(ckpt_dir, task_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(join(model_save_path, 'train_seg_vis'), exist_ok=True)
    os.makedirs(join(model_save_path, 'eval'), exist_ok=True)
    
    # training choice
    precompute_embeddings = False # False if already precomputed and saved
    resume_training = False
    visualization_debug = False
    ignore_background = False
    bg_mask_given = True
    random_flip = True
    # prepare SAM model
    model_type = 'vit_b'

     # load original SAM cackpoint for finetuning, or load an existing checkpoint for further training
    init_checkpoint = join(sam_original_ckpt_path)
    epoch_start = 0
    if resume_training:
        epoch_start = 48
        resume_ckpt = join(ckpt_dir, 'animseg_synth_3k_decoder_only\\model_best.pth')
        init_checkpoint = resume_ckpt

    device = 'cuda:0'
    num_classes = 18 
    sam_model = sam_model_registry[model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)

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
            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image_data)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
            # precompute and save the image embedding
            with torch.no_grad():
                embedding = sam_model.image_encoder(input_image)
                np.save(join(embedding_dir_path, name.split('.png')[0]+'.npy'), embedding.cpu().numpy()[0])
        print('Image embeddings saved at -->', embedding_dir_path)

    # create dataset
    train_dataset = Dataset_body(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, img_embed_dir_name = embed_dir_name, label_id_dir_name = label_id_dir_name, mode='train')
    test_dataset = Dataset_body(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, img_embed_dir_name = embed_dir_name, label_id_dir_name = label_id_dir_name, mode='test')

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # training config
    num_epochs = 1000
    save_frequency = 1
    eval_frequency = 1
    train_loss_log = []
    eval_loss_log = []
    best_loss = 1e10
    best_eval_loss = 1e10
    visualize = SemanticSegmentation(labels_definition_file_path, num_classes=num_classes)

    # Set up the optimizer, losses, hyperparameters
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    focal_loss = monai.losses.FocalLoss(reduction='mean', gamma=2.0)

   # Freeze all layers of image encoder
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False

    ## verify
    # for name, param in sam_model.image_encoder.named_parameters():
    #     print(name, param.requires_grad)

    # augmentations
    crop_size = (512, 512)
    input_size = (1024, 1024)

    # start training
    sam_model.train()
    for epoch in range(epoch_start, num_epochs):
        epoch_loss = 0
        # TRAINING
        for step, (image_data, image_embedding, gt, bg_mask, bbox) in enumerate(tqdm(train_dataloader)):
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
                    fliph = torchvision.transforms.RandomHorizontalFlip(p=0.5)                
                    flipv = torchvision.transforms.RandomVerticalFlip(p=0.5)                
                    input_all = torch.cat([image_data, gt, bg_mask], axis=1)
                    input_all = fliph(input_all)
                    input_all = flipv(input_all)
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
                    plt.show()
                ######### -------------------------------------------------- #########

                # convert gt to one hot encoding
                gt = gt.long().squeeze(1)
                gt = torch.nn.functional.one_hot(gt,num_classes)
                B,_, H, W = gt.shape
                gt = torch.permute(gt,(0,3,1,2))

                bbox = bbox.to(device)            

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=bbox[:, None, :],
                    masks=bg_mask if bg_mask_given else None,
                )


                # predict image embedding
                image_data = F.resize(image_data, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
                embedding = sam_model.image_encoder(image_data)

            # computing gradients for mask decoder only
            mask_predictions, _ = sam_model.mask_decoder(
                image_embeddings=embedding, # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
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

        epoch_loss /= step
        train_loss_log.append(epoch_loss)
        
        # save the latest model checkpoint as required
        if epoch%save_frequency==0:
            print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
            # save the latest model checkpoint
            torch.save(sam_model.state_dict(), join(model_save_path, 'model_latest.pth'))
            labels_out = torch.argmax(torch.Tensor(mask_predictions[-1]), dim=0) # last sample from randomized current batch
            plt.imshow(visualize.labels_to_colors(labels_out.cpu().numpy().astype('uint8')))
            plt.savefig(join(model_save_path, f"train_seg_vis/{epoch}.png"))
            # save the best model checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(sam_model.state_dict(), join(model_save_path, 'model_best.pth'))
        
        # EVALUATION (on test set)
        if epoch%eval_frequency==0:    
            # reset metrics for latest epoch
            eval_loss = 0
            for step, (image_data, image_embedding, gt, bg_mask, bbox) in enumerate(test_dataloader):
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
                    
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=bbox[:, None, :],
                        masks=bg_mask if bg_mask_given else None,
                    )
                    
                    # no need to predict image embedding, use precomputed embedding for validation

                    mask_predictions, _ = sam_model.mask_decoder(
                        image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                        image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                        multimask_output=True,
                    )
                    labels_out = torch.argmax(torch.Tensor(mask_predictions[-1]), dim=0) # last sample from fixed current batch
                    plt.imshow(visualize.labels_to_colors(labels_out.cpu().numpy().astype('uint8')))
                    plt.savefig(f"{eval_epoch_dir}/{step}.png")

                    # compute eval loss
                    eval_loss += dice_loss(mask_predictions, gt.to(device)).item()
            
            # logging eval loss and metrics
            print(f'EVAL: {epoch}, Loss: {eval_loss}')
            eval_loss_log.append(eval_loss)
            np.save(join(model_save_path,f"eval_loss_log_latest.npy"),np.array(eval_loss_log))

            # save best eval model checkpoint
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(sam_model.state_dict(), join(model_save_path, 'model_eval_best.pth'))

        # logging train loss
        np.save(join(model_save_path,f"train_loss_log_latest.npy"),np.array(train_loss_log))
