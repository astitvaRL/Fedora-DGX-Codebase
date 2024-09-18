import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
# from skimage import io
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

from utils.dataset import Dataset_body, Dataset_body_real
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentation, SemanticSegmentationTernary
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
    # image_dir_name = 'AD_SegMaps/drawings_synth_20k'
    label_id_dir_name = 'AD_SegMaps/labels_2k_1024' 
    embed_dir_name = f"{image_dir_name}_embeddings" # precomputed image embeddings will be saved here if not saved already
    embedding_dir_path = join(data_root, embed_dir_name)
    cache_dir = join(data_root, 'cache_ternary_real_2k')
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = join(cache_dir, 'train_cache.pt')
    test_cache_path = join(cache_dir, 'test_cache.pt')

    # EXPERIMENT CONFIG
    task_name = 'multigpu_ternary_randomaug_syn_17k'
    encoder_pretrained = False

    model_save_path = join(ckpt_dir, task_name)
    assert os.path.exists(model_save_path), f"Model path {model_save_path} does not exist"
    os.makedirs(join(model_save_path, 'inference'), exist_ok=True)
    
    # inference choice
    cache_available = True # save dataset cache after first epoch
    ignore_background = False
    bbox_given = False
    bg_mask_given = True
    model_type = 'vit_b'
    num_classes = 18 

     # load original SAM cackpoint for finetuning, or load an existing checkpoint for further training
    init_checkpoint = join(ckpt_dir, f'{task_name}/model_latest.pth')

    device = 'cuda:0'
    device_ids = [i for i in range(torch.cuda.device_count())]

    sam_model = sam_model_registry[model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)
    sam_model_encoder = sam_model_registry[model_type](num_classes = num_classes, checkpoint=sam_original_ckpt_path).to(device)
    
    if not encoder_pretrained:
        sam_model_encoder = sam_model

    sam_model_encoder.image_encoder.to(device)
    sam_model_encoder.prompt_encoder.to(device)
    sam_model_encoder.prompt_encoder.parallel_training = True
    sam_model.mask_decoder.to(device)

    image_encoder = torch.nn.DataParallel(sam_model_encoder.image_encoder, device_ids=device_ids)
    prompt_encoder = torch.nn.DataParallel(sam_model_encoder.prompt_encoder, device_ids=device_ids)
    mask_decoder = torch.nn.DataParallel(sam_model.mask_decoder, device_ids=device_ids)

    # create dataset
    # train_dataset = Dataset_body_real(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, img_embed_dir_name = embed_dir_name, label_id_dir_name = label_id_dir_name, mode='train')
    test_dataset = Dataset_body_real(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, img_embed_dir_name = embed_dir_name, label_id_dir_name = label_id_dir_name, mode='test')
    
    # define semantics
    semantics = None
    if num_classes==18:
        semantics = SemanticSegmentation(labels_definition_file_path, num_classes=num_classes)
    elif num_classes==3:
        semantics = SemanticSegmentationTernary(labels_definition_file_path, num_classes=num_classes)
    # train_dataset.semantics = semantics
    test_dataset.semantics = semantics

    print("Preloading Caches...")
    # train_dataset.cache_available = True
    # train_dataset.load_cache(train_cache_path)
    test_dataset.cache_available = True
    test_dataset.load_cache(test_cache_path)
    
    # print(f"INFERENCE CACHE READY! ---", train_dataset.cache.shape)
    print(f"INFERENCE CACHE READY! ---", test_dataset.cache.shape)

    # create dataloader
    # train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=True)

    # training config

    # Set up the optimizer, losses, hyperparameters
    dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    focal_loss = monai.losses.FocalLoss(reduction='mean', gamma=2.0)

    epoch = 89
    eval_loss = 0
    for step, (image_data_eval, gt, bg_mask, bbox) in enumerate(tqdm(test_dataloader,"EVAL")):
    # loading precomputed embeddings during training
        eval_epoch_dir = join(model_save_path, f"inference/{epoch}")
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
            image_data_eval = image_data_eval.to(device)
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
            
            # compute eval loss
            eval_loss += dice_loss(mask_predictions, gt.to(device)).item()
    # logging eval loss and metrics
    eval_loss /= (step+1)
    print(f'INFERENCE, Loss: {eval_loss}')
