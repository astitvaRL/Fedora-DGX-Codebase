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
import torchmetrics

from segment_anything_parallel import SamPredictor, sam_model_registry
from segment_anything_parallel.utils.transforms import ResizeLongestSide

from utils.dataset import DrawingsDatasetInference
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationNoFace, SemanticSegmentationTernary, SemanticSegmentationCoarse, SemanticSegmentationFace
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
    image_dir_name = 'LIP/Testing_images/Testing_images/testing_images'
    # image_dir_name = 'MANIFOLD/Stimuli_HighRes'
    cache_dir = join(data_root, 'dataset_caches/cache_IN_THE_WILD')
    # cache_dir = join(data_root, 'dataset_caches/cache_Infer_REAL300')
    os.makedirs(cache_dir, exist_ok=True)
    inference_cache_path = join(cache_dir, 'inference_cache.pt')

    # EXPERIMENT CONFIG
    task_name = 'ANIMSEG_E2E_NoBinmask_Coarse_REAL7k'
    all_ckpts_dir = 'all_ckpts'
    out_dir = 'inference_LIP'
    load_best_eval_ckpt = True
    epoch = 250
    encoder_original = False
    face_only = False
    coarse = True
    cache_available = False
    ignore_background = False
    bbox_given = False
    bg_mask_given = False
    visualize_heatmap = False
    num_classes = 6
    model_type = 'vit_b'

    # for facial details only
    if face_only:
        coarse = False
        num_classes = 11
    
    # model checkpoint path
    model_load_path = join(ckpt_dir, task_name)
    assert os.path.exists(model_load_path), f"Model path {model_load_path} does not exist"
    os.makedirs(join(model_load_path, out_dir), exist_ok=True)
    

     # load original SAM cackpoint for finetuning, or load an existing checkpoint for further training
    init_checkpoint = join(ckpt_dir, f'{task_name}/model_eval_best.pth')
    if not load_best_eval_ckpt:
        init_checkpoint = join(ckpt_dir, f'{task_name}/{all_ckpts_dir}/model_{epoch}.pth')

    device = 'cuda:0'
    device_ids = [i for i in range(torch.cuda.device_count())]

    sam_model = sam_model_registry[model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)
    sam_model_encoder = sam_model_registry[model_type](num_classes = num_classes, checkpoint=sam_original_ckpt_path).to(device)
    
    if not encoder_original:
        sam_model_encoder = sam_model

    sam_model_encoder.image_encoder.to(device)
    sam_model_encoder.prompt_encoder.to(device)
    sam_model_encoder.prompt_encoder.parallel_training = True
    sam_model.mask_decoder.to(device)

    image_encoder = torch.nn.DataParallel(sam_model_encoder.image_encoder, device_ids=device_ids)
    prompt_encoder = torch.nn.DataParallel(sam_model_encoder.prompt_encoder, device_ids=device_ids)
    mask_decoder = torch.nn.DataParallel(sam_model.mask_decoder, device_ids=device_ids)

    # create dataset
    inference_dataset = DrawingsDatasetInference(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name)

    # define semantics
    semantics = None
    if face_only:
        semantics = SemanticSegmentationFace(labels_definition_file_path, num_classes=num_classes)
    elif num_classes==18:
        semantics = SemanticSegmentationNoFace(labels_definition_file_path, num_classes=num_classes)
    elif num_classes==3:
        semantics = SemanticSegmentationTernary(labels_definition_file_path, num_classes=num_classes)
    elif num_classes==6:
        semantics = SemanticSegmentationCoarse(labels_definition_file_path, num_classes=num_classes)
    
    # train_dataset.semantics = semantics
    inference_dataset.semantics = semantics

    # label id definitions
    label_to_id = semantics.data['label_name_to_id']
    id_to_label = {i:j for j,i in label_to_id.items()}

    if cache_available:
        print("Preloading Cache...")
        inference_dataset.cache_available = True
        inference_dataset.load_cache(inference_cache_path)
    else:
        print("NEW INFERENCE CACHE will be saved here -->", cache_dir)
        inference_dataset.init_cache()
    
    print(f"INFERENCE CACHE READY! --- Cache Size:", inference_dataset.cache.shape)

    # create dataloader
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    # Set up the metrics
    dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    focal_loss = monai.losses.FocalLoss(reduction='mean', gamma=2.0)
    label_quality_score = monai.metrics.LabelQualityScore(include_background=False, scalar_reduction='mean')
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # define differntiable non-learnable upsampling layer
    upsample = torch.nn.Upsample(scale_factor=4, mode='nearest')

    if not load_best_eval_ckpt:
        print(f'INFERENCE at Epoch-{epoch}')
    else:
        print(f'INFERENCE at Best-Eval Epoch')

    # output directory for inference
    eval_epoch_dir = join(model_load_path, f"{out_dir}/{epoch}")
    if load_best_eval_ckpt:
        eval_epoch_dir = join(model_load_path, f"{out_dir}/best_eval")
    os.makedirs(eval_epoch_dir, exist_ok=True)
    print(f"INFERENCE results will be SAVED here --> {eval_epoch_dir}")

    eval_loss = 0
    mAcc = 0
    classwise_mIoU = [0]*num_classes
    for step, image_data_infer in enumerate(tqdm(inference_dataloader,"INFERENCE")):
        image_data_infer = image_data_infer.to(device)

        # not computing gradients for image encoder, prompt encoder and mask decoder during evaluation
        with torch.no_grad():
            # resize gt to 1024x1024
            B,_, H, W = image_data_infer.shape
            sparse_embeddings, dense_embeddings, image_pe = prompt_encoder(
                points=None,
                boxes=bbox[:, None, :] if bbox_given else None,
                masks=bg_mask if bg_mask_given else None,
            )
            # image embedding estimation
            image_data_infer = image_data_infer.to(device)
            image_data_infer = F.resize(image_data_infer, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
            embedding = image_encoder(image_data_infer)
            # segmentation prediction
            mask_predictions, _ = mask_decoder(
                image_embeddings=embedding.to(device), # (B, 256, 64, 64)
                image_pe=image_pe, # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
            )

            #upsample mask predictions to 1024x1024
            mask_predictions = upsample(mask_predictions)

            # visualizing samples from every batch
            for batch_idx in range(image_data_infer.shape[0]):
                labels_out = torch.argmax(torch.Tensor(mask_predictions[batch_idx]), dim=0)  # last sample from batch
                labels_out_vis = semantics.labels_to_colors(labels_out.cpu().numpy().astype('uint8'))
                labels_out_vis = cv2.resize(labels_out_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                image_data_vis = image_data_infer.cpu().numpy()[batch_idx] # last sample from batch
                image_data_vis = (image_data_vis - image_data_vis.min()) / (image_data_vis.max() - image_data_vis.min())
                image_data_vis = np.transpose(image_data_vis,(1,2,0))

                # plot eval results
                TITLE_SIZE = 30
                if bg_mask_given:
                    fig, ax = plt.subplots(1,2, figsize=(30,10))
                    ax[0].imshow(image_data_vis)
                    ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    ax[1].imshow(bg_mask_vis, cmap='gray')
                    ax[1].set_title("Mask", fontsize=TITLE_SIZE)
                    ax[1].axis('off')
                    ax[2].imshow(labels_out_vis)
                    ax[2].set_title("Prediction", fontsize=TITLE_SIZE)
                    ax[2].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{step}_{batch_idx}.png")
                    plt.close()
                else:
                    fig, ax = plt.subplots(1,2, figsize=(20,10))
                    ax[0].imshow(image_data_vis)
                    ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    ax[1].imshow(labels_out_vis)
                    ax[1].set_title("Prediction", fontsize=TITLE_SIZE)
                    ax[1].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{step}_{batch_idx}.png")
                    plt.close()
                
                if visualize_heatmap:
                    # show heatmap
                    fig, ax = plt.subplots(1,num_classes, figsize=((num_classes)*10,10))
                    ax[0].imshow(image_data_vis)
                    ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    #normalize mask predictions across channels
                    heatmaps = mask_predictions[batch_idx]
                    heatmaps = heatmaps/heatmaps.sum(dim=0, keepdim=True)
                    for i in range(1,num_classes): #skipping background class
                        heatmap = heatmaps[i].cpu().numpy().astype('float32')
                        remapped_id = semantics.reverse_remap[i]
                        class_name = id_to_label[remapped_id]
                        if coarse:
                            class_name = semantics.class_names[i]
                        ax[i].imshow(heatmap, cmap='jet_r')
                        ax[i].set_title(f"{class_name}", fontsize=TITLE_SIZE)
                        ax[i].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{step}_{batch_idx}_heatmap.png")
                    plt.close()

    # save cache if not already saved
    if not(cache_available):
        inference_dataset.save_cache(inference_cache_path)
