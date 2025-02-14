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
import segmentation_refinement as segref

from segment_anything_parallel import sam_model_registry
from segment_anything_parallel_fine_infer import sam_model_registry as sam_model_registry_fine

from segment_anything_parallel_fine_infer.utils.transforms import ResizeLongestSide

from utils.dataset import DrawingsDatasetInferFull
from utils.SurfaceDice import compute_dice_coefficient
from utils.SemanticSegmentation import SemanticSegmentationCoarse, SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationAll
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

    # inference_image_dir_path = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/dog_val_images'
    # inference_image_dir_path = '/mnt/users_scratch/astitva/DATA/IN_THE_WILD/'
    inference_image_dir_path = '/mnt/users_scratch/astitva/DATA/OOD_paper/'
    # inference_image_dir_path = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/IN_THE_WILD_IMAGES/images/'
    # inference_image_dir_path = '/mnt/users_scratch/astitva/WORKSPACE/ToonLight/data_generation/3DBiCar_drawings/'
    save_predcitions_dir_path = './INFERENCE/OOD_paper/SINGLE_STAGE/'
    # save_predcitions_dir_path = './PREDICTIONS/3DBiCar_drawings'

    # EXPERIMENT CONFIG
    task_name = '16k_ANIMSEG_E2E_ALL_CLASSES_SlowLR'
    all_ckpts_dir = 'all_ckpts'
    save_task_name = f'infer_{task_name}'
    mode = 'test'
    BATCH_SIZE = 1
    load_best_eval_ckpt = True
    epoch = 300
    encoder_original = False
    bbox_given = False
    bg_mask_given = False
    visualize_heatmap = False
    refine_masks = False
    model_type = 'vit_b'

    # semantic definitions
    num_classes = 27
    semantics = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=num_classes)


    # model checkpoint directory
    model_load_path = join(ckpt_dir, task_name)
    assert os.path.exists(model_load_path), f"Model path {model_load_path} does not exist"

    # checkpoint name and path
    init_checkpoint = join(ckpt_dir, f'{task_name}/model_eval_best.pth')
    if not load_best_eval_ckpt:
        init_checkpoint = join(ckpt_dir, f'{task_name}/{all_ckpts_dir}/model_{epoch}.pth')
        

    device = 'cuda:0'
    device_ids = [i for i in range(torch.cuda.device_count())]

    refiner = None
    if refine_masks:
        refiner = segref.Refiner(device=device) # device can also be 'cpu'

    # prepare and load SAM coarse model
    sam_model = sam_model_registry[model_type](num_classes = num_classes, checkpoint=init_checkpoint).to(device)
    sam_model.image_encoder.to(device)
    sam_model.prompt_encoder.to(device)
    sam_model.prompt_encoder.parallel_training = True
    sam_model.mask_decoder.to(device)

    # parallelize components of SAM coarse model
    image_encoder = torch.nn.DataParallel(sam_model.image_encoder, device_ids=device_ids)
    prompt_encoder = torch.nn.DataParallel(sam_model.prompt_encoder, device_ids=device_ids)
    mask_decoder = torch.nn.DataParallel(sam_model.mask_decoder, device_ids=device_ids)

    # create dataset
    test_dataset = DrawingsDatasetInferFull(sam_model, img_dir_name=inference_image_dir_path)

    # label id definitions
    label_to_id = semantics.data['label_name_to_id']
    id_to_label = {i:j for j,i in label_to_id.items()}

    # create dataloader
    assert BATCH_SIZE==1
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)


    # define differntiable non-learnable upsampling layer
    upsample = torch.nn.Upsample(scale_factor=4, mode='nearest')

    # augmentations
    input_size = (1024, 1024)
    crop_size = (800, 800)

    if not load_best_eval_ckpt:
        print(f'EVAL at Epoch-{epoch}')
    else:
        print(f'EVAL at Best-Eval Epoch')

    # output directory for EVAL
    eval_epoch_dir = join(save_predcitions_dir_path, f"{save_task_name}/{epoch}")
    if load_best_eval_ckpt:
        eval_epoch_dir = join(save_predcitions_dir_path, f"{save_task_name}/best_eval_epoch")
    os.makedirs(eval_epoch_dir, exist_ok=True)
    print(f"EVAL results will be SAVED here --> {eval_epoch_dir}")

    for step, (image_name_string, image_data_eval, image_batch_dims) in enumerate(tqdm(test_dataloader,"EVAL")):
        valid_face_detected = False # reset flag for each image
        image_data_eval = image_data_eval.to(device)
        # not computing gradients for image encoder, prompt encoder and mask decoder during evaluation
        with torch.no_grad():
            # predict coarse mask
            sparse_embeddings, dense_embeddings, image_pe = prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )
            image_data_eval = F.resize(image_data_eval, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
            embedding = image_encoder(image_data_eval)
            mask_predictions, _ = mask_decoder(
                image_embeddings=embedding.to(device), # (B, 256, 64, 64)
                image_pe=image_pe, # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
            )

            mask_predictions = upsample(mask_predictions)

            # convert mask predictions to one-hot
            pred_labels = torch.argmax(mask_predictions, dim=1)
            pred_one_hot = torch.nn.functional.one_hot(pred_labels,num_classes)
            pred_one_hot = torch.permute(pred_one_hot,(0,3,1,2))

            # visualizing samples from every batch
            for batch_idx in range(image_data_eval.shape[0]):
                save_name_string = image_name_string[batch_idx].split('.')[0]
                image_data_vis = image_data_eval.cpu().numpy()[batch_idx] # last sample from batch
                image_data_vis = 255*((image_data_vis - image_data_vis.min()) / (image_data_vis.max() - image_data_vis.min()))
                image_data_vis = np.transpose(image_data_vis,(1,2,0))
                image_data_vis = image_data_vis.astype('uint8')
                labels_out = torch.argmax(torch.Tensor(mask_predictions[batch_idx]), dim=0)  # last sample from batch
                labels_out_refined = torch.clone(labels_out)
                if refine_masks:
                    for label in torch.unique(labels_out_refined):
                        binmask = labels_out==label
                        binmask = binmask.cpu().numpy().astype('uint8')*255
                        refined_binmask = refiner.refine(image_data_vis, binmask, fast=False, L=900)
                        labels_out_refined[refined_binmask>0] = label
                    labels_out_refined_vis = semantics.labels_to_colors(labels_out_refined.cpu().numpy().astype('uint8'))
                    labels_out_refined_vis = cv2.resize(labels_out_refined_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                labels_out_vis = semantics.labels_to_colors(labels_out.cpu().numpy().astype('uint8'))
                labels_out_vis = cv2.resize(labels_out_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                overlayed = cv2.addWeighted(image_data_vis, 0.5, labels_out_vis, 0.5, 0)
                if refine_masks:
                    overlayed_refined = cv2.addWeighted(image_data_vis, 0.5, labels_out_refined_vis, 0.5, 0)

                cv2.imwrite(f'{eval_epoch_dir}/{save_name_string}_seg.png', cv2.cvtColor(labels_out_vis, cv2.COLOR_BGR2RGB))
                # plot eval results
                TITLE_SIZE = 35
                fig, ax = plt.subplots(1,3, figsize=(30,10))
                if refine_masks:
                    fig, ax = plt.subplots(1,4, figsize=(40,10))
                ax[0].imshow(image_data_vis)
                ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                ax[0].axis('off')
                ax[1].imshow(labels_out_vis)
                ax[1].set_title("Prediction", fontsize=TITLE_SIZE)
                ax[1].axis('off')
                ax[2].imshow(overlayed)
                ax[2].set_title("Overlayed Prediction", fontsize=TITLE_SIZE)
                ax[2].axis('off')
                if refine_masks:
                    ax[4].imshow(overlayed_refined)
                    ax[4].set_title("Overlayed (Refined)", fontsize=TITLE_SIZE)
                    ax[4].axis('off')
                plt.savefig(f"{eval_epoch_dir}/{save_name_string}_vis.png")
                plt.close()