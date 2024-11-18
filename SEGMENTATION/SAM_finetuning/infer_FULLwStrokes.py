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

from utils.dataset import DrawingsDatasetInferFullWithCoarsePrior
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

    # suffix =  '119'
    # inference_image_dir_path = '/mnt/users_scratch/astitva/WORKSPACE/ToonLight/data_generation/3DBiCar_360_Renders/' + suffix
    # save_predcitions_dir_path = './PREDICTIONS/3BiCar_360_Renders/' + suffix + '/'

    inference_image_dir_path = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/DOG_DATASET/val_images'
    coarse_labels_dir_path = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/DOG_DATASET/val_segmentations'
    save_predcitions_dir_path = './PREDICTIONS/Dog/infer_w_strokes/val/'

    # EXPERIMENT CONFIG
    task_name_coarse = 'ANIMSEG_E2E_Strokes2Coarse_randomrot'
    task_name_fine = 'ANIMSEG_E2E_C2F_REAL7k'
    task_name_face = 'ANIMSEG_E2E_FaceOnly_REAL7k_with_face_prior'
    all_ckpts_dir = 'all_ckpts'
    save_task_name = 'eval_real_400_ALL'
    mode = 'test'
    BATCH_SIZE = 1
    load_best_eval_ckpt = True
    epoch = 500
    epoch_coarse = 500
    epoch_face = 500
    encoder_original = False
    bbox_given = False
    visualize_coarse = True
    coarse_includes_neck = True
    bg_mask_given = False
    visualize_heatmap = False
    refine_masks = False
    model_type = 'vit_b'

    # semantic definitions
    num_classes_coarse = 6 # coarse network also includes 'Neck' class which will be merged to torso before feeding to fine network if required
    num_classes_fine = 18
    num_classes_face = 11
    num_classes_all = 27
    exclude_neck_from_coarse = True
    semantics_coarse = SemanticSegmentationCoarse(labels_definition_path=labels_definition_file_path, num_classes=(num_classes_coarse-1) if exclude_neck_from_coarse else num_classes_coarse, exclude_neck=True)
    semantics_fine = SemanticSegmentationNoFace(labels_definition_path=labels_definition_file_path, num_classes=num_classes_fine)
    semantics_face = SemanticSegmentationFace(labels_definition_path=labels_definition_file_path, num_classes=num_classes_face)
    semantics_all = SemanticSegmentationAll(labels_definition_path=labels_definition_file_path, num_classes=num_classes_all)


    # model checkpoint directory
    model_load_path = join(ckpt_dir, task_name_fine)
    assert os.path.exists(model_load_path), f"Model path {model_load_path} does not exist"

    # checkpoint name and path
    init_checkpoint_coarse = join(ckpt_dir, f'{task_name_coarse}/model_eval_best.pth')
    init_checkpoint_fine = join(ckpt_dir, f'{task_name_fine}/model_eval_best.pth')
    init_checkpoint_face = join(ckpt_dir, f'{task_name_face}/model_eval_best.pth')
    if not load_best_eval_ckpt:
        init_checkpoint_coarse = join(ckpt_dir, f'{task_name_coarse}/{all_ckpts_dir}/model_{epoch_coarse}.pth')
        init_checkpoint_fine = join(ckpt_dir, f'{task_name_fine}/{all_ckpts_dir}/model_{epoch}.pth')
        init_checkpoint_face = join(ckpt_dir, f'{task_name_face}/{all_ckpts_dir}/model_{epoch_face}.pth')
        

    device = 'cuda:0'
    device_ids = [i for i in range(torch.cuda.device_count())]

    refiner = None
    if refine_masks:
        refiner = segref.Refiner(device=device) # device can also be 'cpu'

    # prepare and load SAM coarse model with coarse support (uses implementation of fine model for adding semantic prior)
    sam_model_coarse = sam_model_registry_fine[model_type](num_classes = num_classes_coarse, checkpoint=init_checkpoint_coarse).to(device)
    sam_model_coarse.image_encoder.to(device)
    sam_model_coarse.prompt_encoder.to(device)
    sam_model_coarse.prompt_encoder.parallel_training = True
    sam_model_coarse.mask_decoder.to(device)

    # prepare and load SAM fine model
    sam_model = sam_model_registry_fine[model_type](num_classes = num_classes_fine, checkpoint=init_checkpoint_fine).to(device)
    sam_model.image_encoder.to(device)
    sam_model.prompt_encoder.to(device)
    sam_model.prompt_encoder.parallel_training = True
    sam_model.mask_decoder.to(device)

    #prepare and load SAM face model
    sam_model_face = sam_model_registry[model_type](num_classes = num_classes_face, checkpoint=init_checkpoint_face).to(device)
    sam_model_face.image_encoder.to(device)
    sam_model_face.prompt_encoder.to(device)
    sam_model_face.prompt_encoder.parallel_training = True
    sam_model_face.mask_decoder.to(device)

    # parallelize components of SAM coarse model
    image_encoder_coarse = torch.nn.DataParallel(sam_model_coarse.image_encoder, device_ids=device_ids)
    prompt_encoder_coarse = torch.nn.DataParallel(sam_model_coarse.prompt_encoder, device_ids=device_ids)
    mask_decoder_coarse = torch.nn.DataParallel(sam_model_coarse.mask_decoder, device_ids=device_ids)

    # parallelize components of SAM fine model
    image_encoder = torch.nn.DataParallel(sam_model.image_encoder, device_ids=device_ids)
    prompt_encoder = torch.nn.DataParallel(sam_model.prompt_encoder, device_ids=device_ids)
    mask_decoder = torch.nn.DataParallel(sam_model.mask_decoder, device_ids=device_ids)

    # parallelize components of SAM face model
    image_encoder_face = torch.nn.DataParallel(sam_model_face.image_encoder, device_ids=device_ids)
    prompt_encoder_face = torch.nn.DataParallel(sam_model_face.prompt_encoder, device_ids=device_ids)
    mask_decoder_face = torch.nn.DataParallel(sam_model_face.mask_decoder, device_ids=device_ids)


    # create dataset
    test_dataset = DrawingsDatasetInferFullWithCoarsePrior(sam_model, img_dir_name=inference_image_dir_path, label_id_dir_name=coarse_labels_dir_path, semantics=semantics_coarse)

    # label id definitions
    label_to_id = semantics_fine.data['label_name_to_id']
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

    valid_face_detected = False # flag to check if face is detected in the image
    for step, (image_name_string, image_data_eval, strokes_coarse_prior) in enumerate(tqdm(test_dataloader,"EVAL")):

        valid_face_detected = False # reset flag for each image
        image_data_eval = image_data_eval.to(device)
        strokes_coarse = strokes_coarse_prior.to(device)

        #using input coarse strokes prior
        strokes_coarse = F.resize(strokes_coarse, 256, torchvision.transforms.InterpolationMode.BILINEAR)
        strokes_coarse = strokes_coarse.squeeze(1)
        strokes_coarse = torch.nn.functional.one_hot(strokes_coarse,(num_classes_coarse-1) if exclude_neck_from_coarse else num_classes_coarse)
        strokes_coarse = torch.permute(strokes_coarse,(0,3,1,2))

        # not computing gradients for image encoder, prompt encoder and mask decoder during evaluation
        with torch.no_grad():
            # predict coarse mask
            sparse_embeddings_coarse, dense_embeddings_coarse, image_pe_coarse = prompt_encoder_coarse(
                points=None,
                boxes=None,
                masks=strokes_coarse.float()
            )
            image_data_eval = F.resize(image_data_eval, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
            embedding_coarse = image_encoder_coarse(image_data_eval)
            mask_predictions_coarse, _ = mask_decoder_coarse(
                image_embeddings=embedding_coarse.to(device), # (B, 256, 64, 64)
                image_pe=image_pe_coarse, # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings_coarse, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings_coarse, # (B, 256, 64, 64)
                multimask_output=True,
            )
            coarse_mask = torch.argmax(mask_predictions_coarse, dim=1)
            if exclude_neck_from_coarse:
                coarse_mask[coarse_mask==5] = 4 # merging neck with torso

            coarse_mask = torch.nn.functional.one_hot(coarse_mask,(num_classes_coarse-1) if exclude_neck_from_coarse else num_classes_coarse)
            coarse_mask = torch.permute(coarse_mask,(0,3,1,2))

            # predict fine mask
            sparse_embeddings, dense_embeddings, image_pe = prompt_encoder(
                points=None,
                boxes=None,
                masks=coarse_mask.float(),
            )
            image_data_eval = F.resize(image_data_eval, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
            embedding = image_encoder(image_data_eval)
            mask_predictions_fine, _ = mask_decoder(
                image_embeddings=embedding.to(device), # (B, 256, 64, 64)
                image_pe=image_pe, # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
            )
            mask_predictions_fine = upsample(mask_predictions_fine)

            # masking the background
            bg = 1 - coarse_mask[:,0,:,:].unsqueeze(1)
            bg = upsample(bg.float())
            mask_predictions_fine = mask_predictions_fine * bg

            # extract face region
            face_binmask = torch.argmax(mask_predictions_fine, dim=1)
            face_binmask = face_binmask.squeeze(0)
            face_binmask[face_binmask!=2] = 0 # '2' is the label-id of face in fine segmap definiton
            face_binmask[face_binmask==2] = 1
            if face_binmask.sum().item()>0: # no face detected
                Xs = torch.where(face_binmask>0)[0]
                Ys = torch.where(face_binmask>0)[1]
                image_data_eval_face = image_data_eval[:,:,Xs.min():Xs.max(),Ys.min():Ys.max()] # crop image to face
                _,_,fw,fh = image_data_eval_face.shape
                if fw==0 or fh==0: # invalid prediction
                    continue
                valid_face_detected = True
                face_binmask = face_binmask[Xs.min():Xs.max(),Ys.min():Ys.max()]
                face_prior = F.resize(face_binmask.unsqueeze(0), (256,256), torchvision.transforms.InterpolationMode.NEAREST).float()
                face_prior = face_prior.unsqueeze(0)
                # predict facial details
                sparse_embeddings_face, dense_embeddings_face, image_pe_face = prompt_encoder_face(
                points=None,
                boxes=None,
                masks=face_prior,
                )
                image_data_eval_face = F.resize(image_data_eval_face, (1024,1024), torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
                embedding_face = image_encoder_face(image_data_eval_face)
                mask_predictions_face, _ = mask_decoder_face(
                    image_embeddings=embedding_face.to(device), # (B, 256, 64, 64)
                    image_pe=image_pe_face, # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings_face, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings_face, # (B, 256, 64, 64)
                    multimask_output=True,
                )
                mask_predictions_face = upsample(mask_predictions_face)


            # convert mask predictions to one-hot
            pred_labels = torch.argmax(mask_predictions_fine, dim=1)
            pred_one_hot = torch.nn.functional.one_hot(pred_labels,num_classes_fine)
            pred_one_hot = torch.permute(pred_one_hot,(0,3,1,2))

            # visualizing samples from every batch
            for batch_idx in range(image_data_eval.shape[0]):
                save_name_string = image_name_string[batch_idx].split('.')[0]
                image_data_vis = image_data_eval.cpu().numpy()[batch_idx] # last sample from batch
                image_data_vis = 255*((image_data_vis - image_data_vis.min()) / (image_data_vis.max() - image_data_vis.min()))
                image_data_vis = np.transpose(image_data_vis,(1,2,0))
                image_data_vis = image_data_vis.astype('uint8')
                labels_out = torch.argmax(torch.Tensor(mask_predictions_fine[batch_idx]), dim=0)  # last sample from batch
                labels_out_refined = torch.clone(labels_out)
                if refine_masks:
                    for label in torch.unique(labels_out_refined):
                        binmask = labels_out==label
                        binmask = binmask.cpu().numpy().astype('uint8')*255
                        refined_binmask = refiner.refine(image_data_vis, binmask, fast=False, L=900)
                        labels_out_refined[refined_binmask>0] = label
                    labels_out_refined_vis = semantics_fine.labels_to_colors(labels_out_refined.cpu().numpy().astype('uint8'))
                    labels_out_refined_vis = cv2.resize(labels_out_refined_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                labels_out_vis = semantics_fine.labels_to_colors(labels_out.cpu().numpy().astype('uint8'))

                if valid_face_detected:
                    labels_out_face = torch.argmax(torch.Tensor(mask_predictions_face[batch_idx]), dim=0)  # last sample from batch
                    labels_out_face_vis = semantics_face.labels_to_colors(labels_out_face.cpu().numpy().astype('uint8'))
                    facial_details = cv2.resize(labels_out_face_vis, (fh,fw), interpolation=cv2.INTER_NEAREST)
                    canvas = np.zeros((1024,1024,3)).astype('uint8')
                    canvas[Xs.min():Xs.max(),Ys.min():Ys.max(),:] = facial_details
                    face_mask = canvas.sum(axis=2)>0
                    labels_out_vis[face_mask] = [0,0,0]
                    labels_out_vis += canvas

                labels_out_vis = cv2.resize(labels_out_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                coarse_mask_labels = torch.argmax(coarse_mask[batch_idx], dim=0)
                coarse_mask_vis = cv2.resize(coarse_mask_labels.cpu().numpy().astype('uint8'), (1024,1024), interpolation=cv2.INTER_NEAREST)
                coarse_mask_vis = semantics_coarse.labels_to_colors(coarse_mask_vis)
                strokes_vis = cv2.resize(strokes_coarse_prior[batch_idx].squeeze(0).numpy().astype('uint8'), (1024,1024), interpolation=cv2.INTER_NEAREST)
                strokes_vis = semantics_coarse.labels_to_colors(strokes_vis)
                overlayed = cv2.addWeighted(image_data_vis, 0.5, labels_out_vis, 0.5, 0)
                if refine_masks:
                    overlayed_refined = cv2.addWeighted(image_data_vis, 0.5, labels_out_refined_vis, 0.5, 0)

                # plot eval results
                TITLE_SIZE = 35
                if visualize_coarse:
                    fig, ax = plt.subplots(1,5, figsize=(50,10))
                    ax[0].imshow(image_data_vis)
                    ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    ax[1].imshow(strokes_vis)
                    ax[1].set_title("Input Strokes", fontsize=TITLE_SIZE)
                    ax[1].axis('off')
                    ax[2].imshow(coarse_mask_vis)
                    ax[2].set_title("Prediction (Coarse)", fontsize=TITLE_SIZE)
                    ax[2].axis('off')
                    ax[3].imshow(labels_out_vis)
                    ax[3].set_title("Prediction (Fine)", fontsize=TITLE_SIZE)
                    ax[3].axis('off')
                    ax[4].imshow(overlayed)
                    ax[4].set_title("Overlayed Prediction (Fine)", fontsize=TITLE_SIZE)
                    ax[4].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{save_name_string}_vis.png")
                    plt.close()
                else:
                    fig, ax = plt.subplots(1,3, figsize=(30,10))
                    if refine_masks:
                        fig, ax = plt.subplots(1,5, figsize=(50,10))
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
                
                if visualize_heatmap:
                    # show heatmap
                    fig, ax = plt.subplots(1,num_classes_all+1, figsize=((num_classes_all+1)*10,10))
                    ax[0].imshow(labels_out_vis)
                    ax[0].set_title("Prediction", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    #normalize mask predictions across channels
                    heatmaps = mask_predictions_fine[batch_idx]
                    heatmaps = heatmaps/heatmaps.sum(dim=0, keepdim=True)
                    for i in range(1,num_classes_all+1):
                        heatmap = heatmaps[i-1].cpu().numpy().astype('float32')
                        remapped_id = semantics_fine.reverse_remap[i-1]
                        class_name = id_to_label[remapped_id]
                        # if coarse:
                        #     class_name = semantics_fine.class_names[i-1]
                        ax[i].imshow(heatmap, cmap='jet_r')
                        ax[i].set_title(f"{class_name}", fontsize=TITLE_SIZE)
                        ax[i].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{save_name_string}_heatmap.png")
                    plt.close()
                    