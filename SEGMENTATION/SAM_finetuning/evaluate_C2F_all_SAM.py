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

from utils.dataset import DrawingsDatasetC2FAll
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
    # image_dir_name = 'MANIFOLD/animated_drawings_images_prior_april22/cropped_image'
    image_dir_name = 'MANIFOLD/EVAL400'
    label_id_dir_name = 'AD_SegMaps/labels_EVAL400' 
    cache_dir = join(data_root, 'dataset_caches/cache_ALL_EVAL400')
    os.makedirs(cache_dir, exist_ok=True)

    # EXPERIMENT CONFIG
    task_name_coarse = 'ANIMSEG_E2E_NoBinmask_Coarse_REAL7k'
    task_name_fine = 'ANIMSEG_E2E_C2F_REAL7k'
    task_name_face = 'ANIMSEG_E2E_FaceOnly_REAL7k_with_face_prior'
    all_ckpts_dir = 'all_ckpts'
    out_dir = 'eval_real_400_ALL'
    mode = 'test'
    BATCH_SIZE = 1
    load_best_eval_ckpt = True
    epoch = 500
    epoch_coarse = 500
    epoch_face = 500
    encoder_original = False
    cache_available = False 
    ignore_background = False
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

    # set dataset cache path
    cache_path = join(cache_dir, f'{mode}_cache.pt')

    # model checkpoint directory
    model_load_path = join(ckpt_dir, task_name_fine)
    assert os.path.exists(model_load_path), f"Model path {model_load_path} does not exist"
    os.makedirs(join(model_load_path, out_dir), exist_ok=True)
    

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

    # prepare and load SAM coarse model
    sam_model_coarse = sam_model_registry[model_type](num_classes = num_classes_coarse, checkpoint=init_checkpoint_coarse).to(device)
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
    test_dataset = DrawingsDatasetC2FAll(sam_model, labels_definition_file_path=labels_definition_file_path, data_root = data_root, img_dir_name=image_dir_name, label_id_dir_name = label_id_dir_name, mode=mode, num_test_samples=0)

    # assign semantics
    test_dataset.semantics_coarse = semantics_coarse
    test_dataset.semantics_fine = semantics_fine
    test_dataset.semantics_face = semantics_face
    test_dataset.semantics_all = semantics_all

    # label id definitions
    label_to_id = semantics_fine.data['label_name_to_id']
    id_to_label = {i:j for j,i in label_to_id.items()}

    if cache_available:
        print("Preloading Caches...")
        test_dataset.cache_available = True
        test_dataset.load_cache(cache_path)
    else:
        test_dataset.init_cache()

    
    print(f"EVAL CACHE READY! --- Cache Size:", test_dataset.cache.shape)

    # create dataloader
    assert BATCH_SIZE==1
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)

    # Set up the metrics
    dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    focal_loss = monai.losses.FocalLoss(reduction='mean', gamma=2.0)
    label_quality_score = monai.metrics.LabelQualityScore(include_background=False, scalar_reduction='mean')
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_fine).to(device)

    # define differntiable non-learnable upsampling layer
    upsample = torch.nn.Upsample(scale_factor=4, mode='nearest')

    # augmentations
    input_size = (1024, 1024)
    crop_size = (800, 800)
    randomaug = RandomAug(target_size=input_size, crop_size=crop_size)

    if not load_best_eval_ckpt:
        print(f'EVAL at Epoch-{epoch}')
    else:
        print(f'EVAL at Best-Eval Epoch')

    # output directory for EVAL
    eval_epoch_dir = join(model_load_path, f"{out_dir}/{epoch}")
    if load_best_eval_ckpt:
        eval_epoch_dir = join(model_load_path, f"{out_dir}/best_eval")
    os.makedirs(eval_epoch_dir, exist_ok=True)
    print(f"EVAL results will be SAVED here --> {eval_epoch_dir}")

    eval_loss = 0
    mAcc = 0
    classwise_mIoU = [0]*num_classes_all
    valid_face_detected = False # flag to check if face is detected in the image
    for step, (image_data_eval, gt_coarse, gt_fine, gt_face, gt_all) in enumerate(tqdm(test_dataloader,"EVAL")):
        valid_face_detected = False # reset flag for each image
        image_data_eval = image_data_eval.to(device)
        gt_coarse = gt_coarse.to(device)
        gt_fine = gt_fine.to(device)
        gt_face = gt_face.to(device)
        gt_all = gt_all.to(device)
        # not computing gradients for image encoder, prompt encoder and mask decoder during evaluation
        with torch.no_grad():
            # resize gt to 1024x1024
            gt_coarse = F.resize(gt_coarse, 1024, torchvision.transforms.InterpolationMode.NEAREST)
            gt_fine = F.resize(gt_fine, 1024, torchvision.transforms.InterpolationMode.NEAREST)
            gt_face = F.resize(gt_face, 1024, torchvision.transforms.InterpolationMode.NEAREST)
            gt_all = F.resize(gt_all, 1024, torchvision.transforms.InterpolationMode.NEAREST)
            gt_coarse = torch.nn.functional.one_hot(gt_coarse.squeeze(1),(num_classes_coarse-1) if exclude_neck_from_coarse else num_classes_coarse)
            gt_fine = torch.nn.functional.one_hot(gt_fine.squeeze(1),num_classes_fine)
            gt_face = torch.nn.functional.one_hot(gt_face.squeeze(1),num_classes_face)
            gt_all = torch.nn.functional.one_hot(gt_all.squeeze(1),num_classes_all)
            B,_, H, W = gt_all.shape
            gt_coarse = torch.permute(gt_coarse,(0,3,1,2))
            gt_fine = torch.permute(gt_fine,(0,3,1,2))
            gt_face = torch.permute(gt_face,(0,3,1,2))
            gt_all = torch.permute(gt_all,(0,3,1,2))
            
            # predict coarse mask
            sparse_embeddings_coarse, dense_embeddings_coarse, image_pe_coarse = prompt_encoder_coarse(
                points=None,
                boxes=None,
                masks=None
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


            # # compute eval loss
            # eval_loss += dice_loss(mask_predictions_fine, gt_face).item()

            # # convert mask predictions to one-hot
            # pred_labels = torch.argmax(mask_predictions_fine, dim=1)
            # pred_one_hot = torch.nn.functional.one_hot(pred_labels,num_classes_fine)
            # pred_one_hot = torch.permute(pred_one_hot,(0,3,1,2))

            # #compute accuracy
            # batch_accuracy = accuracy(pred_one_hot, gt_fine).mean().item()
            # mAcc += batch_accuracy
            # # compute batch IoU
            # for class_idx in range(1, mask_predictions_fine.shape[1]):
            #     batch_IoU = monai.metrics.compute_iou(pred_one_hot[:,class_idx,:,:].unsqueeze(1), gt_fine[:,class_idx,:,:].unsqueeze(1), include_background=False, ignore_empty=False)
            #     sum_notnans = torch.nan_to_num(batch_IoU, nan=0.0).sum()
            #     count_notnans = torch.isfinite(batch_IoU).sum()
            #     batch_mean_IoU = sum_notnans / count_notnans
            #     classwise_mIoU[class_idx] += batch_mean_IoU.item()
            


            # visualizing samples from every batch
            for batch_idx in range(image_data_eval.shape[0]):
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
                gt_vis = torch.argmax(torch.Tensor(gt_all[batch_idx]), dim=0)  # last sample from batch
                gt_vis = semantics_all.labels_to_colors(gt_vis.cpu().numpy().astype('uint8'))
                gt_vis = cv2.resize(gt_vis, (1024,1024), interpolation=cv2.INTER_NEAREST)
                coarse_mask_labels = torch.argmax(coarse_mask[batch_idx], dim=0)
                coarse_mask_vis = cv2.resize(coarse_mask_labels.cpu().numpy().astype('uint8'), (1024,1024), interpolation=cv2.INTER_NEAREST)
                coarse_mask_vis = semantics_coarse.labels_to_colors(coarse_mask_vis)
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
                    ax[1].imshow(coarse_mask_vis, cmap='gray')
                    ax[1].set_title("Prediction (Coarse)", fontsize=TITLE_SIZE)
                    ax[1].axis('off')
                    ax[2].imshow(labels_out_vis)
                    ax[2].set_title("Prediction (Fine)", fontsize=TITLE_SIZE)
                    ax[2].axis('off')
                    ax[3].imshow(gt_vis)
                    ax[3].set_title("GT", fontsize=TITLE_SIZE)
                    ax[3].axis('off')
                    ax[4].imshow(overlayed)
                    ax[4].set_title("Overlayed Prediction (Fine)", fontsize=TITLE_SIZE)
                    ax[4].axis('off')
                    metric_string = '999' # str(batch_accuracy).replace('.','_')[:7]
                    plt.savefig(f"{eval_epoch_dir}/{metric_string}_{step}_{batch_idx}.png")
                    plt.close()
                else:
                    fig, ax = plt.subplots(1,4, figsize=(40,10))
                    if refine_masks:
                        fig, ax = plt.subplots(1,5, figsize=(50,10))
                    ax[0].imshow(image_data_vis)
                    ax[0].set_title("Input Image", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    ax[1].imshow(labels_out_vis)
                    ax[1].set_title("Prediction", fontsize=TITLE_SIZE)
                    ax[1].axis('off')
                    ax[2].imshow(gt_vis)
                    ax[2].set_title("GT", fontsize=TITLE_SIZE)
                    ax[2].axis('off')
                    ax[3].imshow(overlayed)
                    ax[3].set_title("Overlayed Prediction", fontsize=TITLE_SIZE)
                    ax[3].axis('off')
                    if refine_masks:
                        ax[4].imshow(overlayed_refined)
                        ax[4].set_title("Overlayed (Refined)", fontsize=TITLE_SIZE)
                        ax[4].axis('off')
                    metric_string = str(batch_accuracy).replace('.','_')[:7]
                    plt.savefig(f"{eval_epoch_dir}/{metric_string}_{step}_{batch_idx}.png")
                    plt.close()
                
                if visualize_heatmap:
                    # show heatmap
                    fig, ax = plt.subplots(1,num_classes_all+1, figsize=((num_classes_all+1)*10,10))
                    ax[0].imshow(labels_out_vis)
                    ax[0].set_title("Prediction", fontsize=TITLE_SIZE)
                    ax[0].axis('off')
                    #normalize mask predictions across channels
                    heatmaps = mask_predictions[batch_idx]
                    heatmaps = heatmaps/heatmaps.sum(dim=0, keepdim=True)
                    for i in range(1,num_classes_all+1):
                        heatmap = heatmaps[i-1].cpu().numpy().astype('float32')
                        remapped_id = semantics_fine.reverse_remap[i-1]
                        class_name = id_to_label[remapped_id]
                        if coarse:
                            class_name = semantics_fine.class_names[i-1]
                        ax[i].imshow(heatmap, cmap='jet_r')
                        ax[i].set_title(f"{class_name}", fontsize=TITLE_SIZE)
                        ax[i].axis('off')
                    plt.savefig(f"{eval_epoch_dir}/{step}_{batch_idx}_heatmap.png")
                    plt.close()
                    
    # # logging metrics
    # eval_loss /= (step+1)
    # mAcc /= (step+1)
    # classwise_mIoU = np.array(classwise_mIoU)/(step+1)
    # mIoU = classwise_mIoU.sum()/(num_classes_fine-1)

    # print(f'EVAL-->{step} steps')
    # print(f'Eval Loss: {eval_loss}')
    # print(f'Mean-Accuracy: {mAcc}')
    # print(f'Classwise Mean-IoU: {classwise_mIoU}')
    # print(f'Total Mean-IoU: {mIoU}')

    # save cache if not already saved
    if not(cache_available):
        test_dataset.save_cache(cache_path)
