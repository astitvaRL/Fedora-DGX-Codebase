import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
# from skimage import io
import imageio as io
from tqdm import tqdm
import torch
import torchvision.transforms.functional as F
import torchvision
from torchvision import transforms
import json
import segmentation_refinement as segref

import sys
sys.path.append('./external/SAM/')
from segment_anything_parallel import sam_model_registry
from segment_anything_parallel_fine_infer import sam_model_registry as sam_model_registry_fine
from segment_anything_parallel_fine_infer.utils.transforms import ResizeLongestSide

from .SemanticSegmentation import SemanticSegmentationCoarse, SemanticSegmentationNoFace, SemanticSegmentationFace, SemanticSegmentationAll

join = os.path.join


# inference dataset definition for all classes, coarse-to-fine + face
class SampleDatasetInference(): 
    def __init__(self, sam_model, image_np):
        self.sam_model = sam_model
        self.num_classes_coarse = 5
        self.num_classes_fine = 18
        self.num_classes_face = 11
        self.num_classes_all = 27
        self.device = torch.device('cuda:0')
        self.image = image_np

    def get_input_tensor(self):

        if self.image.shape[-1]==4:
            alpha = self.image[:,:,3]
            self.image = self.image[:,:,:3]
            self.image[alpha==0] = [255,255,255]
        dims = self.image.shape
        image = cv2.resize(self.image, (1024,1024), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)

        return input_image_tensor, dims

    
###########################################################################################################################################
### ----------------------------------------------------------------------------------------------------------------------------------- ###
###########################################################################################################################################


class SAM_face:


    def __init__(self, ckpt_dir, labels_definition_file_path='label_definition.json', device=torch.device('cuda:0') ):

        # set paths
        self.labels_definition_file_path = labels_definition_file_path
        self.ckpt_dir = ckpt_dir
        self.task_name_coarse = '16k_ANIMSEG_E2E_COARSE'
        self.task_name_fine = '16k_ANIMSEG_E2E_FINE'
        self.task_name_face = '16k_ANIMSEG_E2E_FACE_wBinMask'
        self.all_ckpts_dir = 'all_ckpts'
        self.mode = 'test'
        self.BATCH_SIZE = 1
        self.load_best_eval_ckpt = True
        self.epoch_coarse = 400
        self.epoch_face = 400
        self.epoch_fine = 400
        self.coarse_includes_neck = True
        self.model_type = 'vit_b'
        self.device = device
        self.device_ids = [i for i in range(1)]

        # semantic definitions
        self.num_classes_coarse = 6 # coarse network also includes 'Neck' class which will be merged to torso before feeding to fine network if required
        self.num_classes_fine = 18
        self.num_classes_face = 11
        self.num_classes_all = 27
        self.exclude_neck_from_coarse = True
        self.semantics_coarse = SemanticSegmentationCoarse(labels_definition_path=self.labels_definition_file_path, num_classes=(self.num_classes_coarse-1) if self.exclude_neck_from_coarse else self.num_classes_coarse, exclude_neck=True)
        self.semantics_fine = SemanticSegmentationNoFace(labels_definition_path=self.labels_definition_file_path, num_classes=self.num_classes_fine)
        self.semantics_face = SemanticSegmentationFace(labels_definition_path=self.labels_definition_file_path, num_classes=self.num_classes_face)
        self.semantics_all = SemanticSegmentationAll(labels_definition_path=self.labels_definition_file_path, num_classes=self.num_classes_all)

        # model checkpoint directory
        self.model_load_path = join(self.ckpt_dir, self.task_name_fine)
        assert os.path.exists(self.model_load_path), f"Model path {self.model_load_path} does not exist"

        # checkpoint name and path
        self.init_checkpoint_coarse = join(self.ckpt_dir, f'{self.task_name_coarse}/model_eval_best.pth')
        self.init_checkpoint_fine = join(self.ckpt_dir, f'{self.task_name_fine}/model_eval_best.pth')
        self.init_checkpoint_face = join(self.ckpt_dir, f'{self.task_name_face}/model_eval_best.pth')
        if not self.load_best_eval_ckpt:
            self.init_checkpoint_coarse = join(self.ckpt_dir, f'{self.task_name_coarse}/{self.all_ckpts_dir}/model_{self.epoch_coarse}.pth')
            self.init_checkpoint_fine = join(self.ckpt_dir, f'{self.task_name_fine}/{self.all_ckpts_dir}/model_{self.epoch}.pth')
            self.init_checkpoint_face = join(self.ckpt_dir, f'{self.task_name_face}/{self.all_ckpts_dir}/model_{self.epoch_face}.pth')

        # prepare and load SAM coarse model
        self.sam_model_coarse = sam_model_registry[self.model_type](num_classes = self.num_classes_coarse, checkpoint=self.init_checkpoint_coarse).to(self.device)
        self.sam_model_coarse.image_encoder.to(self.device)
        self.sam_model_coarse.prompt_encoder.to(self.device)
        self.sam_model_coarse.prompt_encoder.parallel_training = True
        self.sam_model_coarse.mask_decoder.to(self.device)
        # prepare and load SAM fine model
        self.sam_model_fine = sam_model_registry_fine[self.model_type](num_classes = self.num_classes_fine, checkpoint=self.init_checkpoint_fine).to(self.device)
        self.sam_model_fine.image_encoder.to(self.device)
        self.sam_model_fine.prompt_encoder.to(self.device)
        self.sam_model_fine.prompt_encoder.parallel_training = True
        self.sam_model_fine.mask_decoder.to(self.device)
        #prepare and load SAM face model
        self.sam_model_face = sam_model_registry[self.model_type](num_classes = self.num_classes_face, checkpoint=self.init_checkpoint_face).to(self.device)
        self.sam_model_face.image_encoder.to(self.device)
        self.sam_model_face.prompt_encoder.to(self.device)
        self.sam_model_face.prompt_encoder.parallel_training = True
        self.sam_model_face.mask_decoder.to(self.device)
        
        # parallelize components of SAM coarse model
        self.image_encoder_coarse = torch.nn.DataParallel(self.sam_model_coarse.image_encoder, device_ids=self.device_ids)
        self.prompt_encoder_coarse = torch.nn.DataParallel(self.sam_model_coarse.prompt_encoder, device_ids=self.device_ids)
        self.mask_decoder_coarse = torch.nn.DataParallel(self.sam_model_coarse.mask_decoder, device_ids=self.device_ids)
        # parallelize components of SAM fine model
        self.image_encoder_fine = torch.nn.DataParallel(self.sam_model_fine.image_encoder, device_ids=self.device_ids)
        self.prompt_encoder_fine = torch.nn.DataParallel(self.sam_model_fine.prompt_encoder, device_ids=self.device_ids)
        self.mask_decoder_fine = torch.nn.DataParallel(self.sam_model_fine.mask_decoder, device_ids=self.device_ids)
        # parallelize components of SAM face model
        self.image_encoder_face = torch.nn.DataParallel(self.sam_model_face.image_encoder, device_ids=self.device_ids)
        self.prompt_encoder_face = torch.nn.DataParallel(self.sam_model_face.prompt_encoder, device_ids=self.device_ids)
        self.mask_decoder_face = torch.nn.DataParallel(self.sam_model_face.mask_decoder, device_ids=self.device_ids)

        # define differntiable non-learnable upsampling layer
        self.upsample = torch.nn.Upsample(scale_factor=4, mode='nearest')




    def predict(self, image_np):
        # create sample dataset
        image_dataset = SampleDatasetInference(self.sam_model_face, image_np)

        # assign semantics
        image_dataset.semantics_coarse = self.semantics_coarse
        image_dataset.semantics_face = self.semantics_face
        image_dataset.semantics_fine = self.semantics_fine

        # convert images to tensors
        image_tensor, dims = image_dataset.get_input_tensor()
        image_data_eval = image_tensor.to(self.device)

       # not computing gradients for image encoder, prompt encoder and mask decoder during evaluation
        with torch.no_grad():
            # predict coarse mask
            sparse_embeddings_coarse, dense_embeddings_coarse, image_pe_coarse = self.prompt_encoder_coarse(
                points=None,
                boxes=None,
                masks=None
            )
            image_data_eval = F.resize(image_data_eval, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
            embedding_coarse = self.image_encoder_coarse(image_data_eval)
            mask_predictions_coarse, _ = self.mask_decoder_coarse(
                image_embeddings=embedding_coarse.to(self.device), # (B, 256, 64, 64)
                image_pe=image_pe_coarse, # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings_coarse, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings_coarse, # (B, 256, 64, 64)
                multimask_output=True,
            )
            coarse_mask = torch.argmax(mask_predictions_coarse, dim=1)
            if self.exclude_neck_from_coarse:
                coarse_mask[coarse_mask==5] = 4 # merging neck with torso
            coarse_mask = torch.nn.functional.one_hot(coarse_mask,(self.num_classes_coarse-1) if self.exclude_neck_from_coarse else self.num_classes_coarse)
            coarse_mask = torch.permute(coarse_mask,(0,3,1,2))

            # predict fine mask
            sparse_embeddings, dense_embeddings, image_pe = self.prompt_encoder_fine(
                points=None,
                boxes=None,
                masks=coarse_mask.float(),
            )
            image_data_eval = F.resize(image_data_eval, 1024, torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
            embedding = self.image_encoder_fine(image_data_eval)
            mask_predictions_fine, _ = self.mask_decoder_fine(
                image_embeddings=embedding.to(self.device), # (B, 256, 64, 64)
                image_pe=image_pe, # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
            )
            mask_predictions_fine = self.upsample(mask_predictions_fine)

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
                    return False, -1
                valid_face_detected = True
                face_binmask = face_binmask[Xs.min():Xs.max(),Ys.min():Ys.max()]
                face_prior = F.resize(face_binmask.unsqueeze(0), (256,256), torchvision.transforms.InterpolationMode.NEAREST).float()
                face_prior = face_prior.unsqueeze(0)
                # predict facial details
                sparse_embeddings_face, dense_embeddings_face, image_pe_face = self.prompt_encoder_face(
                points=None,
                boxes=None,
                masks=face_prior,
                )
                image_data_eval_face = F.resize(image_data_eval_face, (1024,1024), torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
                embedding_face = self.image_encoder_face(image_data_eval_face)
                mask_predictions_face, _ = self.mask_decoder_face(
                    image_embeddings=embedding_face.to(self.device), # (B, 256, 64, 64)
                    image_pe=image_pe_face, # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings_face, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings_face, # (B, 256, 64, 64)
                    multimask_output=True,
                )
                mask_predictions_face = self.upsample(mask_predictions_face)


        # convert mask predictions to numpy image
        pred_labels = torch.argmax(mask_predictions_face, dim=1)
        pred_labels_np = pred_labels.squeeze(0).cpu().numpy().astype('uint8')
        pred_labels_np = self.semantics_face.labels_to_colors(pred_labels_np)
        pred_labels_np = cv2.resize(pred_labels_np, (fh,fw), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((1024,1024,3)).astype('uint8')
        canvas[Xs.min():Xs.max(),Ys.min():Ys.max(),:] = pred_labels_np

        return True, canvas
