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


# dataset definition for only face
class DrawingsDatasetInference(): 
    def __init__(self, sam_model, labels_definition_file_path, image, face_binmask, device):
        self.sam_model = sam_model
        self.labels_definition_file_path = labels_definition_file_path
        self.num_classes = 11
        self.semantics = SemanticSegmentationFace(labels_definition_path=self.labels_definition_file_path, num_classes=self.num_classes)
        self.device = device
        self.image = image
        self.face_binmask = face_binmask

    def __len__(self):
        return len(self.files)

    def get_input_tensors(self):
        # image_name = f"{self.files[index]}"
        # image = cv2.imread(join(self.data_root, self.image_dir_name, image_name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(self.image, (1024,1024), interpolation=cv2.INTER_LINEAR) # face bbox is computed for 1024x1024
        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        input_image_tensor = input_image_tensor.squeeze(0).to(self.device)
        face_binmask_tensor = torch.tensor(self.face_binmask[None, :,:]).float().to(self.device)
        face_binmask_tensor = F.resize(face_binmask_tensor.unsqueeze(0), (256,256), torchvision.transforms.InterpolationMode.NEAREST).float()
        # face_binmask_tensor = face_binmask_tensor.unsqueeze(0)
        return input_image_tensor, face_binmask_tensor


class SAM_face:
    def __init__(self, ckpt_dir='/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/checkpoints', labels_definition_file_path='label_definition.json', task_name='ANIMSEG_E2E_FaceOnly_REAL7k_with_face_prior', device=torch.device('cuda:0') ):

        torch.manual_seed(999)
        np.random.seed(999)
        # torch.multiprocessing.set_start_method('spawn')

        self.labels_definition_file_path = labels_definition_file_path
        self.ckpt_dir = ckpt_dir
        self.task_name_face = task_name
        self.all_ckpts_dir = 'all_ckpts'
        self.mode = 'test'
        self.BATCH_SIZE = 1
        self.load_best_eval_ckpt = True
        self.epoch_face = 500
        self.encoder_original = False
        self.bbox_given = False
        self.visualize_coarse = True
        self.coarse_includes_neck = True
        self.bg_mask_given = False
        self.visualize_heatmap = False
        self.refine_masks = False
        self.model_type = 'vit_b'

        # semantic definitions
        self.num_classes_face = 11
        self.semantics = SemanticSegmentationFace(labels_definition_path=self.labels_definition_file_path, num_classes=self.num_classes_face)

        # model checkpoint directory
        self.model_load_path = join(self.ckpt_dir, self.task_name_face)
        assert os.path.exists(self.model_load_path), f"Model path {self.model_load_path} does not exist"

        # checkpoint name and path
        self.init_checkpoint_face = join(self.ckpt_dir, f'{self.task_name_face}/model_eval_best.pth')
        if not self.load_best_eval_ckpt:
            self.init_checkpoint_face = join(self.ckpt_dir, f'{self.task_name_face}/{self.all_ckpts_dir}/model_{self.epoch_face}.pth')

        self.device = device
        self.device_ids = [i for i in range(torch.cuda.device_count())]

        #prepare and load SAM face model
        self.sam_model_face = sam_model_registry[self.model_type](num_classes = self.num_classes_face, checkpoint=self.init_checkpoint_face).to(self.device)
        self.sam_model_face.image_encoder.to(self.device)
        self.sam_model_face.prompt_encoder.to(self.device)
        self.sam_model_face.prompt_encoder.parallel_training = True
        self.sam_model_face.mask_decoder.to(self.device)
    
        # parallelize components of SAM face model
        self.image_encoder_face = torch.nn.DataParallel(self.sam_model_face.image_encoder, device_ids=self.device_ids)
        self.prompt_encoder_face = torch.nn.DataParallel(self.sam_model_face.prompt_encoder, device_ids=self.device_ids)
        self.mask_decoder_face = torch.nn.DataParallel(self.sam_model_face.mask_decoder, device_ids=self.device_ids)

        # label id definitions
        self.label_to_id = self.semantics.data['label_name_to_id']
        self.id_to_label = {i:j for j,i in self.label_to_id.items()}

        # define differntiable non-learnable upsampling layer
        self.upsample = torch.nn.Upsample(scale_factor=4, mode='nearest')


    def predict(self, image, face_binmask):
        # create sample dataset
        image_dataset = DrawingsDatasetInference(self.sam_model_face, self.labels_definition_file_path, image, face_binmask, self.device)

        # assign semantics
        image_dataset.semantics = self.semantics

        # convert images to tensors
        image_tensor, face_binmask_tensor = image_dataset.get_input_tensors()

        #  predict facial details
        sparse_embeddings_face, dense_embeddings_face, image_pe_face = self.prompt_encoder_face(
        points=None,
        boxes=None,
        masks=face_binmask_tensor,
        )
        image_tensor = F.resize(image_tensor, (1024,1024), torchvision.transforms.InterpolationMode.BILINEAR) # encoder takes image size 1024x1024
        embedding_face = self.image_encoder_face(image_tensor.unsqueeze(0))
        mask_predictions_face, _ = self.mask_decoder_face(
            image_embeddings=embedding_face.to(self.device), # (B, 256, 64, 64)
            image_pe=image_pe_face, # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings_face, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings_face, # (B, 256, 64, 64)
            multimask_output=True,
        )
        mask_predictions_face = self.upsample(mask_predictions_face)

        # convert mask predictions to one-hot
        pred_labels = torch.argmax(mask_predictions_face, dim=1)
        pred_labels = pred_labels.squeeze(0).cpu().numpy()
        pred_segmap = self.semantics.labels_to_colors(pred_labels)

        return pred_segmap, pred_labels
