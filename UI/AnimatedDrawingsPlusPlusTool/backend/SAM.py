import sys

import torch
from PIL import Image
import requests
import os
import numpy as np
from transformers import SamModel, SamProcessor
import json
import cv2

from .repos.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from .repos.segment_anything.utils.transforms import ResizeLongestSide

# setup cache path for huggingface
os.environ['HF_HOME'] = "D:\\CACHE"
os.environ['HF_DATASETS_CACHE']="D:\\CACHE"
os.environ['TRANSFORMERS_CACHE']="D:\\CACHE"

# hugging face implementation of SAM
class SAM():
    def __init__(self, device="cuda"):
        # configuration
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = device
        self.repo_name = "facebook/sam-vit-huge"
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    def segment_with_points(self, input_image, input_points = [[[0, 0]]]):
        input_image_np = np.array(input_image)
        input_image_np = input_image_np[:,:,:3]
        #convert to PIL
        input_image_pil = Image.fromarray(input_image_np).convert("RGB")
        inputs = self.processor(input_image_pil, input_points=input_points, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks

    def segment_with_bbox(self, input_image, input_bbox = None):
        input_image_np = np.array(input_image)
        input_image_np = input_image_np[:,:,:3]
        #convert to PIL
        input_image_pil = Image.fromarray(input_image_np).convert("RGB")
        inputs = self.processor(input_image_pil, input_boxes=input_bbox,return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks


# custom implementation of multiclass-SAM
class SAM_custom():
    def __init__(self, original_ckpt_path, custom_ckpt_path, num_classes, label_definitions_path, model_type="vit_b", device="cuda"):
        # configuration
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = device
        self.model_type = model_type
        self.original_ckpt_path = original_ckpt_path
        self.custom_ckpt_path = custom_ckpt_path
        self.num_classes = num_classes
        self.label_definitions_path = label_definitions_path
        self.embedding_model = sam_model_registry[self.model_type](num_classes = self.num_classes, checkpoint=self.original_ckpt_path).to(self.device)
        self.decoding_model = sam_model_registry[self.model_type](num_classes = self.num_classes, checkpoint=self.custom_ckpt_path).to(self.device)
        # load label definitions
        self.label_definitons = None
        with open(self.label_definitions_path) as json_file:
            self.label_definitons = json.load(json_file)
        self.label_definitons['label_name_to_id']['CONFLICT']=26
        self.label_definitons['label_name_to_id']['Conflicted']=26
        self.label_definitons['label_name_to_id']['Unlabeled']=26
        self.color_dict = {}
        for label_name in self.label_definitons['label_name_to_color']:
            self.color_dict[int(self.label_definitons['label_name_to_id'][label_name])] = self.label_definitons['label_name_to_color'][label_name]
    
    def id_to_color(self, label_id):
        return self.color_dict[label_id]

    def labels_to_colors(self, img):
        w,h = img.shape[:2]
        img_rgb = np.zeros((w,h,3)).astype('uint8')
        for label_id in self.color_dict:
            img_rgb[img==label_id] = self.color_dict[label_id]
        return img_rgb

    def img_to_label_id(self, img, bgr_format=False):
        if bgr_format:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.reshape(-1,3)
        b = img[:,0]
        g = img[:,1]
        r = img[:,2]
        labels = np.zeros_like(r)
        for classname in self.label_definitons['label_name_to_color']:
            color = self.label_definitons['label_name_to_color'][classname]
            mask = (r==color[0]) & (g==color[1]) & (b==color[2])
            labels[mask] = int(self.label_definitons['label_name_to_id'][classname])
        labels = labels.reshape(1024,1024)
        return labels

    def segment_with_bbox(self, input_image, input_bbox):
        # pre-process image
        sam_transform = ResizeLongestSide(self.embedding_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(input_image)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        input_image_tensor = self.embedding_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        # pre-process bbox
        bbox = torch.from_numpy(np.array(input_bbox)).float().to(self.device)
        bbox = bbox//(self.embedding_model.image_encoder.img_size//256)
        bbox = bbox.unsqueeze(0)
        # inference
        with torch.no_grad():
            # compute the image embedding via original SAM
            img_embedding = self.embedding_model.image_encoder(input_image_tensor)
            img_embedding = img_embedding.to(self.device)
            # compute the prompt embedding via cutom decoding model
            sparse_embeddings, dense_embeddings = self.decoding_model.prompt_encoder(
                points=None,
                boxes=bbox[:, None, :], # (B, 4) -> (B, 1, 4)
                masks=None,
            )
            # predict masks via custom decoder
            seg_prob, _ = self.decoding_model.mask_decoder(
                image_embeddings=img_embedding, # (B, 256, 64, 64)
                image_pe=self.decoding_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
                )
            seg_prob = torch.sigmoid(seg_prob)
            # convert soft mask to hard mask
            seg_prob = seg_prob.cpu().numpy().squeeze()
            seg = (seg_prob > 0.5).astype(np.uint8)
            seg_out = np.transpose(seg,(1,2,0))
            labels_out = torch.argmax(torch.Tensor(seg_out), dim=2)
            seg_map = labels_out.numpy().astype('uint8')
            seg_map = cv2.resize(seg_map, (1024,1024), interpolation=cv2.INTER_NEAREST)
            seg_map_colors = self.labels_to_colors(seg_map)
            return seg_map, seg_map_colors
        

# SAM with automatic mask generation
class SAM_automatic():
    def __init__(self, device="cuda", num_classes=26):
        # configuration
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = device
        self.num_classes = num_classes
        self.ckpt = "sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.sam = sam_model_registry[self.model_type](num_classes = self.num_classes, checkpoint=self.ckpt)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def automatic_segmentation(self, image):
        masks = self.mask_generator.generate(image)
        return masks
