import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision
from torchvision import transforms

class RandomAug:
    def __init__(self, target_size=(1024,1024), crop_size=(512,512), crop_probability=0.3, fliph_probability=0.7, flipv_probability=0.0, rotate_probability=0.5, rotate_max_angle=30, color_jitter_probability=0.5, noise_probability=0.4):
        self.target_size = target_size
        self.crop_size = crop_size
        self.crop_probability = crop_probability
        self.fliph_probability = fliph_probability
        self.flipv_probability = flipv_probability
        self.rotate_probability = rotate_probability
        self.rotate_max_angle = rotate_max_angle
        self.color_jitter_probability = color_jitter_probability
        self.noise_probability = noise_probability
        self.crop = torchvision.transforms.RandomCrop(self.crop_size)
        self.fliph = torchvision.transforms.RandomHorizontalFlip(p=self.fliph_probability)
        self.flipv = torchvision.transforms.RandomVerticalFlip(p=self.flipv_probability)
        self.rotate = torchvision.transforms.RandomRotation(degrees=self.rotate_max_angle, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.color_jitter = torchvision.transforms.ColorJitter(hue=0.5)

    def apply_augmentation(self, image, segmap, bg_mask, bbox=None):
        input_all = torch.cat([image, segmap, bg_mask], axis=1)
        if np.random.uniform(0,1)<self.crop_probability:
            # input_all = self.crop(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.crop(x_) for x_ in x]))(input_all)
        if np.random.uniform(0,1)<self.fliph_probability:               
            # input_all = self.fliph(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.fliph(x_) for x_ in x]))(input_all)
        if np.random.uniform(0,1)<self.flipv_probability:
            # input_all = self.flipv(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.flipv(x_) for x_ in x]))(input_all)
        if np.random.uniform(0,1)<self.rotate_probability:
            # input_all = self.rotate(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.rotate(x_) for x_ in x]))(input_all)
        image = input_all[:,:3,:,:]
        segmap = input_all[:,3:4,:,:]
        bg_mask = input_all[:,4:,:,:]
        return image, segmap, bg_mask, bbox
        
    def apply_augmentation_list(self, input_list, bbox=None):
        input_all = torch.cat(input_list, axis=1)
        if np.random.uniform(0,1)<self.crop_probability:
            # input_all = self.crop(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.crop(x_) for x_ in x]))(input_all)
        if np.random.uniform(0,1)<self.fliph_probability:               
            # input_all = self.fliph(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.fliph(x_) for x_ in x]))(input_all)
        if np.random.uniform(0,1)<self.flipv_probability:
            # input_all = self.flipv(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.flipv(x_) for x_ in x]))(input_all)
        if np.random.uniform(0,1)<self.rotate_probability:
            # input_all = self.rotate(input_all)
            input_all = transforms.Lambda(lambda x: torch.stack([self.rotate(x_) for x_ in x]))(input_all)
        return input_all

    def apply_color_jitter(self, image, segmap, num_jitters=5):
        if np.random.uniform(0,1)<self.color_jitter_probability:
            ids = torch.unique(segmap)
            random_ids = ids[torch.randperm(len(ids))][:num_jitters]
            for rid in random_ids:
                mask = segmap==rid
                mask = torch.repeat_interleave(mask, 3, dim=1)
                # jittered = self.color_jitter(image)
                jittered = transforms.Lambda(lambda x: self.color_jitter(x))(image)
                image[mask] = jittered[mask]
        return image

    def apply_noise(self, segmap, num_classes):
        if np.random.uniform(0,1)<self.noise_probability:
            segmap = segmap.long().squeeze(1)
            segmap = torch.nn.functional.one_hot(segmap,num_classes).float()
            noise = torch.randn_like(segmap) * np.random.uniform(0.2,0.6)
            segmap += noise
            segmap = torch.argmax(segmap,dim=-1).unsqueeze(1).float()
        return segmap
