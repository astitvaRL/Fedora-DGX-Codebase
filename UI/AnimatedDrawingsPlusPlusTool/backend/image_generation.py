import os
import cv2
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image
from PIL import Image,ImageOps
from controlnet_aux.canny import CannyDetector
from huggingface_hub import hf_hub_download
import torch
import json

# setup cache path for huggingface
os.environ['HF_HOME'] = "D:\\CACHE"
os.environ['HF_DATASETS_CACHE']="D:\\CACHE"
os.environ['TRANSFORMERS_CACHE']="D:\\CACHE"


class DMD2_T2IAdapt():
    def __init__(self, device="cuda"):
        # configuration
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = device
        self.weight_type = "fp16"
        self.adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient=self.weight_type).to(self.device)
        self.vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.repo_name = "tianweiy/DMD2"
        self.ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

        # init models
        self.unet = UNet2DConditionModel.from_config(self.base_model_id, subfolder="unet").to(self.device, torch.float16)
        self.unet.load_state_dict(torch.load(hf_hub_download(self.repo_name, self.ckpt_name), map_location=self.device))
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            self.base_model_id, unet=self.unet, vae=self.vae, adapter=self.adapter, torch_dtype=torch.float16, variant=self.weight_type, 
        ).to(self.device)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.preprocessor = CannyDetector()

    def generate_from_canny(self, input_image, text_prompt, num_inference_steps=4, guidance_scale=0, adapter_conditioning_scale=1.0, adapter_conditioning_factor=1.0):
        condition_image_np = np.array(input_image)
        condition_image_np = condition_image_np[:,:,:3]
        condition_image_np = cv2.resize(condition_image_np, (1024, 1024))
        #convert to PIL
        condition_image = Image.fromarray(condition_image_np).convert("RGB")
        processed_image = self.preprocessor(condition_image, detect_resolution=384, image_resolution=1024)
        gen_img = self.pipe(prompt=text_prompt, image=processed_image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, adapter_conditioning_scale=adapter_conditioning_scale, adapter_conditioning_factor=adapter_conditioning_factor, timesteps=[999, 749, 499, 249]).images[0]
        display_image = gen_img.resize((512, 512))
        print("Returning")
        return display_image
