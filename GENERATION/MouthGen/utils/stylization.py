import os
os.environ['HF_HUB_OFFLINE'] = '1'

from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
import cv2
from PIL import Image

from .external.ControlStyleAligned import inversion, pipeline_calls, sa_handler

class ControllableStylization:
    def __init__(self, device=torch.device('cuda:0')):
        self.device = device
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False)
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
            scheduler=self.scheduler
        ).to(self.device)
        # style aligned args
        self.sa_args = sa_handler.StyleAlignedArgs(share_group_norm=True,
                                            share_layer_norm=True,
                                            share_attention=True,
                                            adain_queries=True,
                                            adain_keys=True,
                                            adain_values=False,
                                            shared_score_shift=np.log(4),
                                            shared_score_scale=1.0,
                                            )
        # create handler
        self.handler = sa_handler.Handler(self.pipeline)
        self.handler.register(self.sa_args, )

    

    def generate(self, reference_image, condition_image, reference_prompt, style_prompt, target_prompt, preprocessor='canny', num_inference_steps=50, guidance_scale=7.5):
        assert reference_image.shape[0] == 1024 and reference_image.shape[1] == 1024
        reference_prompt = f"{reference_prompt}, {style_prompt}."
        target_prompt = f"{target_prompt}, {style_prompt}."
        canny_image = cv2.Canny(condition_image, 5, 50)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = Image.fromarray(canny_image).resize((1024, 1024), 0)
        num_images_per_prompt = 1
        latents = torch.randn(1+num_images_per_prompt, 4, 128, 128, dtype=self.pipeline.unet.dtype,).to(self.device) 
        # run inversion
        x0 = reference_image
        zts = inversion.ddim_inversion(self.pipeline, x0, reference_prompt, num_inference_steps, 2)
        zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)
        latents[0] = zT
        # run pipeline
        images = self.pipeline([reference_prompt, target_prompt],
                latents=latents,
                image=canny_image,
                controlnet_conditioning_scale=0.99,
                callback_on_step_end=inversion_callback,
                num_inference_steps=num_inference_steps,
                guidance_scale=10).images
        generated = images[-1]
        return generated, canny_image
