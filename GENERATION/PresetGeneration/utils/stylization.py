import os

os.environ["HF_HUB_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

from .external.ControlStyleAligned import inversion, pipeline_calls, sa_handler


class ControllableStylization:
    def __init__(self, device=torch.device("cuda:0")):
        self.device = device
        self.reference_latent = None
        self.inversion_callback = None
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to(self.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
            scheduler=self.scheduler,
        ).to(self.device)
        # style aligned args
        self.sa_args = sa_handler.StyleAlignedArgs(
            share_group_norm=True,
            share_layer_norm=True,
            share_attention=True,
            adain_queries=True,
            adain_keys=True,
            adain_values=False,
            shared_score_shift=np.log(2),
            shared_score_scale=1.0,
        )
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas"
        ).to("cuda")
        self.feature_processor = DPTImageProcessor.from_pretrained(
            "Intel/dpt-hybrid-midas"
        )
        # create handler
        self.handler = sa_handler.Handler(self.pipeline)
        self.handler.register(
            self.sa_args,
        )

    def generate(
        self,
        reference_image,
        condition_image,
        reference_prompt,
        style_prompt,
        target_prompt,
        preprocessor="canny",
        num_inference_steps=25,
        guidance_scale=7.5,
    ):
        assert reference_image.shape[0] == 1024 and reference_image.shape[1] == 1024
        reference_prompt = f"{reference_prompt}, {style_prompt}."
        target_prompt = f"{target_prompt}, {style_prompt}."

        proc_cond_image = None
        if preprocessor == "canny":
            canny_image = cv2.Canny(condition_image, 5, 45)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate(
                [canny_image, canny_image, canny_image], axis=2
            )
            proc_cond_image = Image.fromarray(canny_image).resize((1024, 1024), 0)
        elif preprocessor == "depth":
            proc_cond_image = pipeline_calls.get_depth_map(
                Image.fromarray(condition_image),
                self.feature_processor,
                self.depth_estimator,
            )

        num_images_per_prompt = 1
        latents = torch.randn(
            1 + num_images_per_prompt,
            4,
            128,
            128,
            dtype=self.pipeline.unet.dtype,
        ).to(self.device)
        # run inversion
        if self.reference_latent is None or self.inversion_callback is None:
            print("Running inversion...")
            x0 = reference_image
            zts = inversion.ddim_inversion(
                self.pipeline, x0, reference_prompt, num_inference_steps, 2
            )
            zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)
            latents[0] = zT
            self.reference_latent = zT
            self.inversion_callback = inversion_callback
        else:
            latents[0] = self.reference_latent
        # run pipeline
        print("Generating with style...")
        images = self.pipeline(
            [reference_prompt, target_prompt],
            latents=latents,
            image=proc_cond_image,
            controlnet_conditioning_scale=0.9,
            callback_on_step_end=self.inversion_callback,
            num_inference_steps=num_inference_steps,
            guidance_scale=15,
        ).images
        generated = images[-1]
        return generated, proc_cond_image
