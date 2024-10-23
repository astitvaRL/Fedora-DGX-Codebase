import os

# setup cache path for huggingface
os.environ["CACHE_DIR"] = "/mnt/users_scratch/astitva/CACHE/"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = os.environ["CACHE_DIR"]
os.environ["HF_DATASETS_CACHE"] = os.environ["CACHE_DIR"]
os.environ["TRANSFORMERS_CACHE"] = os.environ["CACHE_DIR"]

print("HF_HOME", os.environ["HF_HOME"])
print("HF_DATASETS_CACHE", os.environ["HF_DATASETS_CACHE"])
print("TRANSFORMERS_CACHE", os.environ["TRANSFORMERS_CACHE"])

import cv2
import inversion
import mediapy
import numpy as np
import pipeline_calls

import sa_handler
import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForInpainting,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
from PIL import Image, ImageOps
from transformers import DPTForDepthEstimation, DPTImageProcessor

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(
    "cuda"
)
feature_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")


scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)

pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
    scheduler=scheduler,
).to("cuda")

inversion_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
    vae = vae,
    use_safetensors=True,
    scheduler=scheduler
).to("cuda")


shared_score_shift = np.log(4)
shared_score_scale = 1.0
sa_args = sa_handler.StyleAlignedArgs(
    share_group_norm=True,
    share_layer_norm=True,
    share_attention=True,
    adain_queries=True,
    adain_keys=True,
    adain_values=False,
    shared_score_shift=shared_score_shift,
    shared_score_scale=shared_score_scale,
)
handler = sa_handler.Handler(pipeline)
handler.register(
    sa_args,
)

ref_image_path = "example_image/image.png"
ref_image = load_image(ref_image_path).resize((1024, 1024))
ref_style = "hand-drawn"
ref_prompt = f"2D character, {ref_style}."
num_inference_steps = 25
image_inversion = True
num_images_per_prompt = 1

mask_path = "example_image/mask.png"
mask = load_image(mask_path).resize((1024, 1024))
mask = ImageOps.invert(mask)

# initialize random latents
# g_cpu = torch.Generator(device='cpu')
# g_cpu.manual_seed(999)
latents = torch.randn(
    1 + num_images_per_prompt,
    4,
    128,
    128,
    dtype=inversion_pipeline.unet.dtype,
).to("cuda:0")

if image_inversion:
    # latent inversion
    print("Running inversion...")
    x0 = np.array(ref_image.resize((1024, 1024)))
    zts = inversion.ddim_inversion(inversion_pipeline, x0, ref_prompt, num_inference_steps, 2)
    zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)
    latents[0] = zT

target_prompt = f"2D character with an open mouth, {ref_style}."
control_strength = 0.99
guidance = 10
while True:
    print("Generating...")
    # images = pipeline_calls.controlnet_call(pipeline, [ref_prompt, target_prompt],
    #                                         image=canny_image,
    #                                         num_inference_steps=50,
    #                                         controlnet_conditioning_scale=controlnet_conditioning_scale,
    #                                         num_images_per_prompt=1,
    #                                         latents=latents)
    # images = pipeline(
    #     [ref_prompt, target_prompt],
    #     latents=latents,
    #     image=canny_image,
    #     controlnet_conditioning_scale=control_strength,
    #     callback_on_step_end=inversion_callback,
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=guidance,
    # ).images
    out = pipeline(prompt = [ref_prompt, target_prompt], latents=latents, negative_prompt=['',''], image=ref_image, mask_image=mask, guidance_scale=7.5, num_inference_steps=num_inference_steps, strength=0.99)
    breakpoint()

    out.images[1].resize(ref_image.size).save(f"{mask_path[:-4]}_stylized.png")
