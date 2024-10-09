import os
# setup cache path for huggingface
os.environ['CACHE_DIR'] = '/mnt/users_scratch/astitva/CACHE/'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HOME'] = os.environ['CACHE_DIR']
os.environ['HF_DATASETS_CACHE'] = os.environ['CACHE_DIR']
os.environ['TRANSFORMERS_CACHE']= os.environ['CACHE_DIR']

print('HF_HOME',os.environ['HF_HOME'])
print('HF_DATASETS_CACHE',os.environ['HF_DATASETS_CACHE'])
print('TRANSFORMERS_CACHE',os.environ['TRANSFORMERS_CACHE'])

from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
import mediapy
import sa_handler
import pipeline_calls

import inversion

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
# pipeline.enable_model_cpu_offload()

# inversion pipeline
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False)
inversion_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True,
    scheduler=scheduler
).to("cuda")

sa_args = sa_handler.StyleAlignedArgs(share_group_norm=False,
                                      share_layer_norm=False,
                                      share_attention=True,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=False,
                                     )
handler = sa_handler.Handler(pipeline)
handler.register(sa_args, )


ref_image = load_image("./example_image/medieval-bed.jpeg")
ref_prompt = "a flat design poster"
num_inference_steps = 50
image_inversion = False
num_images_per_prompt = 1

cond_image = load_image("./example_image/monster.jpg").resize((1024, 1024))
depth_image = pipeline_calls.get_depth_map(cond_image, feature_processor, depth_estimator)
target_prompt = "a cartoon drawing of a funny monster"

# initialize random latents
g_cpu = torch.Generator(device='cpu')
g_cpu.manual_seed(999)
latents = torch.randn(1+num_images_per_prompt, 4, 128, 128, device='cpu', generator=g_cpu,
                      dtype=pipeline.unet.dtype,).to('cuda:0') 

if image_inversion:
    # latent inversion
    print("Running inversion...")
    x0 = np.array(ref_image.resize((1024, 1024)))
    zts = inversion.ddim_inversion(inversion_pipeline, x0, ref_prompt, num_inference_steps, 2)
    zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)
    latents[1] = zT # 0 is reference, 1 is target

controlnet_conditioning_scale = 0.9
print("Generating...")
images = pipeline_calls.controlnet_call(pipeline, [ref_prompt, target_prompt],
                                        image=depth_image,
                                        num_inference_steps=50,
                                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                                        num_images_per_prompt=1,
                                        latents=latents)

images[1].save("./example_image/generated.png")
