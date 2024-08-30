import os
# setup cache path for huggingface
os.environ['HF_HUB_OFFLINE'] = True
os.environ['HF_HOME'] = os.environ['CACHE_DIR']
os.environ['HF_DATASETS_CACHE'] = os.environ['CACHE_DIR']
os.environ['TRANSFORMERS_CACHE']= os.environ['CACHE_DIR']

print('HF_HOME',os.environ['HF_HOME'])
print('HF_DATASETS_CACHE',os.environ['HF_DATASETS_CACHE'])
print('TRANSFORMERS_CACHE',os.environ['TRANSFORMERS_CACHE'])

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

# paths
ROOT_DIR = "/mnt/users_scratch/astitva/DATA/"
drawings_dir = os.path.join(ROOT_DIR, "drawings_resized")
labels_dir = os.path.join(ROOT_DIR, "labels_resized")
prompt_file_path = "prompts.txt"

# output directory
output_dir = os.path.join(ROOT_DIR, "drawings_resized_synth_dgx")
os.makedirs(output_dir, exist_ok=True)

# taken from https://huggingface.co/tianweiy/DMD2

# configuration
device = "cuda"
weight_type = "fp16"
torch.backends.cuda.matmul.allow_tf32 = True
adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient=weight_type).to(device)
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

# init models
unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device))
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    base_model_id, unet=unet, vae=vae, adapter=adapter, torch_dtype=torch.float16, variant="fp16", 
).to(device)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
preprocessor = CannyDetector()

# load the prompts
with open(prompt_file_path, "r") as f:
    prompts = f.readlines()
prompts = [p.strip() for p in prompts]
# remove empty lines
processed_prompts = []
for p in prompts:
    if p:
        processed_prompts.append(p)

# load the labels
NUM_IMAGES = 10
labels = sorted(os.listdir(labels_dir))
prompts_dict = {}
for label_name in tqdm(labels):
    # preprocess the label image to get conditioning
    label_img = Image.open(os.path.join(labels_dir, label_name)).convert("RGB")
    label_img_np = np.array(label_img)
    mask = label_img_np.sum(2)
    binmask = mask == 0
    cond_img = preprocessor(label_img, detect_resolution=384, image_resolution=1024)
    # randomly select NUM_IMAGES unique prompts
    sampled_prompts = np.random.choice(processed_prompts, NUM_IMAGES, replace=False)
    for idx in range(NUM_IMAGES):
        # text_prompt = f"an image of a drawing of an animated cartoon character."
        text_prompt = sampled_prompts[idx]
        if np.random.rand() < 0.5:
            text_prompt += f", smudgy, colors leaking from the edges, hand drawn on a white paper."
        else:
            text_prompt = f"a line-art sketch" + text_prompt
            text_prompt += ", solid white background"
        # generate image
        gen_img = pipe(prompt=text_prompt, image=cond_img, num_inference_steps=4, guidance_scale=0, adapter_conditioning_scale=1.0, adapter_conditioning_factor=1.0, timesteps=[999, 749, 499, 249]).images[0]
        #mask the image
        gen_img_np = np.array(gen_img)
        gen_img_np[binmask] = [255, 255, 255]
        gen_img = Image.fromarray(gen_img_np)
        # save image
        gen_img.save(os.path.join(output_dir, f"{label_name[:-4]}_{idx}.png"))
        # save the prompts as json dict 
        prompts_dict[f"{label_name[:-4]}_{idx}"] = text_prompt
    # save the prompts
    with open(os.path.join(output_dir, "synth_prompts.json"), "w" ) as f:
        json.dump(prompts_dict, f, indent=4)
