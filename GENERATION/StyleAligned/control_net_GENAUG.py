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

from tqdm import tqdm
import cv2
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import json

join = os.path.join

# # load stylization model
# control_stylization = ControllableStylization()

DATA_ROOT = '/mnt/users_scratch/astitva/WORKSPACE/Fedora-DGX-Codebase/SEGMENTATION/SAM_finetuning/DOG_DATASET'

# DRAWINGS_ROOT = "/mnt/users_scratch/astitva/DATA/MANIFOLD/animated_drawings_images_prior_april22/cropped_image"
DRAWINGS_ROOT = join(DATA_ROOT, 'train_images/')
# SEG_ROOT = "/mnt/users_scratch/astitva/DATA/AD_SegMaps/labels_7k_1024"
SEG_ROOT = join(DATA_ROOT, 'train_segmentations/')
# SAVE_ROOT = "/mnt/users_scratch/astitva/DATA/MANIFOLD/animated_drawings_images_prior_april22/cropped_image_GENAUG10k"
SAVE_ROOT = join(DATA_ROOT, 'train_images_GENAUG5k/')

os.makedirs(SAVE_ROOT, exist_ok=True)

LABELS_SAMPLED_NUM = -1
labels = sorted(os.listdir(SEG_ROOT))
if LABELS_SAMPLED_NUM>-1:
    labels = labels[:LABELS_SAMPLED_NUM]


controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe = pipe.to('cuda')

GEN_PER_LABEL = 5

with open("./generated_random_prompts.json", "r") as f:
    generated_prompts = json.load(f)

for label in tqdm(labels):
    if label.startswith('annotations'):
        continue
    # image_name = label.split('_')[0]
    image_name = label.split('.')[0]
    img=None
    try:
        img = cv2.imread(f'{DRAWINGS_ROOT}/{image_name}.jpg')
        w,h,_ = img.shape
    except:
        print("Image not found!")
        continue
    img = img[:,:,:3]
    w,h,_ = img.shape
    img = cv2.resize(img, (1024,1024))
    canny_img = cv2.Canny(img, 5, 100)
    canny_img = canny_img[:, :, None]
    canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)
    canny_img = Image.fromarray(canny_img)
    for save_idx in tqdm(range(GEN_PER_LABEL)):
        save_path = f"{SAVE_ROOT}/{image_name}_GENAUG{save_idx}.png"
        prompt_idx = np.random.randint(0, len(generated_prompts))
        random_prompt = generated_prompts[str(prompt_idx)]
        images = pipe(random_prompt, negative_prompt='shadows, photoreal, human', image=canny_img, num_inference_steps=30, controlnet_conditioning_scale=0.97,).images
        generated = images[0]
        generated.save(save_path)