import os
import json
from tqdm import tqdm
import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt


DATA_DIR = '/mnt/users_scratch/astitva/DATA/MANIFOLD/animated_drawings_images_prior_april22/cropped_image/'

SAVE_DIR = './VIS/'
os.makedirs(SAVE_DIR, exist_ok=True)

images = sorted(os.listdir(DATA_DIR))[:100]

with open("drawings_descriptions.json", 'r') as f:
    prompts = json.load(f)
keys = [k for k in prompts.keys()]

for key in tqdm(keys):
    # image_name = key.split('_')[0]
    image = Image.open(f"{DATA_DIR}/{key}")
    description  = prompts[key]
    w,h = image.size
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")
    ax.text(w/2,-5, description, ha='center', wrap=True, fontsize=12.0)
    plt.savefig(f'{SAVE_DIR}/{key}')
    plt.close()

