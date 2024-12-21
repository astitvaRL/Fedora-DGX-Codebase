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

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# from huggingface_hub import login
# login(token='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')

model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# url = "https://img.freepik.com/premium-photo/intricate-landscape-wallpaper-with-vibrant-night-colors_936668-977.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

image_path = "/mnt/users_scratch/astitva/DATA/MANIFOLD/animated_drawings_images_prior_april22/cropped_faces/0d400067e05e4aae8027f7cf73147e0d__1742984519568414_1024.png"
image = Image.open(image_path)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Write a text prompt in not more than 50 words for the image, this text prompt would be feeded to stable diffusion to generate the image, write like a prompt-engineer. DON'T PRINT ANYTHIN ELSE, EXCEPT the text prompt."}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=75)
print(processor.decode(output[0]))
