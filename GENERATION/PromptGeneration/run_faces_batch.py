import os
import json
from tqdm import tqdm

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

model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "In less than 60 words, write comma seperated keywords intricately describing the face in this drawing. Write keywords to describe detailed facial expression, drawing characteristics, colors, art-style, background etc. DON'T PRINT ANYTHIN ELSE, EXCEPT the text prompt."}
    ]}
]

IMG_DIR = '/mnt/users_scratch/astitva/DATA/FACES_DRAWINGS16k/images/'

images = sorted(os.listdir(IMG_DIR))
generated_prompts = {}
for image_name in tqdm(images):
    image = Image.open(f"{IMG_DIR}/{image_name}")
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=75)
    text_output_raw = processor.decode(output[0])
    text_output = text_output_raw.split('<|end_header_id|>')[-1][:-len("<|eot_id|>")].split('\n')[-1]
    generated_prompts[image_name] = text_output
    print(text_output)

    json_string = json.dumps(generated_prompts, indent=4)
    with open("/mnt/users_scratch/astitva/DATA/FACES_DRAWINGS16k/drawings_descriptions_faces_16k.json", "w") as f:
        f.write(json_string)
