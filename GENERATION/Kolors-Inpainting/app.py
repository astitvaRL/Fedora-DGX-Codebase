import os
# setup cache path for huggingface
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HOME'] = os.environ['CACHE_DIR']
os.environ['HF_DATASETS_CACHE'] = os.environ['CACHE_DIR']
os.environ['TRANSFORMERS_CACHE']= os.environ['CACHE_DIR']

print('HF_HOME',os.environ['HF_HOME'])
print('HF_DATASETS_CACHE',os.environ['HF_DATASETS_CACHE'])
print('TRANSFORMERS_CACHE',os.environ['TRANSFORMERS_CACHE'])

import spaces
import random
import torch
from huggingface_hub import snapshot_download
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
import gradio as gr
import numpy as np

device = "cuda:0"
ckpt_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors-Inpainting")
text_encoder = ChatGLMModel.from_pretrained(f'{ckpt_dir}/text_encoder',torch_dtype=torch.float16).half().to(device)
tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half().to(device)
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half().to(device)

pipe = StableDiffusionXLInpaintPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler
)
    
pipe.to(device)
pipe.enable_attention_slicing()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

@spaces.GPU
def infer(prompt, 
          image,
          mask_image = None,
          negative_prompt = "", 
          seed = 0, 
          randomize_seed = False, 
          guidance_scale = 6.0, 
          num_inference_steps = 25
          ):
    if not isinstance(image, dict):
        image = dict({'background': image, 'layers': [mask_image]})
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    width, height = image['background'].size
    width = (width // 8 + 1) * 8
    height = (height // 8 + 1) * 8
    result = pipe(
        prompt = prompt,
        image = image['background'],
        mask_image = image['layers'][0],
        height=height,
        width=width,
        guidance_scale = guidance_scale,
        generator= generator,
        num_inference_steps= num_inference_steps,
        negative_prompt = negative_prompt,
        num_images_per_prompt = 1,
        strength = 0.999
    ).images[0]

    return result

examples = [
    ["一只带着红色帽子的小猫咪，圆脸，大眼，极度可爱，高饱和度，立体，柔和的光线", 
     "image/1.png", "image/1_masked.png"],
    ["这是一幅令人垂涎欲滴的火锅画面，各种美味的食材在翻滚的锅中煮着，散发出的热气和香气令人陶醉。火红的辣椒和鲜艳的辣椒油熠熠生辉，具有诱人的招人入胜之色彩。锅内肉质细腻的薄切牛肉、爽口的豆腐皮、鲍汁浓郁的金针菇、爽脆的蔬菜，融合在一起，营造出五彩斑斓的视觉呈现", 
     "image/2.png", "image/2_masked.png"],
    ["穿着美少女战士的衣服，一件类似于水手服风格的衣服，包括一个白色紧身上衣，前胸搭配一个大大的红色蝴蝶结。衣服的领子部分呈蓝色，并且有白色条纹。她还穿着一条蓝色百褶裙，超高清，辛烷渲染，高级质感，32k，高分辨率，最好的质量，超级细节，景深", 
     "image/3.png", "image/3_masked.png"],
    ["穿着钢铁侠的衣服，高科技盔甲，主要颜色为红色和金色，并且有一些银色装饰。胸前有一个亮起的圆形反应堆装置，充满了未来科技感。超清晰，高质量，超逼真，高分辨率，最好的质量，超级细节，景深", 
     "image/4.png", "image/4_masked.png"],
]

css="""
#col-left {
    margin: 0 auto;
    max-width: 600px;
}
#col-right {
    margin: 0 auto;
    max-width: 700px;
}
"""

def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

with gr.Blocks(css=css) as Kolors:
    gr.HTML(load_description("assets/title.md"))
    with gr.Row():
        with gr.Column(elem_id="col-left"):
            with gr.Row():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt",
                    lines=2
                )
            with gr.Row():
                image = gr.ImageEditor(label='Image', type='pil', sources=["upload", "webcam"], image_mode='RGB', layers=False, brush=gr.Brush(colors=["#AAAAAA"], color_mode="fixed"))
                mask_image = gr.Image(label='Mask_Example',type='pil', visible=False, value=None)
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    placeholder="Enter a negative prompt",
                    value='残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量'
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=6.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=25,
                    )
            with gr.Row():
                run_button = gr.Button("Run")
            
        with gr.Column(elem_id="col-right"):
            result = gr.Image(label="Result", show_label=False)
    
    with gr.Row():
        gr.Examples(
                fn = infer,
                examples = examples,
                inputs = [prompt, image, mask_image],
                outputs = [result]
            )

    run_button.click(
        fn = infer,
        inputs = [prompt, image, mask_image, negative_prompt, seed, randomize_seed, guidance_scale, num_inference_steps],
        outputs = [result]
    )

Kolors.queue().launch(debug=True, share=True)
