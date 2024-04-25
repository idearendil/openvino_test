from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, LMSDiscreteScheduler
import torch
from transformers import CLIPTokenizer
import time
from controlnet_aux import OpenposeDetector
import matplotlib.pyplot as plt
import requests

model_id = "runwayml/stable-diffusion-v1-5"

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32, use_safetensors=False)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=False
)
pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.unet = torch.compile(pipe.unet, backend="aot_eager")
pipe.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
pipe = pipe.to("cpu")
# pipe.enable_model_cpu_offload()

prompt_lst = [
    "Dancing Darth Vader, best quality, extremely detailed, vivid, sharp, clear, detailed, vibrant, rich, sophisticated, balanced, dynamic, realistic",
    "A Marine soldier sitting in a chair, best quality, extremely detailed, vivid, sharp, clear, detailed, vibrant, rich, sophisticated, balanced, dynamic, realistic",
    "A figure skater in performance, best quality, extremely detailed, vivid, sharp, clear, detailed, vibrant, rich, sophisticated, balanced, dynamic, realistic"
]
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, strange face, monochrome, lowres, worst quality, low quality, blurry, fuzzy, grainy, pixelated, distorted, dull, flat, muddy, washed-out, low-resolution, unfocused, hazy, rough, jagged, patchy, overexposed, underexposed, noisy, smeared, unrefined"

original_image_lst = [
    load_image("test_pose_src.jpg"),
    load_image("test_pose_src2.jpg"),
    load_image("test_pose_src3.jpg")
]
pose_image_lst = [
    pose_estimator(original_image_lst[0]),
    pose_estimator(original_image_lst[1]),
    pose_estimator(original_image_lst[2])
]

time_lst = []
output = pipe(prompt_lst[0], image=pose_image_lst[0], num_inference_steps=50, negative_prompt=negative_prompt).images[0]  # for cache

for i in range(10):
    generator = torch.Generator("cpu").manual_seed(i)
    start = time.perf_counter()
    output = pipe(prompt_lst[i % 3], image=pose_image_lst[i % 3], num_inference_steps=50, negative_prompt=negative_prompt, generator=generator).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image = make_image_grid([original_image_lst[i % 3], pose_image_lst[i % 3], output], rows=1, cols=3)
    image.save("outputs/CN_cpu" + str(i) + ".png")

with open("CN_cpu.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")