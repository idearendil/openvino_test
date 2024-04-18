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

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet = torch.compile(pipe.unet, backend="aot_eager")
pipe.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
pipe = pipe.to("cpu")
# pipe.enable_model_cpu_offload()

prompt = "Dancing Darth Vader, best quality, extremely detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

original_image = load_image("test_pose_src.jpg")
pose_image = pose_estimator(original_image)

time_lst = []
output = pipe(prompt, image=pose_image, num_inference_steps=20, negative_prompt=negative_prompt).images[0]  # for cache

for i in range(10):
    idx = i % 2
    generator = torch.Generator("cpu").manual_seed(i)

    start = time.perf_counter()
    output = pipe(prompt, image=pose_image, num_inference_steps=20, negative_prompt=negative_prompt).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image = make_image_grid([original_image, pose_image, output], rows=1, cols=3)
    image.save("outputs/CN_cpu" + str(i) + ".png")

with open("CN_cpu.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")