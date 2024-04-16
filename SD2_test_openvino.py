import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, PNDMScheduler, LMSDiscreteScheduler
from pathlib import Path
from implementation.conversion_helper_utils import convert_encoder, convert_unet, convert_vae_decoder, convert_vae_encoder
from openvino.runtime import Core
from transformers import CLIPTokenizer
from implementation.ov_stable_diffusion_pipeline import OVStableDiffusionPipeline
import time
from optimum.intel import OVStableDiffusionPipeline
import numpy as np

model_id = "stabilityai/stable-diffusion-2-1-base"

prompt = "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k"
neg_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"

pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
# Define the shapes related to the inputs and desired outputs
batch_size, num_images, height, width = 1, 1, 512, 512

# Statically reshape the model
pipeline.reshape(batch_size, height, width, num_images)

conf = pipeline.scheduler.config
scheduler = LMSDiscreteScheduler.from_config(conf)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

pipeline.scheduler = scheduler
pipeline.tokenizer = tokenizer

# Compile the model before inference
pipeline.compile()

image = pipeline(prompt, negative_prompt=neg_prompt, num_inference_steps=50, height=height, width=width).images[0] # for cache


time_lst = []

for i in range(10):
    np.random.seed(i)

    start = time.perf_counter()
    image = pipeline(prompt, negative_prompt=neg_prompt, num_inference_steps=50, height=height, width=width).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image.save("outputs/SD2_openvino" + str(i) + ".png")

with open("SD2_openvino.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")
