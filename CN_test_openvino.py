from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, LMSDiscreteScheduler
import torch
from transformers import CLIPTokenizer
from pathlib import Path
import openvino as ov
from CN_conversion import controlnet_conversion, unet_conversion, text_encoder_conversion, vae_decoder_conversion
from CN_OVpipeline import OVContrlNetStableDiffusionPipeline
import time

model_id = "stabilityai/stable-diffusion-2-1-base"
CONTROLNET_OV_PATH = Path('./cn_openvino/controlnet.xml')
UNET_OV_PATH = Path('./cn_openvino/unet_controlnet.xml')
TEXT_ENCODER_OV_PATH = Path('./cn_openvino/text_encoder.xml')
VAE_DECODER_OV_PATH = Path('./cn_openvino/vae_decoder.xml')

original_image = [load_image("test_cat.png"), load_image("test_bird.png")]
image = np.array(original_image[0])
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
image = np.array(original_image[1])
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = [canny_image, Image.fromarray(image)]

controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers", torch_dtype=torch.float32, use_safetensors=False)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=False
)

a, b = controlnet_conversion(controlnet)
unet_conversion(pipe, a, b)
text_encoder_conversion(pipe.text_encoder)
vae_decoder_conversion(pipe.vae)

scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

core = ov.Core()
ov_pipe = OVContrlNetStableDiffusionPipeline(tokenizer, scheduler, core, CONTROLNET_OV_PATH, TEXT_ENCODER_OV_PATH, UNET_OV_PATH, VAE_DECODER_OV_PATH, device='CPU')

prompt = ["a black cat", "a yellow bird"]
negative_prompt = ""


time_lst = []
result = ov_pipe(prompt[0], canny_image[0], num_inference_steps=20, negative_prompt=negative_prompt)  # for cache

for i in range(10):
    idx = i % 2
    np.random.seed(i)

    start = time.perf_counter()
    result = ov_pipe(prompt[idx], canny_image[idx], num_inference_steps=20, negative_prompt=negative_prompt)
    end = time.perf_counter()
    time_lst.append(end - start)

    image = make_image_grid([original_image[idx], canny_image[idx], result[0]], rows=1, cols=3)
    image.save("outputs/CN_openvino" + str(i) + ".png")

with open("CN_openvino.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")