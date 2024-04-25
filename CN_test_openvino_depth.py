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
from CN_get_pose import visualize_pose_results
import time
from controlnet_aux import OpenposeDetector
import matplotlib.pyplot as plt
import requests

model_id = "runwayml/stable-diffusion-v1-5"
CONTROLNET_OV_PATH = Path('./cn_openvino_depth/controlnet.xml')
UNET_OV_PATH = Path('./cn_openvino_depth/unet_controlnet.xml')
TEXT_ENCODER_OV_PATH = Path('./cn_openvino_depth/text_encoder.xml')
VAE_DECODER_OV_PATH = Path('./cn_openvino_depth/vae_decoder.xml')

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32, use_safetensors=False)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=False
)

core = ov.Core()

a, b = controlnet_conversion(controlnet)
unet_conversion(pipe, a, b)
text_encoder_conversion(pipe.text_encoder)
vae_decoder_conversion(pipe.vae)

scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

ov_pipe = OVContrlNetStableDiffusionPipeline(tokenizer, scheduler, core, CONTROLNET_OV_PATH, TEXT_ENCODER_OV_PATH, UNET_OV_PATH, VAE_DECODER_OV_PATH, device='CPU')

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(torch.device("cpu"))
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

prompt_lst = [
    "A warship on fire, best quality, vivid, sharp, clear, detailed, vibrant, rich, polished, sophisticated, balanced, stunning, dynamic, captivating, realistic",
]
negative_prompt = "monochrome, lowres, worst quality, low quality, blurry, fuzzy, grainy, pixelated, distorted, dull, flat, muddy, washed-out, low-resolution, unfocused, hazy, rough, jagged, patchy, overexposed, underexposed, noisy, smeared, unrefined"

original_image_lst = [
    load_image("test_depth_src.jpg"),
]

pose_image_lst = []
for i in range(1):
    with torch.no_grad():
        prediction = midas(midas_transforms.dpt_transform(np.array(original_image_lst[i])))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=np.array(original_image_lst[i]).shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        prediction -= np.min(prediction)
        prediction = np.stack([prediction, prediction, prediction], axis=2)
        pose_image_lst.append(Image.fromarray(prediction.astype(np.uint8)))

time_lst = []
result = ov_pipe(prompt_lst[0], pose_image_lst[0], num_inference_steps=50, negative_prompt=negative_prompt)  # for cache

for i in range(10):
    np.random.seed(i)

    start = time.perf_counter()
    result = ov_pipe(prompt_lst[i % 1], pose_image_lst[i % 1], num_inference_steps=50, negative_prompt=negative_prompt)
    end = time.perf_counter()
    time_lst.append(end - start)

    image = make_image_grid([original_image_lst[i % 1], pose_image_lst[i % 1], result[0]], rows=1, cols=3)
    image.save("outputs/CN_openvino_depth" + str(i) + ".png")

with open("CN_openvino_depth.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")