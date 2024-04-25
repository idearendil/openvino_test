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

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32, use_safetensors=False)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=False
)

pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.unet = torch.compile(pipe.unet, backend="aot_eager")
pipe.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
pipe = pipe.to("cpu")
# pipe.enable_model_cpu_offload()

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(torch.device("cpu"))
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

prompt_lst = [
    "A warship on fire, best quality, vivid, sharp, clear, detailed, vibrant, polished, sophisticated, balanced, stunning, dynamic, realistic",
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
        pose_image_lst.append(Image.fromarray(prediction))

time_lst = []
output = pipe(prompt_lst[0], image=pose_image_lst[0], num_inference_steps=50, negative_prompt=negative_prompt).images[0]  # for cache

for i in range(10):
    generator = torch.Generator("cpu").manual_seed(i)
    start = time.perf_counter()
    output = pipe(prompt_lst[i % 1], image=pose_image_lst[i % 1], num_inference_steps=50, negative_prompt=negative_prompt, generator=generator).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image = make_image_grid([original_image_lst[i % 1], pose_image_lst[i % 1], output], rows=1, cols=3)
    image.save("outputs/CN_cpu_depth" + str(i) + ".png")

with open("CN_cpu_depth.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")