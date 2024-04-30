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

controlnet1 = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float32, use_safetensors=False)
controlnet2 = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32, use_safetensors=False)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=(controlnet1, controlnet2), torch_dtype=torch.float32, use_safetensors=False
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

original_image_lst = [
    load_image("test_depth_src.jpg"),
    load_image("test_edge_src.jpg"),
    load_image("test_edge_src2.jpg"),
    load_image("test_edge_src3.jpg"),
    load_image("test_edge_src4.jpg"),
    load_image("test_edge_src5.jpg")
]

prompt_lst = [
    "A warship on fire, best quality, vivid, sharp, clear, detailed, vibrant, polished, sophisticated, balanced, stunning, dynamic, realistic",
    "A blue truck on the road next to forest, best quality, vivid, sharp, clear, detailed, vibrant, rich, polished, sophisticated, balanced, stunning, dynamic, captivating, realistic",
    "An American politician, white shirt, black hair, best quality, vivid, sharp, clear, detailed, rich, sophisticated, balanced, stunning, dynamic, captivating, atmospheric",
    "A street in a US city at night, some crosswalks, best quality, clear, detailed, rich, dark road, polished, sophisticated, balanced, realistic, likely",
    "A military tank on a snowy field, best quality, extremely detailed, vivid, sharp, clear, vibrant, rich, polished, sophisticated, balanced, dynamic, realistic",
    "A space fighter flying fast in the space, black and stars background, best quality, sharp, clear, detailed, rich, polished, sophisticated, balanced, stunning, dynamic, captivating, dreamy"
]
negative_prompt = "monochrome, lowres, worst quality, low quality, blurry, fuzzy, grainy, pixelated, distorted, dull, flat, muddy, washed-out, low-resolution, unfocused, hazy, rough, jagged, patchy, overexposed, underexposed, noisy, smeared, unrefined"

pose_image_lst1 = []
for i in range(6):
    image = cv2.Canny(np.array(original_image_lst[i]), 100, 200)[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    pose_image_lst1.append(Image.fromarray(image))

pose_image_lst2 = []
for i in range(6):
    with torch.no_grad():
        prediction = midas(midas_transforms.dpt_transform(np.array(original_image_lst[i])))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=np.array(original_image_lst[i]).shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        prediction -= np.min(prediction)
        pose_image_lst2.append(Image.fromarray(prediction))

time_lst = []
output = pipe(prompt_lst[0], image=[pose_image_lst1[0], pose_image_lst2[0]], num_inference_steps=50, negative_prompt=negative_prompt).images[0]  # for cache

for i in range(24):
    generator = torch.Generator("cpu").manual_seed(i)
    start = time.perf_counter()
    output = pipe(prompt_lst[i % 6], image=[pose_image_lst1[i % 6], pose_image_lst2[i % 6]], num_inference_steps=50, negative_prompt=negative_prompt, generator=generator).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image = make_image_grid([original_image_lst[i % 6], pose_image_lst1[i % 6], pose_image_lst2[i % 6], output], rows=1, cols=4)
    image.save("outputs/CN_cpu_edge_depth" + str(i) + ".png")

with open("CN_cpu_edge_depth.csv", "a") as file:
    for i in range(24):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")