from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, LMSDiscreteScheduler
import torch
from transformers import CLIPTokenizer
import time

model_id = "stabilityai/stable-diffusion-2-1-base"

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

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet = torch.compile(pipe.unet, backend="aot_eager")
pipe.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

prompt = ["a black cat", "a yellow bird"]

time_lst = []
output = pipe(prompt[0], image=canny_image[0], num_inference_steps=20).images[0]  # for cache

for i in range(10):
    idx = i % 2
    generator = torch.Generator("cuda").manual_seed(i)

    start = time.perf_counter()
    output = pipe(prompt[idx], image=canny_image[idx], generator=generator, num_inference_steps=20).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image = make_image_grid([original_image[idx], canny_image[idx], output], rows=1, cols=3)
    image.save("outputs/CN_gpu" + str(i) + ".png")

with open("CN_gpu.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")