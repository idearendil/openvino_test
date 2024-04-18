import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, PNDMScheduler, LMSDiscreteScheduler
from transformers import CLIPTokenizer
import time

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

base = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, variant="fp32")
base.unet = torch.compile(base.unet, backend="aot_eager")
base = base.to("cpu")

# prompt = "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k"
prompt = "a busy intersection in a city seen on the road, realistic, sunny day, 4k, 8k"
neg_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"

time_lst = []
image = base(prompt, negative_prompt=neg_prompt, num_inference_steps=40).images[0]  # for cache

for i in range(10):
    generator = torch.Generator("cpu").manual_seed(i) 

    start = time.perf_counter()
    image = base(prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=40).images[0]  # for cache
    end = time.perf_counter()
    time_lst.append(end - start)

    image.save("outputs/SDXL_cpu" + str(i) + ".png")

with open("SDXL_cpu.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")
