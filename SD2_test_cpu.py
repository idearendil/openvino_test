import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, PNDMScheduler, LMSDiscreteScheduler
from transformers import CLIPTokenizer
import time

model_id = "stabilityai/stable-diffusion-2-1-base"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.unet = torch.compile(pipe.unet, backend="aot_eager")
pipe.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
pipe = pipe.to("cpu")

prompt = "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k"
neg_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"

time_lst = []
image = pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=50).images[0]  # for cache

for i in range(10):
    generator = torch.Generator("cpu").manual_seed(i) 

    start = time.perf_counter()
    image = pipe(prompt, generator=generator, negative_prompt=neg_prompt, num_inference_steps=50).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image.save("outputs/SD2_cpu" + str(i) + ".png")

with open("SD2_cpu.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")
