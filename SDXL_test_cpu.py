import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, PNDMScheduler, LMSDiscreteScheduler
from transformers import CLIPTokenizer
import time

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
from diffusers import DiffusionPipeline
import torch

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True
)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    use_safetensors=True,
)


base.unet = torch.compile(base.unet, backend="aot_eager")
refiner.unet = torch.compile(refiner.unet, backend="aot_eager")
base.to("cpu")
refiner.to("cpu")

prompt = "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k"
neg_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"

time_lst = []
image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=40,
    denoising_start=0.8,
    image=image,
).images[0] # for compile

for i in range(10):
    generator = torch.Generator("cpu").manual_seed(i) 

    start = time.perf_counter()
    image = base(
        prompt=prompt,
        generator=generator,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        generator=generator,
        num_inference_steps=40,
        denoising_start=0.8,
        image=image,
    ).images[0]
    end = time.perf_counter()
    time_lst.append(end - start)

    image.save("outputs/SDXL_cpu" + str(i) + ".png")

with open("SDXL_cpu.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")
