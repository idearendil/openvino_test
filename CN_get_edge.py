import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "lllyasviel/control_v11p_sd15_canny"

image = load_image("test_edge_src.jpg")

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)

control_image.save("test_edge.png")

# controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )

# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

# generator = torch.manual_seed(33)
# image = pipe("a blue paradise bird in the jungle", num_inference_steps=20, generator=generator, image=control_image).images[0]

# image.save('images/image_out.png')
