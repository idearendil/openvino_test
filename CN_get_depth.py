from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(torch.device("cpu"))
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

img = load_image("test_pose_src2.jpg")
img = np.array(img)

input_batch = midas_transforms.dpt_transform(img)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
output = prediction.cpu().numpy()

control_image = Image.fromarray(output)

control_image = control_image.convert('RGB')
control_image.save("test_depth.jpg")