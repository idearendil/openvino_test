from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import torch
import numpy as np
from openvino.runtime import Core, Model
from YOLOv8_utils import detect, draw_results, test, print_stats
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from tqdm import tqdm
from ultralytics.yolo.utils import ops
import nncf
import os
import time

IMAGE_PATH = "datasets/coco/img_for_test"
DET_MODEL_NAME = "yolov8s"

images = os.listdir(IMAGE_PATH)

det_model = YOLO(f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names

image_lst = [Image.open(IMAGE_PATH + '/' + an_image) for an_image in images]
time_lst = []

for i, image in enumerate(image_lst):
    res = det_model(image)
    image = Image.fromarray(res[0].plot()[:, :, ::-1])
    image.save("outputs/YOLOv8_cpu" + str(i) + ".png")
    # for cache

for _ in range(10):
    start = time.perf_counter()
    for image in image_lst:
        res = det_model(image)
    end = time.perf_counter()
    time_lst.append(end - start)

with open("YOLOv8_cpu.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")