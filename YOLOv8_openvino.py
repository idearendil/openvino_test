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

def transform_fn(data_item):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
       data_item: Dict with data item produced by DataLoader during iteration
    Returns:
        input_tensor: Input data for quantization
    """
    input_tensor = det_validator.preprocess(data_item)['img'].numpy()
    return input_tensor

det_model = YOLO(f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names

det_model_path = Path(f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)

core = Core()
det_ov_model = core.read_model(det_model_path)
device = "CPU"
det_compiled_model = core.compile_model(det_ov_model, device)

image_lst = [Image.open(IMAGE_PATH + '/' + an_image) for an_image in images]
time_lst = []

for i, image in enumerate(image_lst):
    input_image = np.array(image)
    detections = detect(input_image, det_compiled_model)[0]
    image_with_boxes = draw_results(detections, input_image, label_map)

    image = Image.fromarray(image_with_boxes)
    image.save("outputs/YOLOv8_openvino" + str(i) + ".png")
    # for cache

for _ in range(10):
    start = time.perf_counter()
    for image in image_lst:
        input_image = np.array(image)
        detections = detect(input_image, det_compiled_model)[0]
    end = time.perf_counter()
    time_lst.append(end - start)

with open("YOLOv8_openvino.csv", "a") as file:
    for i in range(10):
        file.write(str(time_lst[i]) + ',')
    file.write(str(sum(time_lst) / len(time_lst)) + "\n")