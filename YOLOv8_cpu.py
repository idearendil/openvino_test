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
from ultralytics.yolo.utils.metrics import ConfusionMatrix

IMAGE_PATH = "datasets/coco/img_for_test/000000000016.jpg"
DET_MODEL_NAME = "yolov8s"

args = get_cfg(cfg=DEFAULT_CFG)
args.data = str("datasets/coco.yaml")

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

res = det_model(IMAGE_PATH)
image = Image.fromarray(res[0].plot()[:, :, ::-1])
image.save("outputs/YOLOv8_openvino" + str(0) + ".png")

det_validator = det_model.ValidatorClass(args=args)

det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

det_validator.is_coco = True
# det_validator.class_map = ops.coco80_to_coco91_class()
det_validator.names = det_model.model.names
det_validator.metrics.names = det_validator.names
det_validator.nc = det_model.model.model[-1].nc 

det_validator.seen = 0
det_validator.jdict = []
det_validator.stats = []
det_validator.batch_i = 1
det_validator.confusion_matrix = ConfusionMatrix(nc=det_validator.nc)

for batch_i, batch in enumerate(det_data_loader):
    batch = det_validator.preprocess(batch)
    results = det_model(batch["img"])
    preds = [torch.from_numpy(results[det_model.model.output(0)]), torch.from_numpy(results[det_model.model.output(1)])]
    preds = det_validator.postprocess(preds)
    det_validator.update_metrics(preds, batch)
stats = det_validator.get_stats()

print_stats(stats, det_validator.seen, det_validator.nt_per_class.sum())
