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

IMAGE_PATH = "datasets/coco/img_for_test/000000000016.jpg"
DET_MODEL_NAME = "yolov8s"

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

det_model_path = Path(f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)

core = Core()
det_ov_model = core.read_model(det_model_path)
device = "CPU"
det_compiled_model = core.compile_model(det_ov_model, device)

input_image = np.array(Image.open(IMAGE_PATH))
detections = detect(input_image, det_compiled_model)[0]
image_with_boxes = draw_results(detections, input_image, label_map)

image = Image.fromarray(image_with_boxes)
image.save("outputs/YOLOv8_openvino" + str(1) + ".png")

validation_results = det_model.val(
    data='datasets/coco.yaml',
    device='cpu')



args = get_cfg(cfg=DEFAULT_CFG)
args.data = str("datasets/coco.yaml")

det_validator = det_model.ValidatorClass(args=args)

det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

det_validator.is_coco = True
# det_validator.class_map = ops.coco80_to_coco91_class()
det_validator.names = det_model.model.names
det_validator.metrics.names = det_validator.names
det_validator.nc = det_model.model.model[-1].nc

fp_det_stats = test(det_ov_model, core, tqdm(det_data_loader), det_validator)
print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())





quantization_dataset = nncf.Dataset(det_data_loader, transform_fn)

ignored_scope = nncf.IgnoredScope(
    types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
    names=[
        "/model.22/dfl/conv/Conv",           # in the post-processing subgraph
        "/model.22/Add",
        "/model.22/Add_1",
        "/model.22/Add_2",
        "/model.22/Add_3",
        "/model.22/Add_4",
        "/model.22/Add_5",
        "/model.22/Add_6",
        "/model.22/Add_7",
        "/model.22/Add_8",
        "/model.22/Add_9",
        "/model.22/Add_10"
    ]
)

quantized_det_model = nncf.quantize(
    det_ov_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    ignored_scope=ignored_scope
)

from openvino.runtime import serialize
int8_model_det_path = Path(f'{DET_MODEL_NAME}_openvino_int8_model/{DET_MODEL_NAME}.xml')
print(f"Quantized detection model will be saved to {int8_model_det_path}")
serialize(quantized_det_model, str(int8_model_det_path))

quantized_det_compiled_model = core.compile_model(quantized_det_model, device)
input_image = np.array(Image.open(IMAGE_PATH))
detections = detect(input_image, quantized_det_compiled_model)[0]
image_with_boxes = draw_results(detections, input_image, label_map)

iamge = Image.fromarray(image_with_boxes)
image.save("outputs/YOLOv8_openvino" + str(2) + ".png")


det_validator = det_model.ValidatorClass(args=args)

det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

det_validator.is_coco = True
# det_validator.class_map = ops.coco80_to_coco91_class()
det_validator.names = det_model.model.names
det_validator.metrics.names = det_validator.names
det_validator.nc = det_model.model.model[-1].nc 

fp_det_stats = test(quantized_det_model, core, tqdm(det_data_loader), det_validator)
print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())