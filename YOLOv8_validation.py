from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from tqdm.notebook import tqdm
from pathlib import Path

args = get_cfg(cfg=DEFAULT_CFG)
args.data = str("coco.yaml")

det_validator = det_model.ValidatorClass(args=args)

det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

det_validator.is_coco = True
det_validator.class_map = ops.coco80_to_coco91_class()
det_validator.names = det_model.model.names
det_validator.metrics.names = det_validator.names
det_validator.nc = det_model.model.model[-1].nc

fp_det_stats = test(det_ov_model, core, tqdm(det_data_loader), det_validator)

print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())