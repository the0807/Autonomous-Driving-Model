import os
import yaml

from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("2DSS_train",)
cfg.DATASETS.TEST = ("2DSS_val",)
cfg.MODEL.DEVICE = "cuda"
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.STEPS = (240000, 280000)
cfg.SOLVER.MAX_ITER = 300000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 24
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 24
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
with open(cfg.OUTPUT_DIR+'/config.yaml', 'w') as config:
    yaml.dump(cfg, config)
    config.close()
