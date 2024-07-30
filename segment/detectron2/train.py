import os
import copy
import torch
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_path = 'dataset/Preprocessed_2DSS/training/coco'
val_path = 'dataset/Preprocessed_2DSS/validation/coco'
test_path = 'dataset/Preprocessed_2DSS/test/coco'

train_json_path = os.path.join(train_path, 'training_coco.json')
val_json_path = os.path.join(val_path, 'validation_coco.json')
test_json_path = os.path.join(test_path, 'test_coco.json')

register_coco_instances("2DSS_train", {}, train_json_path, train_path)
register_coco_instances("2DSS_val", {}, val_json_path, val_path)
register_coco_instances("2DSS_test", {}, test_json_path, test_path)

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    annos = dataset_dict.get("annotations", [])
    
    transform_list = [
        T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice"),
        T.RandomApply(T.RandomCrop("relative", (0.8, 0.8)), prob=0.3),
        T.RandomApply(T.RandomRotation(angle=[-10, 10]), prob=0.5),
        T.RandomApply(T.RandomBrightness(0.7, 1.3), prob=0.5),
        T.RandomApply(T.RandomContrast(0.8, 1.2), prob=0.5),
        T.RandomApply(T.RandomSaturation(0.8, 1.2), prob=0.5),
        T.RandomApply(T.RandomLighting(0.1), prob=0.5),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    
    return dataset_dict

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file('output/config.yaml')
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
