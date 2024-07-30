# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class seg2DSSDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            "road", 
            "sidewalk", 
            "road roughness", 
            "road boundaries", 
            "crosswalks", 
            "lane", 
            "road color guide", 
            "road marking", 
            "parking", 
            "traffic sign", 
            "traffic light", 
            "pole/structural object", 
            "building", 
            "tunnel", 
            "bridge", 
            "pedestrian", 
            "vehicle", 
            "bicycle", 
            "motorcycle", 
            "personal mobility", 
            "dynamic", 
            "vegetation", 
            "sky", 
            "static"),
        palette=[
            [128, 128, 128],  # road (중간 회색)
            [200, 200, 200],  # sidewalk (밝은 회색)
            [105, 105, 105],  # road roughness (짙은 회색)
            [255, 255, 255],  # road boundaries (흰색)
            [255, 255, 255],  # crosswalks (흰색)
            [255, 255, 255],  # lane (흰색)
            [176, 224, 230],  # road color guide (연한 하늘색)
            [255, 140, 0],    # road marking (주황색)
            [0, 0, 128],      # parking (어두운 파랑)
            [0, 255, 0],      # traffic sign (녹색)
            [0, 0, 0],        # traffic light (검정색)
            [105, 105, 105],  # pole/structural object (짙은 회색)
            [211, 211, 211],  # building (연회색)
            [210, 210, 210],  # tunnel (밝은 회색)
            [220, 220, 220],  # bridge (밝은 회색)
            [0, 0, 255],      # pedestrian (파랑)
            [0, 0, 255],      # vehicle (파랑)
            [255, 0, 0],      # bicycle (빨강)
            [255, 165, 0],    # motorcycle (오렌지)
            [0, 255, 255],    # personal mobility (청록색)
            [255, 140, 0],    # dynamic (주황색)
            [0, 255, 0],      # vegetation (녹색)
            [135, 206, 250],  # sky (하늘색)
            [0, 0, 0]         # static (검정색)
        ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)