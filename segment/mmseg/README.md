<div align="center">

# 2D Semantic Segmentation

### Semantic Segmentation with MMSegmentation

</div>

# 🖥️ Test Environment
<div align="center">

|OS|GPU|CPU|RAM|CUDA|PyTorch|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Ubuntu 20.04|Nvidia Quadro <br>RTX 5000|Intel(R) Xeon(R)<br> W-2245 CPU @ 3.90GHz|128GB|12.1|2.1|

</div>

# ✏️ Prepare
### 1. Install python requirements

```shell
pip install -r requirements.txt
```

### 2. Install MMCV using MIM
```shell
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
```

### 3. Install MMSegmentation
```shell
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

#### Verifying MMSegmentation Installation
```shell
# Step 1
cd mmsegmentation
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .

# Step 2
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg

# check "result.jpg"
```

### 4. Check GPU available

```shell
python gpu_test.py
```

> [!Important]
> - Install compatible `PyTorch` in the `CUDA` version
> 
>     🚀 [`PyTorch Version Check`](https://pytorch.org/get-started/previous-versions/)

### 5. Prepare dataset
    
한국교통안전공단 - 자율주행 공개데이터셋(2DSS)

``` shell
dataset
│
│   # 7.23GB
└── 2DSS
    │
    │   # 5.77GB
    ├── training
    │   ├── images
    │   └── labels
    │
    │   # 0.73GB
    ├── validation
    │   ├── images
    │   └── labels
    │
    │   # 0.73GB
    └── test
        ├── images
        └── labels
```

# 📚 Usage
### 1. Preprocessing Data
run code `data_preprocessing.ipynb`

### 2. Register Dataset
🚀 [`MMSegmentation Docs - add datasets`](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/add_datasets.html)

### 3. Modify config file
```shell
configs
│
│   # dataset setting
├── datasets/
│
│   # model setting
├── models/
│
│   # train schedule setting
├── schedules/
│
│   # runtime setting
├── default_runtime.py
│
│   # main config setting
└── deeplabv3plus.py
```

> [!Important]
> - officially supported Models
> 
>     🚀 [`Model Zoo`](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/model_zoo.md)
>
>     🚀 [`Model configs`](https://github.com/open-mmlab/mmsegmentation/tree/main/configs)
>
> - About configs
>
>     🚀 [`MMSegmentation Docs - learn about configs`](https://mmsegmentation.readthedocs.io/en/main/user_guides/1_config.html)

> [!Tip]
> - If you change the model in 'configs/models', make sure that `num_classes` match the number of classes in your datasets

### 4. Train
```shell
# Single GPU
python train.py 'configs/deeplabv3plus.py'

# Multiple GPU
bash dist_train.sh 'configs/deeplabv3plus.py' 2
```

### 5. Draw graph
```shell
# Plot loss metric
python metrics_plot.py 'path/to/trainlog.json' --out 'path/to/trainlog/loss_plots.png' --keys loss --legend loss

# Plot the mIoU, mAcc, aAcc metrics
python metrics_plot.py 'path/to/trainlog.json' --out 'path/to/trainlog/train_plots.png' --keys mIoU mAcc aAcc --legend mIoU mAcc aAcc
```

### 6. Inference
run code `inference.ipynb`

### 7. Evaluation
```shell
python validation.py 'configs/pspnet.py' 'path/to/trained_model.pth'
```

# ⚡️ Result
### ⭐ DeepLabV3
<div align="center">



</div>

### ⭐ PSPNet
<div align="center">

<img width="1855" alt="스크린샷 2024-07-31 오전 11 39 26" src="https://github.com/user-attachments/assets/2391a756-a854-47be-820e-119046c3a3f4">

<img width="900" alt="스크린샷 2024-07-31 오전 11 48 49" src="https://github.com/user-attachments/assets/c250131b-5d86-45fd-811f-bad00154389f">

![loss_plots](https://github.com/user-attachments/assets/f3dab1dc-d53e-419a-b8c6-f8b6c6d75473)

![train_plots](https://github.com/user-attachments/assets/b4f7c98f-1681-4e6e-a6cc-0f591be206f4)

</div>