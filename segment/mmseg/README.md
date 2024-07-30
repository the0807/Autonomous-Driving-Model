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
🚀 [`MMSegmentation Docs`](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/add_datasets.html)

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
└── pspnet_r50-d8_4xb2-40k_2DSS-512x1024.py
```

### 4. Train
```shell
python train.py --config 'configs/pspnet.py'
```

### 5. Draw graph
```shell
# Saved in path 'output/metrics.png'
python metrics_plot.py
```

### 6. Inference
run code `inference.ipynb`

### 7. Evaluation
```shell
python validation.py
```

# ⚡️ Result
