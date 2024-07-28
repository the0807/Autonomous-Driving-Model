<div align="center">

# 2D Semantic Segmentation

### Semantic Segmentation with Detectron2

</div>

# 🖥️ Test Environment
<div align="center">

|OS|GPU|CPU|RAM|CUDA|PyTorch|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Ubuntu 20.04|Nvidia Quadro <br>RTX 5000|Intel(R) Xeon(R)<br> W-2245 CPU @ 3.90GHz|128GB|12.1|2.1|

</div>

# ✏️ Prepare
### 1. Git clone Detectron2
```shell
git clone https://github.com/facebookresearch/detectron2.git
```

### 2. Install python requirements

```shell
pip install -r requirements.txt
```

### 3. Check GPU available

```shell
python gpu_test.py
```

> [!Important]
> - Install compatible `PyTorch` in the `CUDA` version
> 
>     🚀 [`PyTorch Version Check`](https://pytorch.org/get-started/previous-versions/)

### 3. Prepare dataset
    
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

### 2. Make config file to train
```shell
# change options here(batch_size, learning rate)
python config.py
```

### 3. Train
```shell
python train.py --num-gpus 1
```

### 4. Draw graph
```shell
# Saved in path 'output/metrics.png'
python metrics_plot.py
```

### 5. Inference
run code `inference.ipynb`

### 6. Evaluation
```shell
python validation.py
```

# ⚡️ Result
