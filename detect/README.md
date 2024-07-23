<div align="center">

# 2D Object Detection

### YOLOv8 based Object Detection

</div>

# ✏️ Prepare
### 1. Install python requirements

```shell
pip install -r requirements.txt
```

### 2. Check GPU available

```shell
python gpu_test.py
```

> [!Important]
> - Install compatible `PyTorch` in the `CUDA` version
> 
>     🚀 [`PyTorch Version Check`](https://pytorch.org/get-started/previous-versions/)

### 3. Prepare dataset
    
한국교통안전공단 - 자율주행 공개데이터셋(2DBB)

``` shell
dataset
│
│   # 40GB
└── 2DBB
    │
    │   # 32GB
    ├── training
    │   ├── images
    │   └── labels
    │
    │   # 4GB
    ├── validation
    │   ├── images
    │   └── labels
    │
    │   # 4GB
    └── test
        ├── images
        └── labels
```

# 📚 Usage
### 1. Preprocessing Data
run code `data_preprocessing.ipynb`

### 2. Train
```shell
python train.py
```

### 3. Draw graph
```shell
# Saved in path 'runs/detect/train/result.png'
python result_plot.py
```

### 4. Inference
run code `inference.ipynb`

### 5. Evaluation
```shell
python validation.py
```
