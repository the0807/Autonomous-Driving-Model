<div align="center">

# 2D Object Detection

### YOLOv8 based object detection

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
└── 2DBB
    │
    ├── training
    │   ├── images
    │   └── labels
    │
    ├── validation
    │   ├── images
    │   └── labels
    │
    └── test
        ├── images
        └── labels
```

# 📚 Usage
### 1. Preprocessing Data
run code `data_preprocessing.ipynb`

### 2. Train data

```shell
python train.py
```