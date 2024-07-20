<div align="center">

# 2D Semantic Segmentation

### Semantic Segmentation with Detectron2

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
└── 2DSS
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

### 2. Make config file to train
```shell
# change options here(batch_size, learning rate)
python config.py
```

### 3. Train
```shell
python train.py
```

### 4. Draw graph
```shell
# Saved in path 'output/result_plot.png'
python loss_graph.py
```

### 5. Inference
run code `inference.ipynb`

### 6. Evaluation
```shell
python validation.py
```