<div align="center">

# 2D Object Detection

### YOLOv8 based Object Detection

</div>

# 🖥️ Test Environment
<div align="center">

|OS|GPU|CPU|RAM|CUDA|PyTorch|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Ubuntu 20.04|Nvidia Quadro <br>RTX 5000 x 2|Intel(R) Xeon(R)<br> W-2245 CPU @ 3.90GHz|128GB|12.1|2.1|

</div>

> [!Tip]
> - You can download trained model from the [release](https://github.com/the0807/Autonomous-Driving-Model/releases/tag/v1.0)

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
│   # 40.98GB
└── 2DBB
    │
    │   # 32.78GB
    ├── training
    │   ├── images
    │   └── labels
    │
    │   # 4.1GB
    ├── validation
    │   ├── images
    │   └── labels
    │
    │   # 4.1GB
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

# ⚡️ Result
<div align="center">

<img width="1564" alt="predict" src="https://github.com/user-attachments/assets/4a8d1f3f-4206-49cb-85c9-d22e3458f603">

<img width="800" alt="mAP" src="https://github.com/user-attachments/assets/1573444b-3ab4-4013-a7bf-fb01fb4d15f1">

![confusion_matrix_normalized](https://github.com/user-attachments/assets/81650b69-3f2a-44bb-90fe-4282790abe6f)

![results](https://github.com/user-attachments/assets/c7d7732a-b5ab-4a75-b451-2425b89c79f6)

</div>