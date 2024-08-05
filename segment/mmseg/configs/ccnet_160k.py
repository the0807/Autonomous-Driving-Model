_base_ = [
    'models/ccnet_r50-d8.py', 'datasets/seg2DSS.py',
    'default_runtime.py', 'schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)